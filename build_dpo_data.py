import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import time
import logging
import json
import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from prompt.openai_access import batch_get_chat_api, get_oai_completion
from prompt.prompt_design import createCompareThinkPrompt,createComplexQuestionPrompt,createDetailedCompareThinkPrompt
from util.config import TRAINING_DATA_PATH, OUTPUT_PATH, TRAINING_DATA_PATH_AIME,MATH_DATA_PATH
from data.data_loader import load_simplify_problems
from util.util import reject_sample
import openai
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import util, set_logger
import random

def build_battle_matrix(raw_results, N):
    win1st = 0
    win2nd = 0
    
    # 第一次统计总 win1st, win2nd
    for r in raw_results:
        resA = r['resA']  # prompt: m->n
        resB = r['resB']  # prompt: n->m
        
        # A 顺序
        if resA == +1:
            win1st += 1
        elif resA == -1:
            win2nd += 1
        
        # B 顺序
        if resB == +1:
            win1st += 1
        elif resB == -1:
            win2nd += 1
    
    total_win = win1st + win2nd
    if total_win == 0:
        omega_1, omega_2 = 0.5, 0.5
    else:
        omega_1 = float(win2nd) / total_win
        omega_2 = float(win1st) / total_win
    
    B = np.zeros((N, N), dtype=float)
    
    # 第二次遍历，将结果写入 B
    for r in raw_results:
        m, n = r['j1'], r['j2']
        resA = r['resA']
        resB = r['resB']
        # #m < n
        # # resA=+1 => m 胜 (顺序 m->n)
        if resA == +1:
            B[m, n] += omega_1
        elif resA == -1:
            B[n, m] +=omega_1
        
        # resB=-1 => m 胜 (顺序 n->m)
        if resB == -1:
            B[m, n] += omega_2
        elif resB == +1:
            B[n, m] += omega_2
    
    return B, omega_1, omega_2

def solve_bradley_terry_torch(B, steps=100, lr=1e-5, device='cpu'):
    N = B.shape[0]
    B_t = torch.tensor(B, dtype=torch.float32, device=device)
    scores = torch.zeros(N, dtype=torch.float32, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([scores], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        
        diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # shape (N, N)
        p_mn = torch.sigmoid(diff)
        
        ll_matrix = B_t * torch.log(p_mn + 1e-12)
        log_likelihood = torch.sum(ll_matrix)
        
        loss = -log_likelihood
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            mean_val = torch.mean(scores)
            scores -= mean_val
    
    return scores.detach()

def build_dpo_data(prompt,responses,raw_results,N):
    B, _,_ = build_battle_matrix(raw_results, N)
    print(B)    
    scores = solve_bradley_terry_torch(B, device="cuda",steps=100, lr=1e-2)
    max_index = torch.argmax(scores).item()  # 返回最大值的索引
    min_index = torch.argmin(scores).item()  # 返回最小值的索引
    return {
        "conversations": [
        {
            "from": "human",
            "value": prompt
        }
        ],
        "chosen": {
            "from": "gpt",
            "value": responses[max_index]
        },
        "rejected": {
            "from": "gpt",
            "value": responses[min_index]
        }
    }
def process_response_text(response_text,logger):
    complex_problem, complex_solution = util.parse_answer(response_text, 
                                                                            [
                                                                            "Complexified Problem", 
                                                                            "Complexified Solution"], 
                                                                            logger=logger)
    if complex_problem and complex_solution:
        return {
                "complex_problem":complex_problem,
                "complex_solution":complex_solution,
                "response":response_text
        }
    else:
        return None
def process_compare(response,logger):
    try:
        if  response:
            if "former one is harder" in response:
                return 1
            elif "later one is harder" in response:
                return -1
            elif "tie" in response:
                return 0
        logger.warning(f"Invalid data for compare.")
        return 0
    except Exception as e:
        logger.error(f"Error in process_compare: {e}")
        return 0 
def process_compare_data(i,j,response1,response2,logger):
    return{
        "j1":i,
        "j2":j,
        "resA":process_compare(response1,logger),
        "resB":process_compare(response2,logger)
    }
def main(N=20,
         model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/NewThink-DeepSeek-R1-Distill-Qwen-32B/full/sft",
         input_path=None,
         out_path="/data/xucaijun/Math-Generator/outputs/test_question_process_deepseek.json",
         put_path="/data/xucaijun/Math-Generator/outputs/testest_question_process_deepseek.json",
         output_path="/data/xucaijun/Math-Generator/outputs/dpo_fliter_complex_question_process_deepseek.json"):
    logger = set_logger.setup_logger()
    logger.info("Starting main processing loop.")
     # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device="cuda",tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
    )
    logger.info("Model loaded successfully.")
    # Define sampling parameters
    stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>"]
    max_tokens=32768
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1,
        stop=stop_words,
        n=N
    )

    # Load problems
    logger.info("Loading problems...")
    if input_path:
        with open(input_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)
    else:
        problems = load_simplify_problems()
    problems= problems[:3]
    logger.info(f"Loaded {len(problems)} problems.")
    input_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": createComplexQuestionPrompt(problem['problem'], problem['solution'])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for problem in problems
    ]
    logger.info(f"generate response....")
    generated_responses = model.generate(input_texts, sampling_params=sampling_params)
    generated_responses = [[process_response_text(generated_response.outputs[i].text,logger) for i in range(N)] for generated_response in generated_responses]
    print("total_response",len(generated_responses))
    print("response_size",len(generated_responses[0]))
    generated_responses = [[response_i for response_i in generated_response if response_i] for generated_response in generated_responses]

    with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(generated_responses, f, ensure_ascii=False, indent=4)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        stop=stop_words,
        n=1
    )
    input_texts=[]
    for idx,problem in tqdm(enumerate(problems)):
        response_list=generated_responses[idx]
        for i in range(len(response_list)):
            # 6个
            complex_problem_i = response_list[i]['complex_problem']
            complex_solution_i = response_list[i]['complex_solution']
            for j in range(i+1,len(response_list)):
                complex_problem_j = response_list[i]['complex_problem']
                complex_solution_j = response_list[i]['complex_solution']
                input_texts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": createDetailedCompareThinkPrompt(complex_problem_i, complex_solution_i, complex_problem_j,complex_solution_j)}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ))
                input_texts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": createDetailedCompareThinkPrompt(complex_problem_j, complex_solution_j, complex_problem_i,complex_solution_i)}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ))
    compare_responses = model.generate(input_texts, sampling_params=sampling_params)
    compare_responses = [compare_response.outputs[0].text for compare_response in compare_responses]
    print("compare_responses:",len(compare_responses))
    print((compare_responses[0]))
    cnt = 0
    dpo_data=[]
    input_texts=[]
    for idx,problem in tqdm(enumerate(problems)):
        response_list=generated_responses[idx]
        print("response_list:",len(response_list))
        input_texts=[]
        raw_data=[]
        for i in range(len(response_list)):
            for j in range(i+1,len(response_list)):
                raw_data.append(process_compare_data(i,j,compare_responses[cnt],compare_responses[cnt+1],logger))
                cnt+=2
        dpo_data.append(build_dpo_data(createComplexQuestionPrompt(problem['problem'], problem['solution']),\
                       [response['response'] for response in response_list],\
                        raw_data,len(response_list)))
    with open(put_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=4)
    with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=4)

def test():
    # 假设有 4 个 judgement
    N = 4
    
    # raw_results 里包含正向(resA)和反向(resB)比较结果
    # 这里给个举例：
    # j1=0, j2=1, resA=+1 表示在 m->n (0->1) 的顺序下, 0 胜; resB=-1 表示在 n->m (1->0) 顺序下, 0 胜
    # 你可以根据真实实验数据填充
    raw_results = [
        {'j1':0, 'j2':1, 'resA': +1, 'resB': -1},  # 0 连赢 1
        {'j1':0, 'j2':2, 'resA': -1, 'resB': +1},  # 2 连赢 0
        {'j1':1, 'j2':2, 'resA': +1, 'resB': -1},  # 1 连赢 2
        {'j1':0, 'j2':3, 'resA': +1, 'resB':  0},  # 0 赢 3，换位置 0 平 3
        {'j1':1, 'j2':3, 'resA':  0, 'resB': -1},  # 1 平 3， 换位置 1 赢 3
        {'j1':2, 'j2':3, 'resA': +1, 'resB': +1},  # 2 赢 3，换位置 3 赢 2
    ]
    
    B, w1, w2 = build_battle_matrix(raw_results, N)
    print("Battle Matrix B:")
    print(B)
    print(f"omega_1={w1:.4f}, omega_2={w2:.4f}")
    
    scores = solve_bradley_terry_torch(B, steps=1000, lr=1e-2)
    print("Final Bradley-Terry scores:")
    for i, sc in enumerate(scores):
        print(f"  Judgement {i}: {sc.item():.4f}")


if __name__ == "__main__":
    main()