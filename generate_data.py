import time
import json
from util import util, set_logger
import csv
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import random
import math
from tqdm import tqdm
from data.data_loader import load_problems,load_simplify_problems
from prompt.prompt_design import createComplexQuestionProcessPrompt,createComplexQuestionPrompt
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from filter_problems import filter_problems
from process_train_data import process_train_data
from self_filter import self_filter
from collections import defaultdict
from util.util import extract_think_and_after,process_output_data
def process_problem(problem):
    think,solution=extract_think_and_after(problem['output'])
    return {
        "problem":problem['problem'],
        "solution":solution
    }
def main(stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>","\n**Complexification Process**"],
         max_tokens=32768,
         N=3,
         batch_size=-1,
         enable_filter=True,
         use_chat_templete=True,
         device="cuda",
         input_path="/data/xucaijun/New/Math-Generator/deepseek-math/0/math_output_deepseek.json",
         output_path="./outputs/newthink_first_iter_deepseek_answer.json",
         model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/NewThink-DeepSeek-R1-Distill-Qwen-32B/full/sft"):
    logger = set_logger.setup_logger()
    logger.info("Starting the process...")

    # Load problems
    logger.info("Loading problems...")
    if input_path:
        with open(input_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)
            if input_path == "/data/xucaijun/New/Math-Generator/outputs/math_output_deepseek.json":
                problems=[ problem for problem in problems if problem['value']==True]
                problems =[process_problem(problem) for problem in problems ]
            elif input_path == '/data/xucaijun/New/Math-Generator/outputs/first_iter_deepseek_answer.json':
                data_list=[]
                for data in problems:
                    data_list.append({
                        'problem':data[0]['complex_problem'],
                        'solution':data[0]['complex_solution']})
                problems=data_list
    else:
        problems = load_simplify_problems()
    logger.info(f"Loaded {len(problems)} problems.")

    # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device=device,tensor_parallel_size=8,enforce_eager=False,gpu_memory_utilization=0.95)
    if use_chat_templete:
        tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, trust_remote_code=True
        )
    logger.info("Model loaded successfully.")

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        stop=stop_words,
        n=N
    )
    # Apply_chat_template
    if batch_size==-1:
        batch_size=len(problems)
    total_list= []
    total_batch=math.ceil(len(problems)/batch_size)
    for batch in range(total_batch):
        output_list = []
        now_problems = problems[batch*batch_size:(batch+1)*batch_size]
        logger.info(f"Start Batch {batch}")
        logger.info(f"{len(now_problems)} problems for generating data...")
        if use_chat_templete:
            input_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": createComplexQuestionPrompt(problem['problem'], problem['solution'])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for problem in now_problems
            ]
        else:
            input_texts= [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+ \
                createComplexQuestionProcessPrompt(problem['problem'], problem['solution'])+ \
                "<|im_end|>\n<|im_start|>assistant\n"
                for problem in now_problems
            ]
        logger.info(input_texts[0])
                
            
        # Generate responses using vLLM
        logger.info("Generating responses...")
        tmp_responses = model.generate(input_texts, sampling_params=sampling_params)
        tmp_responses = [[tmp_response.outputs[i].text for i in range(N)] for tmp_response in tmp_responses]
        # Process the generated responses
        for problem, responses in zip(now_problems, tmp_responses):
            for response in responses:
                complex_problem, complex_solution = util.parse_answer(response, 
                                                                        [
                                                                        "Complexified Problem", 
                                                                        "Complexified Solution"], 
                                                                        logger=logger)
                output_object = {
                    "original_problem": problem['problem'],
                    "original_solution": problem['solution'],
                    "complex_problem": complex_problem,
                    "complex_solution": complex_solution,
                    "response": response
                }
                output_list.append(output_object)
        if enable_filter:
            output_list=self_filter(model,tokenizer,output_list,logger,batch_size=N*batch_size,N=1)
        output_list=process_output_data(output_list)
        # Save the output to a JSON file
        total_list+=output_list
        logger.info(f"Saving output to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as output_json:
            json.dump(total_list, output_json, ensure_ascii=False, indent=4)

    logger.info("Process completed.")


if __name__ == "__main__":
    main()
