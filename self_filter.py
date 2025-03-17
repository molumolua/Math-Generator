import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import logging
import json
import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from prompt.openai_access import batch_get_chat_api, get_oai_completion
from prompt.prompt_design import createComparePrompt, createSimpleQuestionPromptV3, createAnsqerPrompt,createCompareThinkPrompt
from util.config import TRAINING_DATA_PATH, OUTPUT_PATH, TRAINING_DATA_PATH_AIME,MATH_DATA_PATH
from data.data_loader import load_simplify_problems
from util.util import reject_sample,reject_muti_sample
import openai
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import util, set_logger
import random
import multiprocessing


import concurrent.futures
import time
# def process_reject_sample(problem, section,response, logger):
#     try:
#         if problem and problem.get(section) and response:
#             return reject_sample(response, problem[section])
#         else:
#             logger.warning(f"Missing data for reject sample.")
#             return False
#     except Exception as e:
#         logger.error(f"Error in reject_sample: {e}")
#         return False
def process_reject_sample(problem, section,response, logger,timeout=10):
    """
    在单独的进程中执行reject_sample相关的操作，
    如果超过设定的超时时间（默认为10秒），直接杀死子进程并返回False
    """

    # 这个内部函数里放需要执行的逻辑，比如调用 reject_sample 及其子函数
    def _worker_func(return_dict, problem, response):
        try:
            if problem and problem.get(section) and response:
                # 如果你还需要传 logger 或其它参数，也可一并加入
                result = reject_sample(response, problem[section],timeout=False)
                return_dict['result'] = result
            else:
                logger.warning("Missing data for reject sample.")
                return_dict['result'] = False
        except Exception as e:
            logger.error(f"Error in reject_sample: {e}")
            return_dict['result'] = False

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # 创建子进程
    p = multiprocessing.Process(
        target=_worker_func,
        args=(return_dict, problem, response)
    )

    try:
        # 启动子进程
        p.start()
        # 设置最大等待时间20秒
        p.join(timeout=timeout)

        # 如果子进程还存活，说明超时
        if p.is_alive():
            logger.warning(problem)
            logger.warning(response)
            logger.warning(f"process_reject_sample exceeded the timeout limit of {timeout} seconds.")
            p.terminate()   # 终止子进程
            p.join()        # 回收子进程
            return False

        # 如果没超时就获取执行结果
        result = return_dict.get('result', False)
        return result

    except Exception as e:
        logger.error(f"Exception in process_reject_sample: {e}")
        if p.is_alive():
            p.terminate()
            p.join()
        return False
def process_muti_reject_sample(problem,section,responses,correct_limit,logger,true_reject=True):
    try:
        if problem and problem.get(section) and responses:
            # 如果你还需要传 logger 或其它参数，也可一并加入
            result=0
            for response in responses:
                if process_reject_sample(problem, section,response, logger,timeout=10):
                    result+=1
            problem['correct_num']=result
            if not true_reject:
                return True
            return result>=correct_limit
        else:
            logger.warning("Missing data for reject sample.")
            problem['correct_num']=0
            if not true_reject:
                return True
            return False
    except Exception as e:
        logger.error(f"Error in reject_sample: {e}")
        problem['correct_num']=0
        if not true_reject:
                return True
        return False
def process_compare(problem, response1,response2,logger):
    try:
        value=0
        response1=response1.lower()
        response2=response2.lower()
        if problem and response1 and response2:
            if "former one is harder" in response1:
                value+=1
            elif "later one is harder" in response1:  #简单的应该放前面
                value-=1
            elif "comparable" in response1:
                value=value
            else:
                logger.error(f"Error!{response1}")
            if "later one is harder" in response2:
                value+=1
            elif "former one is harder" in response2:
                value-=1
            elif "comparable" in response2:
                value=value
            else:
                logger.error(f"Error!{response2}")
            # logger.info(f"value:{value}")
            return value<0
        logger.warning(f"Invalid data for compare.")
        return False
    except Exception as e:
        logger.error(f"Error in process_compare: {e}")
        return False
def process_think(problem,response):
    problem['think_solution']=response
    return problem
def show_reject_result(problems,logger):
    cnt =0
    correct_num=0
    for problem in problems:
        cnt+=1
        correct_num+=problem['correct_num']
    logger.info(f"avg correct num is {correct_num/cnt}")
def self_filter(model,tokenizer,problems,logger,stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>"],
         max_tokens=32768,
         device="cuda",
        #  model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B/full/sft"
         model_name_or_path="/data/xucaijun/DeepSeek-R1-Distill-Qwen-32B",
         original_section_names=["original_problem","original_problem"],
         test_section_names=["complex_problem","complex_solution"],
         complex_section_names=["complex_problem","complex_solution"],
         batch_size=512,
         N=5,
         correct_limit=3,
         enable_compare = True,
         true_reject =True
         ):

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.8,
        stop=stop_words,
        n=N
    )

    compare_sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6,
        stop=stop_words,
        n=1
    )
    output_list=[]
    try:
        total_batch=math.ceil(len(problems)/batch_size)
        for batch in range(total_batch):
            # logger.info(f"Start Batch {batch}")
            try_problems=problems[batch*batch_size:(batch+1)*batch_size]

            if enable_compare:
                input_texts = [
                            tokenizer.apply_chat_template(
                                [{"role": "user", "content": createCompareThinkPrompt(problem[original_section_names[0]], problem[original_section_names[1]], problem[complex_section_names[0]], problem[complex_section_names[1]])}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            for problem in try_problems
                ]
                # logger.info(input_texts[0])
                logger.info(f"Start compare.")
                generated_responses = model.generate(input_texts, sampling_params=compare_sampling_params)
                generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]

                #reversed compare
                

                reversed_input_texts = [
                            tokenizer.apply_chat_template(
                                [{"role": "user", "content": createCompareThinkPrompt(problem[complex_section_names[0]], problem[complex_section_names[1]], problem[original_section_names[0]], problem[original_section_names[1]])}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            for problem in try_problems
                ]
                # logger.info(reversed_input_texts[0])
                logger.info(f"Start reversed compare.")
                reversed_generated_responses = model.generate(reversed_input_texts, sampling_params=compare_sampling_params)
                reversed_generated_responses = [generated_response.outputs[0].text for generated_response in reversed_generated_responses]
                        
                # logger.info(reversed_generated_responses[0])
                compared_problems = [problem for problem,generated_response,reversed_generated_response \
                                            in zip(try_problems,generated_responses,reversed_generated_responses) if process_compare(problem,generated_response,reversed_generated_response,logger)]
                logger.info(f" {len(compared_problems)} problems pass compare.")
                logger.info(f" {len(try_problems)- len(compared_problems)} problems fail in compare.")

                compared_problems=compared_problems
            else:
                compared_problems=try_problems

            # reject sample
            input_texts = [
                    tokenizer.apply_chat_template(
                            [{"role": "user", "content": createAnsqerPrompt(problem[test_section_names[0]])}],
                            tokenize=False,
                            add_generation_prompt=True,
                    )
                    for problem in compared_problems
            ]
            # logger.info(input_texts[0])

            logger.info(f"Start reject sample.")
            if N==1:
                generated_responses = model.generate(input_texts, sampling_params=compare_sampling_params)
                generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]
                reject_sampled_problems = [process_think(problem, generated_response) for problem,generated_response in zip(compared_problems,generated_responses)  ]
                reject_sampled_problems = [
                    problem for problem, generated_response in tqdm(zip(compared_problems, generated_responses), total=len(compared_problems), desc="Processing Problems")
                    if process_reject_sample(problem, test_section_names[1],generated_response, logger)
                ]
            else:
                generated_responses = model.generate(input_texts, sampling_params=sampling_params)
                generated_responses = [[generated_response.outputs[i].text for i in range(len(generated_response.outputs))]for generated_response in generated_responses]
                reject_sampled_problems = [
                    problem for problem, generated_response in tqdm(zip(compared_problems, generated_responses), total=len(compared_problems), desc="Processing Problems")
                    if process_muti_reject_sample(problem, test_section_names[1],generated_response,correct_limit,logger,true_reject=true_reject)
                ]
                show_reject_result(reject_sampled_problems,logger)
            logger.info(f"{len(reject_sampled_problems)} problems pass reject sample.")
            logger.info(f" {len(compared_problems)- len(reject_sampled_problems)} problems fail in reject sample.")
            
            
            

            output_list+=reject_sampled_problems
    except Exception as e:
        logger.error(f"Error :{e}")
    return output_list

def test_exist(pattern,problems,section):
    for problem in problems:
        if(problem[section]==pattern):
            return problem
    return None
def main():
    logger = set_logger.setup_logger()
    logger.info("Starting main processing loop.")
    model_name_or_path="/data/xucaijun/DeepSeek-R1-Distill-Qwen-32B"
    model_name_or_path="/data/modelscope/hub/Qwen/Qwen2.5-7B-Instruct"
    model_name_or_path="/data/modelscope/hub/Qwen/Qwen2___5-0___5B-Instruct"

    model_name_or_path="/data/modelscope/hub/Qwen/Qwen2___5-32B-Instruct"


    model_name_or_path="Qwen/Qwen2.5-Math-1.5B-Instruct"
     # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device="cuda",tensor_parallel_size=4,dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
    )
    logger.info("Model loaded successfully.")

    data_path="./outputs/7b-test.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
        if data_path =="./outputs/7b-test.json":
            data_list=[]
            for data in problems:
                for problem in data:
                    if problem['complex_problem'] != problem['original_problem']:
                        data_list.append(problem)
            problems=data_list
    random.seed(0)
    random.shuffle(problems)
    problems=problems

    # data_path_2="/data/xucaijun/New/Math-Generator/outputs/tmp_2.json"
    # with open(data_path_2, 'r', encoding='utf-8') as f:
    #     problems_2 = json.load(f)
    #     data_list=[]
    #     for data in problems_2:
    #         data_list.append(data[0])
    #     problems_2 =data_list
    
    # test_cnt=100
    # test_problems_1=[]
    # test_problems_2=[]
    # for problem in problems:
    #     test_problem=test_exist(problem['original_problem'],problems_2,'original_problem')
    #     if test_problem:
    #         test_problems_1.append(problem)
    #         test_problems_2.append(test_problem)

    #         if len(test_problems_1)>=test_cnt:
    #             break
    

    output_list=self_filter(model,tokenizer,problems,logger,test_section_names=['complex_problem','complex_solution'],original_section_names=['original_problem','original_solution'],complex_section_names=['complex_problem','complex_solution'],\
                            N=10,true_reject=False,enable_compare=False,batch_size=len(problems))
    output_path="./outputs/7b-generate-1.5b-reject.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    main()
