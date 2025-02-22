import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
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

# def process_reject_sample(problem, response, logger):
#     def reject_sample_with_timeout():
#         if problem and problem['complex_solution'] and response:
#             return reject_sample(response, problem['complex_solution'])
#         else:
#             logger.warning(f"Missing data for reject sample.")
#             return False

#     try:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(reject_sample_with_timeout)
#             result = future.result(timeout=20)  # 设置超时时间为20秒
#             return result
#     except concurrent.futures.TimeoutError:
#         logger.warning(response)
#         logger.warning("process_reject_sample exceeded the timeout limit of 20 seconds.")
#         return False
#     except Exception as e:
#         logger.error(f"Error in process_reject_sample: {e}")
#         return False
def process_reject_sample(problem, section,response, logger):
    """
    在单独的进程中执行reject_sample相关的操作，
    如果超过设定的超时时间（默认为20秒），直接杀死子进程并返回False
    """

    # 这个内部函数里放需要执行的逻辑，比如调用 reject_sample 及其子函数
    def _worker_func(return_dict, problem, response):
        try:
            if problem and problem.get(section) and response:
                # 如果你还需要传 logger 或其它参数，也可一并加入
                result = reject_sample(response, problem[section])
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
        p.join(timeout=20)

        # 如果子进程还存活，说明超时
        if p.is_alive():
            logger.warning(problem)
            logger.warning(response)
            logger.warning("process_reject_sample exceeded the timeout limit of 20 seconds.")
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
def process_muti_reject_sample(problem,section,responses,correct_limit,logger):
    try:
        if problem and problem.get(section) and responses:
            # 如果你还需要传 logger 或其它参数，也可一并加入
            result = reject_muti_sample(responses,problem['section'],correct_limit)
            return result
        else:
            logger.warning("Missing data for reject sample.")
            return False
    except Exception as e:
        logger.error(f"Error in reject_sample: {e}")
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
def self_filter(model,tokenizer,problems,logger,stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>"],
         max_tokens=32768,
         device="cuda",
        #  model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B/full/sft"
         model_name_or_path="/data/xucaijun/DeepSeek-R1-Distill-Qwen-32B",
         batch_size=512,
         N=5,
         correct_limit=3
         ):

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6,
        stop=stop_words,
        n=N
    )
    random.seed(100)
    random.shuffle(problems)
    problems=problems[:100]
    output_list=[]
    try:
        total_batch=math.ceil(len(problems)/batch_size)
        for batch in range(total_batch):
            # logger.info(f"Start Batch {batch}")
            try_problems=problems[batch*batch_size:(batch+1)*batch_size]
            # reject sample
            input_texts = [
                    tokenizer.apply_chat_template(
                            [{"role": "user", "content": createAnsqerPrompt(problem['complex_problem'])}],
                            tokenize=False,
                            add_generation_prompt=True,
                    )
                    for problem in try_problems
            ]
            # logger.info(input_texts[0])

            logger.info(f"Start reject sample.")
            generated_responses = model.generate(input_texts, sampling_params=sampling_params)
            if N==1:
                generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]
                reject_sampled_problems = [process_think(problem, generated_response) for problem,generated_response in zip(try_problems,generated_responses)  ]
                reject_sampled_problems = [
                    problem for problem, generated_response in tqdm(zip(try_problems, generated_responses), total=len(try_problems), desc="Processing Problems")
                    if process_reject_sample(problem, 'complex_solution',generated_response, logger)
                ]
            else:
                generated_responses = [[generated_response.outputs[i].text for i in range(N)]for generated_response in generated_responses]
                reject_sampled_problems = [
                    problem for problem, generated_response in tqdm(zip(try_problems, generated_responses), total=len(try_problems), desc="Processing Problems")
                    if process_muti_reject_sample(problem, 'complex_solution',generated_response,correct_limit,logger)
                ]
            logger.info(f"{len(reject_sampled_problems)} problems pass reject sample.")
            logger.info(f" {len(try_problems)- len(reject_sampled_problems)} problems fail in reject sample.")

            input_texts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": createCompareThinkPrompt(problem['original_problem'], problem['original_solution'], problem['complex_problem'], problem['complex_solution'])}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for problem in reject_sampled_problems
            ]
            # logger.info(input_texts[0])
            logger.info(f"Start compare.")
            generated_responses = model.generate(input_texts, sampling_params=sampling_params)
            generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]

            #reversed compare
            reversed_input_texts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": createCompareThinkPrompt(problem['complex_problem'], problem['complex_solution'], problem['original_problem'], problem['original_solution'])}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for problem in reject_sampled_problems
            ]
            # logger.info(reversed_input_texts[0])
            logger.info(f"Start reversed compare.")
            reversed_generated_responses = model.generate(reversed_input_texts, sampling_params=sampling_params)
            reversed_generated_responses = [generated_response.outputs[0].text for generated_response in reversed_generated_responses]
                    
            # logger.info(reversed_generated_responses[0])
            compared_problems = [problem for problem,generated_response,reversed_generated_response \
                                        in zip(reject_sampled_problems,generated_responses,reversed_generated_responses) if process_compare(problem,generated_response,reversed_generated_response,logger)]
            logger.info(f" {len(compared_problems)} problems pass compare.")
            logger.info(f" {len(reject_sampled_problems)- len(compared_problems)} problems fail in compare.")

            output_list+=compared_problems
    except Exception as e:
        logger.error(f"Error :{e}")
    return output_list

def main():
    logger = set_logger.setup_logger()
    logger.info("Starting main processing loop.")
    model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/SelfThink-DeepSeek-R1-Distill-Qwen-32B/full/sft"
     # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device="cuda",tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
    )
    logger.info("Model loaded successfully.")
    data_path="/data/xucaijun/Math-Generator/outputs/newprompt_complex_question_process_deepseek.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    output_list=self_filter(model,tokenizer,problems,logger)
    output_path="/data/xucaijun/Math-Generator/outputs/newprompt_fliter_complex_question_process_deepseek_backup.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    main()
