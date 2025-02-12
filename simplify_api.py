import os
import time
import logging
import json
import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import multiprocessing 
from tqdm import tqdm
from prompt.openai_access import batch_get_chat_api, get_oai_completion
from prompt.prompt_design import createCompareThinkPrompt, createThinkSimpleQuestionPrompt, createAnsqerPrompt
from util.config import MATH_DATA_PATH
from data.data_loader import load_problems, load_aime_problems,load_simplify_problems
from util.util import reject_sample,parse_answer,extract_think_and_after
from util.set_logger import setup_logger
import openai
import math
from self_filter import process_compare,process_reject_sample
def pre_simplify_fun(example):
    return createThinkSimpleQuestionPrompt(example['problem'], example['solution'])

def pre_reject_fun(example):
    return createAnsqerPrompt(example['problem'])

def pre_reverse_compare_fun(example):
    return createCompareThinkPrompt(example['original_problem'], example['original_solution'], example['problem'], example['solution'])

def pre_compare_fun(example):
    return createCompareThinkPrompt(example['problem'], example['solution'],example['original_problem'], example['original_solution'])

def post_fun(example, reply):
    example['answer'] = reply


def post_problem_fun(example, reply):
    think_process,answer=extract_think_and_after(reply)
    example['simplify_process']=think_process
    example['answer']=answer

def post_reverse_fun(example, reply):
    example['reverse_answer'] = reply

def process_one_side_compare(problem, response1,logger):
    try:
        value=0
        response1=response1.lower()
        if problem and response1:
            if "former one is harder" in response1:
                value+=1
            elif "later one is harder" in response1:  #简单的应该放前面
                value-=1
            elif "comparable" in response1:
                value=value
            else:
                logger.error(f"Error!{response1}")
            # logger.info(f"value:{value}")
            return value>0
        logger.warning(f"Invalid data for compare.")
        return False
    except Exception as e:
        logger.error(f"Error in process_compare: {e}")
        return False
    
def process_problem(problem, sections, logger):
    try:
        logger.debug(f"Processing problem")
        if problem and problem['answer'] and sections:
            parsed_problem, parsed_solution = parse_answer(problem['answer'], sections, logger)
            if parsed_problem and parsed_solution:
                # logger.info(f"Successfully parsed problem: {problem['file_name']}")
                return {
                    "original_problem": problem['problem'],
                    "original_solution": problem['solution'],
                    "problem": parsed_problem,
                    "solution": parsed_solution
                }
            else:
                logger.warning(f"Parsed problem or solution is empty for file.")
                return None
        else:
            logger.warning(f"Invalid problem data.")
            return None
    except Exception as e:
        logger.error(f"Error in process_problem for {e}")
        return None
def main(batch_size=8,
         max_iteration=3,
         enable_keep=False,  # enable_keep 用来确定是否要在小范围中每一轮迭代保持和上一轮同样的选择（配合search_keys）
         save_file_name="simplify_data.json",
         max_try=3,
         n_processes=8,
         start_iteration=0,
         start_problem_idx=0):
    logger = setup_logger()
    logger.info("Starting main processing loop.")

    for iteration in range(start_iteration, max_iteration):
        logger.info(f"Starting iteration {iteration + 1}/{max_iteration}")
        try:
            problems = load_simplify_problems(iteration=iteration)

            problems = problems[start_problem_idx:]
            total_problems = len(problems)
            total_batch=math.ceil(total_problems / batch_size)
            logger.info(f"Loaded {total_problems} problems for iteration {iteration + 1}")
            search_keys = []
            output_list = []
            for batch in range(total_batch):
                logger.info(f"Processing batch {batch + 1}/{total_batch}")
                done_keys = []
                batch_problems = problems[batch * batch_size:(batch + 1) * batch_size]
                for attempt in range(max_try):
                    logger.info(f"Attempt {attempt + 1}/{max_try} for batch {batch + 1}")
                    try_problems = [
                        problem for problem in batch_problems
                        if problem['problem'] not in done_keys
                    ]
                    logger.debug(f"{len(try_problems)} problems to process in this attempt.")

                    if not try_problems:
                        logger.info("No more problems to process in this batch.")
                        break

                    # Simplify Process
                    logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try}, Starting simplify process.")
                    batch_get_chat_api(
                        examples=try_problems,
                        eng="deepseek-r1",
                        pre_fun=pre_simplify_fun,  # simplified
                        post_fun=post_fun,
                        logger=logger,
                        n_processes=n_processes,
                        temperature=0.7,
                        timeout=20,
                        max_try=3,
                        think=False
                    )
                    logger.info(try_problems[0])
                    sections = ["Simplified Problem", "Simplified Solution"]
                    simplified_batch_problems = [process_problem(problem, sections, logger) for problem in try_problems]
                    simplified_batch_problems = [item for item in simplified_batch_problems if item]
                    logger.info(f"Successful simplified {len(simplified_batch_problems)} problems.")

                    if simplified_batch_problems:
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1},  Attempt {attempt + 1}/{max_try}, Starting reject sampling.")
                        batch_get_chat_api(
                            examples=simplified_batch_problems,
                            eng="deepseek-r1",
                            pre_fun=pre_reject_fun,  # 拒绝采样
                            post_fun=post_fun,
                            logger=logger,
                            n_processes=4,
                            temperature=0.7,
                            timeout=20,
                            max_try=3,
                            think=False
                        )

                        reject_sampled_batch_problems = [problem for problem in simplified_batch_problems if process_reject_sample(problem,'solution',problem['answer'], logger)]
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try}, {len(reject_sampled_batch_problems)} problems pass reject sample.")
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try}, {len(simplified_batch_problems)- len(reject_sampled_batch_problems)} problems fail in reject sample.")
                        # Compare
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try}, Starting compare process.")
                    else:
                        logger.warning(f"Iteation {iteration + 1}, Batch {batch + 1},  Attempt {attempt + 1}/{max_try}, No problem get simplified.")
                        reject_sampled_batch_problems=[]
                        
                    if reject_sampled_batch_problems:
                        batch_get_chat_api(
                            examples=reject_sampled_batch_problems,
                            eng="deepseek-r1",
                            pre_fun=pre_compare_fun,  # 比较
                            post_fun=post_fun,
                            logger=logger,
                            n_processes=4,
                            temperature=0.7,
                            timeout=20,
                            max_try=3,
                            think=False
                        )

                        batch_get_chat_api(
                            examples=reject_sampled_batch_problems,
                            eng="deepseek-r1",
                            pre_fun=pre_reverse_compare_fun,  # 比较
                            post_fun=post_reverse_fun,
                            logger=logger,
                            n_processes=4,
                            temperature=0.7,
                            timeout=20,
                            max_try=3,
                            think=False
                        )
                        compared_batch_problems = [problem for problem in reject_sampled_batch_problems if process_compare(problem, problem['answer'],problem['reverse_answer'], logger)]
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try},{len(compared_batch_problems)} problems pass compare.")
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try},{len(reject_sampled_batch_problems)-len(compared_batch_problems)} problems fail in  compare.")
                    else:
                        logger.warning(f"Iteation {iteration + 1}, Batch {batch + 1},  Attempt {attempt + 1}/{max_try}, No problem pass reject sample.")
                        compared_batch_problems=[]
                        
                    if compared_batch_problems:
                        done_keys+=[problem['original_problem'] for problem in compared_batch_problems]
                        output_list +=compared_batch_problems
                        tmp_path=MATH_DATA_PATH+f"/{iteration+1}/{save_file_name}"
                        with open(tmp_path, 'w', encoding='utf-8') as output_json:
                            json.dump(output_list, output_json, ensure_ascii=False, indent=4)
                    else:
                        logger.warning(f"Iteation {iteration + 1}, Batch {batch + 1},  Attempt {attempt + 1}/{max_try}, No problem pass compare.")
                
                    logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try},{len(done_keys)} problems has been done.")
                search_keys+=done_keys
                logger.info(f"Iteation {iteration + 1},Batch {batch + 1},Total {len(search_keys)}/{min(len(problems),(batch+1)*batch_size)} has been simplified.")

            start_problem_idx = 0
            if not enable_keep:
                search_keys=[]
            logger.info(f"Iteration {iteration + 1} completed,Total {len(search_keys)}/{len(problems)} has been simplified.")

        except Exception as e:
            logger.error(f"Error during iteration {iteration + 1}: {e}")

    logger.info("Main processing loop completed.")

if __name__ == "__main__":
    main()
