import os

import time
import logging
import json
import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from prompt.openai_access import batch_get_chat_api, get_oai_completion
from prompt.prompt_design import createComparePrompt, createSimpleQuestionPromptV3, createAnsqerPrompt,createThinkSimpleQuestionPrompt,createCompareThinkPrompt
from util.config import TRAINING_DATA_PATH, OUTPUT_PATH, TRAINING_DATA_PATH_AIME,MATH_DATA_PATH,DEEPSEEK_DATA_PATH
from data.data_loader import load_simplify_problems
from util.util import reject_sample
import openai
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import util, set_logger
from add_think import add_think
from util.util import extract_think_and_after,process_output_data
from self_filter import self_filter
def save_problems_to_jsonl(file_name,problems, iteration,logger):
    """
    参数:
    problems: 列表，其中每个元素都是一个字典
    file_path: 输出文件的路径，通常以 .jsonl 结尾
    """
    now_path = os.path.join(DEEPSEEK_DATA_PATH, str(iteration))
    os.makedirs(now_path, exist_ok=True)
    file_path = os.path.join(now_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        for problem in problems:
            # 将字典对象序列化为 JSON 字符串
            line = json.dumps(problem, ensure_ascii=False)
            # 写入文件，并在结尾加换行符
            f.write(line + '\n')
    logger.info(f"Saved answer to {file_path}")
def save_problems_to_json(file_name,problems, iteration,logger):
    """
    参数:
    problems: 列表，其中每个元素都是一个字典
    file_path: 输出文件的路径，通常以 .json 结尾
    """
    now_path = os.path.join(DEEPSEEK_DATA_PATH, str(iteration))
    os.makedirs(now_path, exist_ok=True)
    file_path = os.path.join(now_path, file_name)
    logger.info(f"Saved answer to {file_path}")
    with open(file_path, 'w', encoding='utf-8') as output_json:
        json.dump(problems, output_json, ensure_ascii=False, indent=4)

def reject_sample_check(problem, solution, model, logger):
    try:
        logger.debug(f"Checking reject sample for problem: {problem}")
        prompt = createAnsqerPrompt(problem)
        gpt_answer = get_oai_completion(prompt, model=model)
        if gpt_answer:
            result = reject_sample(gpt_answer, solution)
            logger.debug(f"Reject sample result: {result}")
            return result
        else:
            logger.warning("GPT answer is empty.")
            return False
    except Exception as e:
        logger.error(f"Error in reject_sample_check: {e}")
        return False

def reject_sample_check_batch(problem, solution, model, logger):
    return reject_sample_check(problem, solution, model, logger)

def pre_simplify_fun(example):
    return createSimpleQuestionPromptV3(example['problem'], example['solution'])

def pre_reject_fun(example):
    return createAnsqerPrompt(example['problem'])

def pre_compare_fun(example):
    return createComparePrompt(example['original_problem'], example['original_solution'], example['problem'], example['solution'])

def post_fun(example, reply):
    example['answer'] = reply


def process_problem(problem, response,sections, logger):
    try:
        if problem and response:
            think,content=extract_think_and_after(response)
            parsed_problem, parsed_solution = util.parse_answer(content, sections, logger)
            if parsed_problem and parsed_solution:
                # logger.info(f"Successfully parsed problem: {problem['file_name']}")
                return {
                    "original_problem": problem['problem'],
                    "original_solution": problem['solution'],
                    "problem": parsed_problem,
                    "solution": parsed_solution,
                    "simplify_think":think
                }
            else:
                logger.warning(f"Parsed problem or solution is empty.")
                return None
        else:
            logger.warning(f"Invalid problem data.")
            return None
    except Exception as e:
        logger.error(f"Error in process_problem: {e}")
        return None

def process_reject_sample(problem, response,logger):
    try:
        if problem and problem['solution'] and response:
            result = reject_sample(response,problem['solution'])
            return result
        else:
            logger.warning(f"Missing data for reject sample.")
            return False
    except Exception as e:
        logger.error(f"Error in process_reject_sample.")
        return False

def process_compare(problem, response1,response2,logger):
    try:
        value=0
        if problem and response1 and response2:
            if "former one is harder" in response1:
                value+=1
            elif "later one is harder" in response1:
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
            logger.info(f"value:{value}")
            return value>0
        logger.warning(f"Invalid data for compare.")
        return False
    except Exception as e:
        logger.error(f"Error in process_compare: {e}")
        return False

def main(stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>"],
         max_tokens=32768,
         device="cuda",
         data_name="DEEPSEEK",
         max_iteration=1,
         N=5,
        #  model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B/full/sft"
         model_name_or_path="/llm/DeepSeek/DeepSeek-R1-Distill-Qwen-32B"
         ):
    logger = set_logger.setup_logger()
    logger.info("Starting main processing loop.")
    # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device=device,tensor_parallel_size=4)
    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
    )
    logger.info("Model loaded successfully.")

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6,
        stop=stop_words,
        n=N
    )
    for iteration in range(max_iteration):
        logger.info(f"Starting iteration {iteration + 1}/{max_iteration}")
        try:
            problems = load_simplify_problems(data_name=data_name,iteration=iteration)
            if iteration>0:
                problems = [problem[0] for problem in problems]
            problems = problems
            total_problems = len(problems)
            logger.info(f"Loaded {total_problems} problems for iteration {iteration + 1}")
            done_problems=[]
            try_problems = [problem for problem in problems]
            logger.debug(f"Iteation {iteration + 1}, {len(try_problems)} problems to process in this attempt.")

            if not try_problems:
                logger.info(f"Iteation {iteration + 1}, No more problems to process in this iteration.")
                continue
                
            # simplify problem
            input_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": createThinkSimpleQuestionPrompt(problem['problem'], problem['solution'])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for problem in try_problems
            ]
            logger.info(input_texts[0])
            logger.info(f"Iteation {iteration + 1}, Start simplify problems.")
            generated_responses = model.generate(input_texts, sampling_params=sampling_params)
            generated_responses = [[generated_response.outputs[i].text for i in range(N)] for generated_response in generated_responses]
            sections = ["Simplified Question", "Simplified Answer"]
            simplified_problems = [[process_problem(problem, response,sections, logger) for response in generated_response ] for problem,generated_response in zip(try_problems,generated_responses)]
            simplified_problems = [[item for item in item_list if item] for item_list in simplified_problems if item_list]
            
            todo_problems=[]
            for item_list in simplified_problems:
                todo_problems +=item_list
            logger.info(f"Iteation {iteration + 1}, Successful simplified {len(todo_problems)} problems.")

            if len(todo_problems) == 0:
                logger.info("No problem for reject sample.")
                continue
            
            compared_problems = todo_problems
            # compared_problems=self_filter(model,tokenizer,todo_problems,logger,batch_size=len(todo_problems),N=1,test_section_names=['problem','solution'],original_section_names=['problem','solution'],complex_section_names=['original_problem','original_solution'])
            #add_think
            # compared_problems=add_think(model,tokenizer,logger,compared_problems,save=0)

            done_problems +=compared_problems

            done_problems = process_output_data(done_problems)
            save_problems_to_json("raw_simplify_problem.json",done_problems,iteration+1,logger)
            # save_problems_to_jsonl("train_data.jsonl",done_problems,iteration+1,logger)
                
            logger.info(f"Iteration {iteration + 1} completed,Total {len(done_problems)}/{len(problems)} has been simplified.")
        except Exception as e:
            logger.error(f"Error during iteration {iteration + 1}: {e}")

    logger.info("Main processing loop completed.")

if __name__ == "__main__":
    main()
