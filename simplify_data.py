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
from util.util import extract_think_and_after
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
         max_iteration=3,
         max_try=3,
        #  model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B/full/sft"
         model_name_or_path="/data/xucaijun/DeepSeek-R1-Distill-Qwen-32B"
         ):
    logger = set_logger.setup_logger()
    logger.info("Starting main processing loop.")
    # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device=device,tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
    )
    logger.info("Model loaded successfully.")

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        stop=stop_words,
        n=1
    )
    for iteration in range(max_iteration):
        logger.info(f"Starting iteration {iteration + 1}/{max_iteration}")
        try:
            problems = load_simplify_problems(data_name=data_name,iteration=iteration)
            problems = problems[:10]
            total_problems = len(problems)
            logger.info(f"Loaded {total_problems} problems for iteration {iteration + 1}")
            done_problems=[]
            done_keys=[]
            for attempt in range(max_try):
                try_problems = [
                    problem for problem in problems
                    if problem['problem'] not in done_keys
                ]
                logger.debug(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try},{len(try_problems)} problems to process in this attempt.")

                if not try_problems:
                    logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try},No more problems to process in this batch.")
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
                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try},Start simplify problems.")
                generated_responses = model.generate(input_texts, sampling_params=sampling_params)
                generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]

                sections = ["Simplified Problem", "Simplified Solution"]
                simplified_problems = [process_problem(problem, generated_response,sections, logger) for problem,generated_response in zip(try_problems,generated_responses)]
                simplified_problems = [item for item in simplified_problems if item]
                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try},Successful simplified {len(simplified_problems)} problems.")

                if len(simplified_problems) == 0:
                    logger.info("No problem for reject sample.")
                    continue

                # reject sample
                input_texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": createAnsqerPrompt(problem['problem'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for problem in simplified_problems
                ]
                logger.info(input_texts[0])

                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try},Start reject sample.")
                generated_responses = model.generate(input_texts, sampling_params=sampling_params)
                generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]

                reject_sampled_problems = [problem for problem,generated_response in zip(simplified_problems,generated_responses) if process_reject_sample(problem,generated_response,logger)]
                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try}, {len(reject_sampled_problems)} problems pass reject sample.")
                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try}, {len(simplified_problems)- len(reject_sampled_problems)} problems fail in reject sample.")

                #compare process
                if len(reject_sampled_problems) == 0:
                    logger.info("No problem for compare.")
                    continue

                input_texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": createCompareThinkPrompt(problem['original_problem'], problem['original_solution'], problem['problem'], problem['solution'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for problem in reject_sampled_problems
                ]
                logger.info(input_texts[0])
                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try},Start compare.")
                generated_responses = model.generate(input_texts, sampling_params=sampling_params)
                generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]
                logger.info(generated_responses[0])

                #reversed compare
                reversed_input_texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": createCompareThinkPrompt(problem['problem'], problem['solution'], problem['original_problem'], problem['original_solution'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for problem in reject_sampled_problems
                ]
                logger.info(reversed_input_texts[0])
                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try},Start reversed compare.")
                reversed_generated_responses = model.generate(reversed_input_texts, sampling_params=sampling_params)
                reversed_generated_responses = [generated_response.outputs[0].text for generated_response in reversed_generated_responses]
                
                logger.info(reversed_generated_responses[0])
                compared_problems = [problem for problem,generated_response,reversed_generated_response \
                                     in zip(reject_sampled_problems,generated_responses,reversed_generated_responses) if process_compare(problem,generated_response,reversed_generated_response,logger)]
                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try}, {len(compared_problems)} problems pass compare.")
                logger.info(f"Iteation {iteration + 1}, Attempt {attempt + 1}/{max_try}, {len(reject_sampled_problems)- len(compared_problems)} problems fail in compare.")

                if len(compared_problems)== 0 :
                    logger.info("No problem for add think and save.")
                    continue
                #add_think
                compared_problems=add_think(model,tokenizer,logger,compared_problems,save=0)

                done_problems +=compared_problems
                done_keys +=[problem["original_problem"] for problem in compared_problems]
                save_problems_to_json("simplify_problem.json",done_problems,iteration+1,logger)
                # save_problems_to_jsonl("train_data.jsonl",done_problems,iteration+1,logger)
                
            logger.info(f"Iteration {iteration + 1} completed,Total {len(done_problems)}/{len(problems)} has been simplified.")
        except Exception as e:
            logger.error(f"Error during iteration {iteration + 1}: {e}")

    logger.info("Main processing loop completed.")

if __name__ == "__main__":
    main()
