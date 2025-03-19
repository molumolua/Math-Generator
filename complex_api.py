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
from prompt.prompt_design import createNewComplexQuestionPrompt
from util.config import MATH_DATA_PATH
from data.data_loader import load_problems, load_aime_problems,load_simplify_problems
from util.util import reject_sample,parse_answer,extract_think_and_after
from util.set_logger import setup_logger
from util import util
import math
def pre_complex_fun(example):
    return createNewComplexQuestionPrompt(example['problem'], example['solution'])

def post_fun(example, reply):
    example['answer'] = reply


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
def main(batch_size=1024,
         max_iteration=1,
         max_try=1,
         n_processes=32,
         start_iteration=0,
         start_problem_idx=0):
    logger = setup_logger()
    logger.info("Starting main processing loop.")
    input_path ="./deepseek-math/0/math_output_deepseek.json"
    output_path ="./outputs/glm_data.json"
    model="glm-4-plus"

    for iteration in range(start_iteration, max_iteration):
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                problems= json.load(f)
            problems = problems[start_problem_idx:]
            total_problems = len(problems)
            total_batch=math.ceil(total_problems / batch_size)
            logger.info(f"Loaded {total_problems} problems ")
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
                    logger.info(f" Batch {batch + 1}, Attempt {attempt + 1}/{max_try}, Starting complex process.")
                    batch_get_chat_api(
                        examples=try_problems,
                        eng=model,
                        pre_fun=pre_complex_fun,  # simplified
                        post_fun=post_fun,
                        logger=logger,
                        n_processes=n_processes,
                        temperature=0.7,
                        timeout=20,
                        max_try=3,
                        think=False
                    )
                    logger.info(try_problems[0])
                    for problem in tqdm(try_problems):
                            response=problem['answer']
                            complex_problem, complex_solution = util.parse_answer(response, 
                                                                                    ["Hard Question", 
                                                                                    "Hard Answer"], 
                                                                                    logger=logger)
                            if complex_solution and complex_problem:
                                output_object = {
                                    "original_problem": problem['problem'],
                                    "original_solution": problem['solution'],
                                    "complex_problem": complex_problem,
                                    "complex_solution": complex_solution,
                                    "response": response
                                }
                                output_list.append(output_object)   
                    with open(output_path, 'w', encoding='utf-8') as output_json:
                            json.dump(output_list, output_json, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error(f"Error during iteration {iteration + 1}: {e}")

    logger.info("Main processing loop completed.")

if __name__ == "__main__":
    main()
