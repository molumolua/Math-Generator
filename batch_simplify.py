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
from prompt.prompt_design import createComparePrompt, createSimpleQuestionPromptV3, createAnsqerPrompt
from util.config import TRAINING_DATA_PATH, OUTPUT_PATH, TRAINING_DATA_PATH_AIME
from data.data_loader import load_problems, load_aime_problems
from util.util import reject_sample
import openai
import math

def save_answer(category, iteration, file_name, obj, logger):
    try:
        now_path = os.path.join(TRAINING_DATA_PATH, str(iteration))
        category_output_path = os.path.join(now_path, category)
        os.makedirs(category_output_path, exist_ok=True)
        answer_file_path = os.path.join(category_output_path, file_name)
        with open(answer_file_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved answer to {answer_file_path}")
    except Exception as e:
        logger.error(f"Failed to save answer for {file_name} in category {category} at iteration {iteration + 1}: {e}")

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

def find_position(section, next_section, section_list, title_list, matches, answer_len, logger, begin=0):
    try:
        logger.debug(f"Finding position for section '{section}' and next_section '{next_section}' starting from index {begin}")
        start = -1
        end = -1

        # 查找当前章节的位置
        for i in range(begin, len(title_list)):
            if title_list[i] == section_list[section]:
                begin = i + 1
                start = matches[i].end()
                logger.debug(f"Found start of section '{section_list[section]}' at position {start}")
                break
        if start == -1:
            logger.error(f"Section '{section_list[section]}' not found in the title list.")
            return start, end, begin

        # 查找下一个章节的位置（如果有的话）
        if next_section < len(section_list):
            for i in range(begin, len(title_list)):
                if title_list[i] == section_list[next_section]:
                    begin = i
                    end = matches[i].start()
                    logger.debug(f"Found end of section '{section_list[section]}' at position {end}")
                    break
        else:
            end = answer_len  # 如果没有下一个章节，则使用提供的 `answer_len` 作为结束位置。
            logger.debug(f"No next section. Using answer_len {answer_len} as end position.")

        if end == -1 and next_section < len(section_list):
            logger.error(f"Next section '{section_list[next_section]}' not found in the title list.")
        return start, end, begin
    except Exception as e:
        logger.error(f"Error in find_position: {e}")
        return -1, -1, begin

def clear_string(s):
    return s[:-1] if s and s[-1] == ':' else s

def parse_answer(answer_text, sections, logger):
    try:
        logger.debug("Parsing answer text.")
        extracted = {section: "" for section in sections}

        # 匹配标题部分的正则表达式（考虑冒号在外和冒号在内两种情况）
        pattern = re.compile(
            r'^\s*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*)?\*\*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*)?(.*?)\*\*:?', re.MULTILINE
        )
        matches = list(pattern.finditer(answer_text))

        # 如果没有找到匹配，再尝试另一种格式的正则表达式
        if not matches:
            pattern = re.compile(
                r'^\s*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*)?\*\*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*)?(.*?):\*\*', re.MULTILINE
            )
            matches = list(pattern.finditer(answer_text))

        # 如果仍没有匹配，返回空内容
        if not matches:
            logger.warning("No section headers matched in the answer text.")
            logger.warning(answer_text)
            return ("" for _ in sections)

        logger.debug(f"Found {len(matches)} section headers.")
        begin = 0
        title_list = [clear_string(match.group(1).strip()) for match in matches]
        for idx, section in enumerate(sections):
            start, end, begin = find_position(idx, idx + 1, sections, title_list, matches, len(answer_text), logger, begin)
            if start == -1 or end == -1:
                logger.warning(f"Could not extract section '{section}'.")
                continue
            # 提取内容并去除前后空白字符
            content = answer_text[start:end].strip()
            extracted[section] = content
            logger.debug(f"Extracted content for section '{section}': {content[:50]}...")  # 只显示前50个字符
        return (extracted[section] for section in sections)
    except Exception as e:
        logger.error(f"Error in parse_answer: {e}")
        return ("" for _ in sections)

def pre_simplify_fun(example):
    return createSimpleQuestionPromptV3(example['problem'], example['solution'])

def pre_reject_fun(example):
    return createAnsqerPrompt(example['problem'])

def pre_compare_fun(example):
    return createComparePrompt(example['original_problem'], example['original_solution'], example['problem'], example['solution'])

def post_fun(example, reply):
    example['answer'] = reply

def setup_logger():
    logger = logging.getLogger("BatchOpenAI")
    logger.setLevel(logging.INFO)
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # 添加处理器到日志记录器
    if not logger.handlers:
        logger.addHandler(ch)
    return logger

def process_problem(problem, sections, logger):
    try:
        logger.debug(f"Processing problem: {problem['file_name']} in {problem['category']}")
        if problem and problem['answer'] and sections:
            simplify_process, parsed_problem, parsed_solution = parse_answer(problem['answer'], sections, logger)
            if parsed_problem and parsed_solution:
                # logger.info(f"Successfully parsed problem: {problem['file_name']}")
                return {
                    'category': problem['category'],
                    'file_name': problem['file_name'],
                    "original_problem": problem['problem'],
                    "original_solution": problem['solution'],
                    'simplify_process': simplify_process,
                    "problem": parsed_problem,
                    "solution": parsed_solution
                }
            else:
                logger.warning(f"Parsed problem or solution is empty for file: {problem['file_name']} in {problem['category']}")
                return None
        else:
            logger.warning(f"Invalid problem data: {problem['file_name']} in {problem['category']}")
            return None
    except Exception as e:
        logger.error(f"Error in process_problem for {problem.get('file_name', 'unknown')}  in {problem['category']}: {e}")
        return None

def process_reject_sample(problem, logger):
    try:
        logger.debug(f"Processing reject sample for: {problem['file_name']}")
        if problem and problem['answer'] and problem['solution']:
            result = reject_sample(problem['answer'], problem['solution'])
            logger.debug(f"Reject sample result for {problem['file_name']}: {result}")
            return result
        else:
            logger.warning(f"Missing data for reject sample in file: {problem.get('file_name', 'unknown')}  in {problem['category']}")
            return False
    except Exception as e:
        logger.error(f"Error in process_reject_sample for {problem.get('file_name', 'unknown')}  in {problem['category']}: {e}")
        return False

def process_compare(problem, sections, logger):
    try:
        logger.debug(f"Processing compare for: {problem['file_name']}")
        if problem and problem['answer'] and sections:
            _,conclusion = parse_answer(problem['answer'], sections, logger)
            if "former" in conclusion:
                logger.debug(f"'former' found in conclusion for {problem['file_name']}  in {problem['category']}")
                return True
            else:
                logger.debug(f"'former' not found in conclusion for {problem['file_name']}  in {problem['category']}")
                return False
        logger.warning(f"Invalid data for compare in file: {problem.get('file_name', 'unknown')}  in {problem['category']}")
        return False
    except Exception as e:
        logger.error(f"Error in process_compare for {problem.get('file_name', 'unknown')}  in {problem['category']}: {e}")
        return False

def main(batch_size=64,
         max_iteration=3,
         enable_keep=False,  # enable_keep 用来确定是否要在小范围中每一轮迭代保持和上一轮同样的选择（配合search_keys）
         search_keys=[],
         min_level=1,
         max_level=5,
         max_try=3,
         n_processes=8,
         start_iteration=2,
         start_problem_idx=0):
    logger = setup_logger()
    logger.info("Starting main processing loop.")

    for iteration in range(start_iteration, max_iteration):
        logger.info(f"Starting iteration {iteration + 1}/{max_iteration}")
        try:
            problems = load_problems(iteration=iteration, search_keys=search_keys, min_level=min_level, max_level=max_level)

            problems = problems[start_problem_idx:]
            total_problems = len(problems)
            total_batch=math.ceil(total_problems / batch_size)
            logger.info(f"Loaded {total_problems} problems for iteration {iteration + 1}")
            search_keys = []
            for batch in range(total_batch):
                logger.info(f"Processing batch {batch + 1}/{total_batch}")
                done_keys = []
                batch_problems = problems[batch * batch_size:(batch + 1) * batch_size]
                for attempt in range(max_try):
                    logger.info(f"Attempt {attempt + 1}/{max_try} for batch {batch + 1}")
                    try_problems = [
                        problem for problem in batch_problems
                        if {'category': problem['category'], 'file_name': problem['file_name']} not in done_keys
                    ]
                    logger.debug(f"{len(try_problems)} problems to process in this attempt.")

                    if not try_problems:
                        logger.info("No more problems to process in this batch.")
                        break

                    # Simplify Process
                    logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try}, Starting simplify process.")
                    batch_get_chat_api(
                        examples=try_problems,
                        eng="gpt-4o",
                        pre_fun=pre_simplify_fun,  # simplified
                        post_fun=post_fun,
                        logger=logger,
                        n_processes=n_processes,
                        temperature=0.7,
                        timeout=20,
                        max_try=3
                    )

                    sections = ["Simplification Process", "Simplified Problem", "Simplified Solution"]
                    simplified_batch_problems = [process_problem(problem, sections, logger) for problem in try_problems]
                    simplified_batch_problems = [item for item in simplified_batch_problems if item]
                    logger.info(f"Successful simplified {len(simplified_batch_problems)} problems.")

                    if simplified_batch_problems:
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1},  Attempt {attempt + 1}/{max_try}, Starting reject sampling.")
                        batch_get_chat_api(
                            examples=simplified_batch_problems,
                            eng="gpt-4o",
                            pre_fun=pre_reject_fun,  # 拒绝采样
                            post_fun=post_fun,
                            logger=logger,
                            n_processes=4,
                            temperature=0.7,
                            timeout=20,
                            max_try=3
                        )

                        reject_sampled_batch_problems = [problem for problem in simplified_batch_problems if process_reject_sample(problem, logger)]
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
                            eng="gpt-4o",
                            pre_fun=pre_compare_fun,  # 比较
                            post_fun=post_fun,
                            logger=logger,
                            n_processes=4,
                            temperature=0.7,
                            timeout=20,
                            max_try=3
                        )

                        sections = [ "Reasoning Steps","Conclusion"]
                        compared_batch_problems = [problem for problem in reject_sampled_batch_problems if process_compare(problem, sections, logger)]
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try},{len(compared_batch_problems)} problems pass compare.")
                        logger.info(f"Iteation {iteration + 1}, Batch {batch + 1}, Attempt {attempt + 1}/{max_try},{len(reject_sampled_batch_problems)-len(compared_batch_problems)} problems fail in  compare.")
                    else:
                        logger.warning(f"Iteation {iteration + 1}, Batch {batch + 1},  Attempt {attempt + 1}/{max_try}, No problem pass reject sample.")
                        compared_batch_problems=[]
                        
                    if compared_batch_problems:
                        for problem in compared_batch_problems:
                            save_answer(problem['category'], iteration + 1, problem['file_name'], problem, logger)
                            done_key = {'category': problem['category'], 'file_name': problem['file_name']}
                            done_keys.append(done_key)
                            logger.debug(f"Problem {problem['file_name']}  in {problem['category']} marked as done.")
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
