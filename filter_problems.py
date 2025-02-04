import os
import json
from util import util, set_logger
from reject_sample import reject_sample
from prompt.openai_access import batch_get_chat_api
from util.util import parse_answer
from prompt.prompt_design import createAnsqerPrompt,createComparePrompt
import math
import random
def pre_reject_fun(example):
    return createAnsqerPrompt(example['problem'])

def pre_compare_fun(example):
    return createComparePrompt(example['problem'], example['solution'],example['original_problem'], example['original_solution'])

def post_fun(example, reply):
    example['answer'] = reply
def process_reject_sample(problem, logger):
    try:
        if problem and problem['answer'] and problem['solution']:
            result = reject_sample(problem['answer'], problem['solution'])
            return result
        else:
            logger.warning(f"Missing data for reject sample in file.")
            return False
    except Exception as e:
        logger.error(f"Error in process_reject_sample : {e}")
        return False

def process_compare(problem, sections, logger):
    try:
        if problem and problem['answer'] and sections:
            _,conclusion = parse_answer(problem['answer'], sections, logger)
            if "former" in conclusion:
                return True
            else:
                return False
        logger.warning(f"Invalid data for compare in file.")
        return False
    except Exception as e:
        logger.error(f"Error in process_compare: {e}")
        return False

def filter_problems(data_list,batch_size=64,save=False,save_path=None,logger=None,enable_reject=True,enable_compare=True):
    problems=[]
    for data in data_list: 
        if(data["complex_problem"] and data["complex_solution"] and data['Complexification Process']):
            problems.append({
                "original_problem":data["original_problem"],
                "original_solution":data["original_solution"],
                "problem":data["complex_problem"],
                "solution":data["complex_solution"],
                "complexify_process":data['Complexification Process']
            })
    #setup logger
    total_problems = len(problems)
    total_batch=math.ceil(total_problems / batch_size)
    logger.info(f"Loaded {total_problems} problems to filter.")
    output_list=[]
    for batch in range(total_batch):
        batch_problems = problems[batch * batch_size:(batch + 1) * batch_size]
        if enable_reject:
            logger.info(f"Batch {batch + 1}, Starting reject sampling.")
            batch_get_chat_api(
                examples=batch_problems,
                eng="gpt-4o",
                pre_fun=pre_reject_fun,  # 拒绝采样
                post_fun=post_fun,
                logger=logger,
                n_processes=8,
                temperature=0.7,
                timeout=20,
                max_try=3
            )
            reject_sampled_batch_problems = [problem for problem in batch_problems if process_reject_sample(problem, logger)]
        else:
            reject_sampled_batch_problems=batch_problems
        logger.info(f"Batch {batch + 1},{len(reject_sampled_batch_problems)} problems pass reject sample.")
        logger.info(f"Batch {batch + 1},{len(batch_problems)- len(reject_sampled_batch_problems)} problems fail in reject sample.")
        if enable_compare:
            logger.info(f"Batch {batch + 1}, Starting compare.")
            batch_get_chat_api(
                examples=reject_sampled_batch_problems,
                eng="gpt-4o",
                pre_fun=pre_compare_fun,  # 比较
                post_fun=post_fun,
                logger=logger,
                n_processes=8,
                temperature=0.7,
                timeout=20,
                max_try=3
            )
            sections = [ "Reasoning Steps","Conclusion"]
            compared_batch_problems = [problem for problem in reject_sampled_batch_problems if process_compare(problem, sections, logger)]
        else:
            compared_batch_problems = reject_sampled_batch_problems
        logger.info(f"Batch {batch + 1}, {len(compared_batch_problems)} problems pass compare.")
        logger.info(f"Batch {batch + 1}, {len(reject_sampled_batch_problems)-len(compared_batch_problems)} problems fail in  compare.")
        
        output_list+=compared_batch_problems
        if save:
            with open(save_path, 'w', encoding='utf-8') as output_json:
                json.dump(output_list, output_json, ensure_ascii=False, indent=4)
    left_list = [ problem for problem in problems if problem not in output_list]
    left_list = get_original_problem_solution(left_list)
    return output_list,left_list
def get_original_problem_solution(data_list):
    output_list=[]
    for problem in data_list:
        output_list.append({
            "problem":problem['original_problem'],
            "solution":problem['solution']
        })
    return output_list
def main(
        data_path="/data/xucaijun/Math-Generator/outputs/complex_question_process_deepseek_check.json",
        output_path='/data/xucaijun/Math-Generator/outputs/filter_complex_question_process_deepseek_200.json',
        batch_size=128):
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    random.seed(100)
    data_list=data_list[:200]
    logger=set_logger.setup_logger()
    filter_problems(data_list=data_list,batch_size=batch_size,save=True,save_path=output_path,logger=logger)
if __name__ == "__main__":
    main()