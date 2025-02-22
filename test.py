# from transformers import AutoModelForCausalLM, AutoTokenizer
# import time
# import json
# from util import config, grader, graph, util
# from util.util import remove_boxed
# import csv
# import sys
# import os
# import json
# import random
# import ast
# import re
# import logging
# import sys
# from tqdm import tqdm
# from data.data_loader import load_problems
# from util.config import TRAINING_DATA_PATH,OUTPUT_PATH
# from prompt.prompt_design import createComplexQuestionPrompt
# device = "cuda" # the device to load the model onto
# model_name_or_path = "/home/bingxing2/home/scx8q73/jobs/LLaMA-Factory-main/saves/qwen2-7b/full/sft"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# prompt = "\nYou are a mathematics expert specializing in increasing the complexity of math problems.\n\nYour task is to transform the **Original Problem** into a more challenging version by introducing advanced mathematical concepts or techniques.\n\nPlease provide the following sections in your answer:\n\n1. **Complexified Problem**:\n   - Provide the **revised** problem statement **without any introductory or explanatory sentences**.\n\n2. **Complexified Solution**:\n   - Present the complexified solution in a logical sequence, ensuring that you demonstrate the use of the more advanced concepts or techniques introduced in the new problem statement.\n\n**Format Requirements**:\n- Use **bold** formatting for each section title.\n- Ensure that the final answer is enclosed in a LaTeX boxed format containing **only the numerical value**.\n\n**Constraints**:\n- **Ensure that the complexified problem has a unique and challenging answer**.\n- **You must change the wording or structure of the original problem statement** enough to reflect the more advanced approach.\n\n\n**Original Problem**:\nIf $x = 2$ and $y = 5$, then what is the value of $\\frac{x^4+2y^2}{6}$ ?\n\n**Original Solution**:\nWe have  \\[\\frac{x^4 + 2y^2}{6} = \\frac{2^4 + 2(5^2)}{6} = \\frac{16+2(25)}{6} = \\frac{16+50}{6} = \\frac{66}{6} = \\boxed{11}.\\]\n"
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(device)

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)


# import os
# import json
# from util import util, set_logger
# from reject_sample import reject_sample
# from prompt.openai_access import batch_get_chat_api
# from util.util import parse_answer
# from prompt.prompt_design import createAddProcessPrompt_2
# import math
# def pre_reject_fun(example):
#     return createAddProcessPrompt_2(example['original_problem'],example['original_solution'],example['problem'],example['solution'])

# def post_fun(example, reply):
#     example['simplify_process'] = reply
# def main(
#         data_path="./outputs/outputs_backup.json",
#         output_path='./outputs/outputs_fixed.json',
#         batch_size=64):
#     with open(data_path, 'r', encoding='utf-8') as f:
#         data_list = json.load(f)
#     data_list=data_list
#     #preprocess problems
#     problems=[]
#     for data in data_list:
#         problems.append(data)
#     problems=problems
#     #setup logger
#     logger=set_logger.setup_logger()
#     total_problems = len(problems)
#     total_batch=math.ceil(total_problems / batch_size)
#     logger.info(f"Loaded {total_problems} problems to add process.")
#     output_list=[]
#     for batch in range(total_batch):
#         batch_problems = problems[batch * batch_size:(batch + 1) * batch_size]
#         logger.info(f"Batch {batch + 1}, Starting add process.")
#         batch_get_chat_api(
#             examples=batch_problems,
#             eng="gpt-4o",
#             pre_fun=pre_reject_fun,  # 拒绝采样
#             post_fun=post_fun,
#             logger=logger,
#             n_processes=8,
#             temperature=0.7,
#             timeout=20,
#             max_try=3
#         )
        
#         output_list+=batch_problems
#         with open(output_path, 'w', encoding='utf-8') as output_json:
#             json.dump(output_list, output_json, ensure_ascii=False, indent=4)
#         logger.info(f"Batch {batch + 1},Total {len(output_list)}/{min(len(problems),(batch+1)*batch_size)} has been left.")

# if __name__ == "__main__":
#     main()


# import os
# import json
# from util import util, set_logger
# from reject_sample import reject_sample
# from prompt.openai_access import batch_get_chat_api
# from util.util import parse_answer
# from prompt.prompt_design import createAddProcessPrompt_2
# import math
# def pre_reject_fun(example):
#     return createAddProcessPrompt_2(example['original_problem'],example['original_solution'],example['problem'],example['solution'])

# def post_fun(example, reply):
#     example['simplify_process'] = reply
# def main(
#         unfilter_data_path="./outputs/complex_question_process_1.5b_math.json",
#         data_path="./outputs/filter_complex_question_process_1.5b_math.json",
#         output_path='./outputs/train_data_filter_complex_question_process_1.5b_math.json',
#         batch_size=64):
#     with open(unfilter_data_path, 'r', encoding='utf-8') as f:
#         unfilter_data_list = json.load(f)
#     with open(data_path, 'r', encoding='utf-8') as f:
#         data_list = json.load(f)
#     output_list=[]
#     for data in data_list:
#         for data2 in unfilter_data_list:
#             if data['problem'] == data2['complex_problem']:
#                 data['complexify_process']=data2['Complexification Process']
#         output_list.append(data)
#     with open(data_path, 'w', encoding='utf-8') as f:
#         json.dump(output_list, f, ensure_ascii=False, indent=4)

# if __name__ == "__main__":
#     main()



import os
import json
from util import util, set_logger
from reject_sample import reject_sample
from prompt.openai_access import batch_get_chat_api
from util.util import parse_answer,extract_think_and_after
from prompt.prompt_design import createAnsqerPrompt,createComparePrompt
import math
import random
from data.data_loader import load_simplify_problems
from prompt.openai_access import get_oai_completion
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
            problem['reject_result']=result
        else:
            logger.warning(f"Missing data for reject sample in file.")
            problem['reject_result']= False
    except Exception as e:
        logger.error(f"Error in process_reject_sample : {e}")
        problem['reject_result']= False
    return problem

def process_compare(problem, sections, logger):
    try:
        if problem and problem['answer'] and sections:
            step,conclusion = parse_answer(problem['answer'], sections, logger)
            if "former" in conclusion:
                problem['compare_result']="former"
                problem["compare_step"]=step
                return problem
            else:
                problem['compare_result']=False
                return problem
        logger.warning(f"Invalid data for compare in file.")
        problem['compare_result']=False
        return problem
    except Exception as e:
        logger.error(f"Error in process_compare: {e}")
        problem['compare_result']=False
        return problem
def main_2(
        data_path_1="./outputs_23/outputs_23.json",
        data_path_2="./outputs/outputs_process.json",
        output_path="./outputs_23/outputs_process_23.json",
        batch_size=128):
    with open(data_path_1, 'r', encoding='utf-8') as f:
        data_list_1 = json.load(f)
    with open(data_path_2, 'r', encoding='utf-8') as f:
        data_list_2 = json.load(f)
    output_list=[]
    for data_2 in data_list_2:
        for data_1 in data_list_1:
            if data_2['original_problem'] == data_1['original_problem'] and data_2['problem'] == data_1['problem']:
                output_list.append(data_2)
    with open(output_path, 'w', encoding='utf-8') as output_json:
        json.dump(output_list, output_json, ensure_ascii=False, indent=4)
# filter problem 带参
def main(
        data_path=".\outputs\complex_question_process_deepseek.json",
        output_path='.\outputs\compared_complex_question_process_deepseek_200.json',
        batch_size=128):
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    random.seed(100)
    random.shuffle(data_list)
    data_list=data_list[:200]
    # data_list=data_list[2000:2500]
    #preprocess problems
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
    logger=set_logger.setup_logger()
    total_problems = len(problems)
    total_batch=math.ceil(total_problems / batch_size)
    logger.info(f"Loaded {total_problems} problems to filter.")
    output_list=[]
    for batch in range(total_batch):
        batch_problems = problems[batch * batch_size:(batch + 1) * batch_size]
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
        batch_problems = [process_reject_sample(problem, logger) for problem in batch_problems]
        
        logger.info(f"Batch {batch + 1}, Starting compare.")
        batch_get_chat_api(
            examples=batch_problems,
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
        batch_problems = [process_compare(problem, sections, logger) for problem in batch_problems]
        
        output_list+=batch_problems
        with open(output_path, 'w', encoding='utf-8') as output_json:
            json.dump(output_list, output_json, ensure_ascii=False, indent=4)
        logger.info(f"Batch {batch + 1},Total {len(output_list)}/{min(len(problems),(batch+1)*batch_size)} has been left.")
def add_newlines_after_think(text: str) -> str:
    """
    在给定字符串的每个 `</think>` 后插入两个换行符。

    :param text: 原始字符串
    :return: 替换后的字符串
    """
    return text.replace("</think>", "</think>\n\n")
def main_3(
        data_path="/data/xucaijun/Math-Generator/outputs/newprompt_complex_question_process_deepseek.json",
        batch_size=128):
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    data_list=data_list
    #preprocess problems
    logger=set_logger.setup_logger()
    for data in data_list: 
        data['response']=add_newlines_after_think(data['response'])
        data['complex_problem'],data['complex_solution']= util.parse_answer(data['response'], 
                                                                            ["Complexified Problem", 
                                                                            "Complexified Solution"], 
                                                                            logger=logger)
    with open(data_path, 'w', encoding='utf-8') as output_json:
        json.dump(data_list, output_json, ensure_ascii=False, indent=4)
def main_4(
        data_path_2="/data/xucaijun/Math-Generator/outputs/newprompt_fliter_complex_question_process_deepseek.json",
        output_path="./outputs/rawMATH_800.json"):
    data_list_1=load_simplify_problems(iteration=0)
    with open(data_path_2, 'r', encoding='utf-8') as f:
        data_list_2 = json.load(f)
    output_list=[]
    for data_2 in data_list_2:
        for data_1 in data_list_1:
            if data_2['original_problem'] == data_1['problem']:
                output_list.append(data_1)
    with open(output_path, 'w', encoding='utf-8') as output_json:
        json.dump(output_list, output_json, ensure_ascii=False, indent=4)

def process_problem(problem):
    if problem['value']==True:
        think,solution=extract_think_and_after(problem['output'])
        return {'problem': problem['problem'],'solution': solution}
    else:
        return {'problem': problem['problem'],'solution': problem['solution']}
if __name__ == "__main__":
    # # main_4()
    # from collections import defaultdict

    # # 示例数据
    # data = [
    #     {'original_problem': 'problem1', 'other_attr': 1},
    #     {'original_problem': 'problem2', 'other_attr': 2},
    #     {'original_problem': 'problem1', 'other_attr': 3},
    #     {'original_problem': 'problem3', 'other_attr': 4},
    #     {'original_problem': 'problem2', 'other_attr': 5}
    # ]

    # # 使用 defaultdict 来聚合
    # grouped = defaultdict(list)

    # # 遍历数据，将相同 original_problem 的 dict 聚集在一起
    # for item in data:
    #     grouped[item['original_problem']].append(item)

    # # 转换成二维 list
    # result = list(grouped.values())

    # # 输出结果
    # print(result)
    # print(get_oai_completion("你是谁？","deepseek-r1",0.7))

    input_path="/data/xucaijun/New/Math-Generator/outputs/math_output_deepseek.json"
    with open(input_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)

    problems=[ process_problem(problem) for problem in problems ]
    output_path="/data/xucaijun/New/Math-Generator/deepseek-math/0/math_output_deepseek.json"
    with open(output_path, 'w', encoding='utf-8') as output_json:
        json.dump(problems, output_json, ensure_ascii=False, indent=4)

