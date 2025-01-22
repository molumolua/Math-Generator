import random
from util.config import TRAINING_DATA_PATH, OPENAI_API_KEY
from openai import OpenAI
import time
import os
import json
import random
import ast
import re
import logging
import sys
from datetime import datetime
from tqdm import tqdm
from data.data_loader import load_problems
from prompt.openai_access import call_chatgpt,get_oai_completion
from util.config import TRAINING_DATA_PATH,OUTPUT_PATH
from prompt.prompt_design import createSimpleQuestionPrompt,createComparePrompt,createSimpleQuestionPromptV2
from prompt.new_prompt_design import createConstructPrompt,createProblemPrompt,createReConstructPrompt,createCheckPrompt
from util.graph import str2graph,findEraseProcess,constrcutDatafromMethod
import os
import json
def load_compare_problems(path):
    problems = []
    with open(path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        for data in data_list:
            problem={
                'problem1': data.get('problem1_problem'),
                'solution1': data.get('problem1_solution'),
                'problem2': data.get('problem2_problem'),
                'solution2': data.get('problem2_solution'),
                'result':data.get('result')
            }
            problems.append(problem)
    return problems
def save_answer(category, iteration, file_name, object):
    now_path = TRAINING_DATA_PATH +"/{}".format(iteration)
    category_output_path = os.path.join(now_path, category)
    os.makedirs(category_output_path, exist_ok=True)
    answer_file_path = os.path.join(category_output_path, file_name)
    with open(answer_file_path, 'w', encoding='utf-8') as f:
        json.dump(object, f, ensure_ascii=False, indent=4)

def find_position(section, next_section, section_list, title_list, matches, answer_len, begin=0):
    start = -1
    end = -1
    
    # 查找当前章节的位置
    for i in range(begin, len(title_list)):
        if title_list[i] == section_list[section]:
            begin = i+1
            start = matches[i].end()
            break
    if start == -1:
         tqdm.write(f"Section '{section_list[section]}' not found in the title list.")

    # 查找下一个章节的位置（如果有的话）
    if next_section < len(section_list):
        for i in range(begin, len(title_list)):
            if title_list[i] == section_list[next_section]:
                begin = i
                end = matches[i].start()
                break
    else:
        end = answer_len  # 如果没有下一个章节，则使用提供的 `answer_len` 作为结束位置。

    if end == -1:
        tqdm.write(f"Next section '{section_list[next_section]}' not found in the title list.")
    return start, end, begin

def clear_string(str):
    if str and str[-1] == ':':
        return str[:-1]
    else:
        return str
def parse_answer(answer_text,sections):
    """
    解析模型生成的回答内容，提取各个部分的信息。
    
    参数:
        answer_text (str): 模型生成的回答文本。
        sections:预期模型应该生成的标题，包裹在**之中。
    """
    
    # 创建一个字典来存储提取的内容
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
        return ("" for _ in sections)
    begin=0
    title_list= [clear_string(match.group(1).strip()) for match in matches ]
    for (idx,section) in enumerate(sections):
        start,end,begin=find_position(idx,idx+1,sections,title_list,matches,len(answer_text),begin)
        if start == -1 or end == -1:
            continue
        # 提取内容并去除前后空白字符
        content = answer_text[start:end].strip()

        # 进一步清理内容，如去除前导的有序或无序列表编号
        content = re.sub(r'^[\*\-\d]+\.\s*', '', content, flags=re.MULTILINE)  # 清理有序列表前缀
        content = re.sub(r'^[\*\-]+\s*', '', content, flags=re.MULTILINE)        # 清理无序列表前缀

        # 将提取的内容存储到字典中
        extracted[section] = content
    return (extracted[section] for section in sections)
def clean_json_string(json_str):
    match = re.search(r'\[.*\]', json_str, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return ''
def compare_random_problems(N, M, a, b, model='gpt-4o', output_dir='comparisons', category=None):
    """
    随机选取N个level为a的题目和M个level为b的题目，确保它们属于同一个类别，并逐对进行比较。
    比较结果以JSON格式保存到指定的目录中。

    参数：
    - N (int): 需要选取的level为a的题目数量
    - M (int): 需要选取的level为b的题目数量
    - a (int): 第一个级别
    - b (int): 第二个级别
    - model (str): 使用的OpenAI模型，默认为'gpt-4o'
    - output_dir (str): 保存比较结果的目录，默认为 'comparisons'
    - category (str, optional): 指定的类别。如果未指定，将随机选择一个类别。

    返回：
    - comparisons (list): 比较结果的列表，每个元素包含两个题目及其比较结果
    """
    # 加载所有题目以确定可用类别
    all_problems = load_problems(iteration=0, search_keys=None, min_level=None, max_level=None)
    available_categories = list(set([problem['category'] for problem in all_problems]))
    
    if not available_categories:
        print("没有可用的类别。")
        return []
    
    # 如果未指定类别，随机选择一个
    if category is None:
        category = random.choice(available_categories)
        print(f"随机选择的类别: {category}")
    else:
        if category not in available_categories:
            print(f"指定的类别 '{category}' 不存在。可用类别: {available_categories}")
            return []
        print(f"使用指定的类别: {category}")
    
    # 加载level为a的所有题目，过滤指定类别
    problems_a = load_problems(iteration=0, search_keys=None, min_level=a, max_level=a)
    problems_a = [p for p in problems_a if p['category'] == category]
    if len(problems_a) < N:
        print(f"警告: 请求 {N} 个level为 {a} 且类别为 '{category}' 的题目，但仅找到 {len(problems_a)} 个。将使用所有可用的题目。")
        N = len(problems_a)
    selected_a = random.sample(problems_a, N) if problems_a else []
    
    # 加载level为b的所有题目，过滤指定类别
    problems_b = load_problems(iteration=0, search_keys=None, min_level=b, max_level=b)
    problems_b = [p for p in problems_b if p['category'] == category]
    if len(problems_b) < M:
        print(f"警告: 请求 {M} 个level为 {b} 且类别为 '{category}' 的题目，但仅找到 {len(problems_b)} 个。将使用所有可用的题目。")
        M = len(problems_b)
    selected_b = random.sample(problems_b, M) if problems_b else []
    
    if not selected_a or not selected_b:
        print("没有足够的题目进行比较。")
        return []
    
    comparisons = []
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 生成唯一的文件名，包含时间戳和类别信息
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"comparison_results_{a}_{b}_{category}_{timestamp}.json")
    former_cnt = 0
    comparable_cnt = 0 
    later_cnt = 0
    # 遍历选中的题目并进行比较
    for idx_a, problem_a in enumerate(selected_a, 1):
        for idx_b, problem_b in enumerate(selected_b, 1):
            print(f"比较题目 {idx_a}/{N} (Level {a}) 和 题目 {idx_b}/{M} (Level {b})")
            prompt = createComparePrompt(
                problem1=problem_a['problem'],
                answer1=problem_a['solution'],
                problem2=problem_b['problem'],
                answer2=problem_b['solution']
            )
            ai_response = get_oai_completion(prompt, model)
            if ai_response:
                sections = ["Reasoning Steps","Conclusion"]
                steps,conclusion= parse_answer(ai_response, sections)
                result = -1
                if "former" in conclusion:
                    result = "former is harder."
                    former_cnt += 1
                elif "comparable" in conclusion:
                    result = "comparable"
                    comparable_cnt+=1
                elif "later" in conclusion:
                    result = "later is harder."
                    later_cnt +=1
                comparison_entry = {
                    'problem1': {
                        'category': problem_a['category'],
                        'file_name': problem_a['file_name'],
                        'problem': problem_a['problem'],
                        'level': problem_a['level'],
                        'solution': problem_a['solution']
                    },
                    'problem2': {
                        'category': problem_b['category'],
                        'file_name': problem_b['file_name'],
                        'problem': problem_b['problem'],
                        'level': problem_b['level'],
                        'solution': problem_b['solution']
                    },
                    'result': result,
                    "reasoning_step": steps
                }
                comparisons.append(comparison_entry)
            else:
                print(f"未能获取题目 {idx_a} 和题目 {idx_b} 的比较结果。")
    
            # 为避免触发API速率限制，建议在请求之间稍作延迟
            time.sleep(1)  # 延迟1秒
    
            # 将比较结果保存为JSON文件
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(comparisons, f, ensure_ascii=False, indent=4)
                print(f"比较结果已保存到 {output_file}")
            except Exception as e:
                print(f"保存比较结果时出错: {e}")
    
    return comparisons,former_cnt,comparable_cnt,later_cnt
import os
import json

def count_results_in_json_folder(folder_path):
    # 用于统计不同 result 的次数
    results_count = {}

    # 遍历文件夹下所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):  # 只处理后缀为 .json 的文件
            file_path = os.path.join(folder_path, filename)

            # 读取 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)  # data 通常是一个列表
                except json.JSONDecodeError:
                    print(f"文件 {filename} 不是有效的 JSON，跳过。")
                    continue

                # 遍历列表中的每一个字典对象
                for item in data:
                    # 假设每个 item 中都有一个 'result' 字段
                    res = item.get("result")
                    if res is not None:
                        results_count[res] = results_count.get(res, 0) + 1

    return results_count

# 请修改成你的文件夹路径
folder_path = "comparisons_incontext/"  # compare 结果的一次存储
output_dir='comparisons_incontext_check/' #compare二次结果
output_file = './output_incontext.json'  # compare一次结果的合并
categories = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus"
]
# total_former_cnt =0
# total_comparable_cnt=0
# total_later_cnt=0
# level_list=[(4,5)]
# for (a_level,b_level) in level_list:
#     for c in categories:
#         _,former_cnt,comparable_cnt,later_cnt=compare_random_problems(3,3,a_level,b_level,'gpt-4o',category=c,output_dir=folder_path)
#         total_former_cnt +=former_cnt
#         total_comparable_cnt +=comparable_cnt
#         total_later_cnt +=later_cnt
#         print("now former cnt:",former_cnt)
#         print("now comparable cnt:",comparable_cnt)
#         print("now later cnt:",later_cnt)
# print("total former cnt:",total_former_cnt)
# print("total comparable cnt:",total_comparable_cnt)
# print("total later cnt:",total_later_cnt)

# results_count = count_results_in_json_folder(folder_path)

# # 打印统计结果
# print("不同 result 的出现次数：")
# for result_value, count in results_count.items():
#     print(f"'{result_value}' 出现了 {count} 次")


# import os
# import json

# def extract_and_reformat_json(input_folder, output_file):
#     # 存储最终提取出来的数据
#     output_data = []
    
#     # 遍历文件夹，找到所有的 JSON 文件
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".json"):
#             file_path = os.path.join(input_folder, filename)
            
#             # 打开每个 JSON 文件
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 print(f)
#                 data = json.load(f)
                
#                 for item in data:
#                     # 提取每个 JSON 对象中的需要字段
#                     problem1_problem = item.get('problem1', {}).get('problem', '')
#                     problem1_solution = item.get('problem1', {}).get('solution', '')
#                     problem2_problem = item.get('problem2', {}).get('problem', '')
#                     problem2_solution = item.get('problem2', {}).get('solution', '')
#                     result = item.get('result', '')
                    
#                     # 组合成新的字典对象
#                     new_object = {
#                         'problem1_problem': problem1_problem,
#                         'problem1_solution': problem1_solution,
#                         'problem2_problem': problem2_problem,
#                         'problem2_solution': problem2_solution,
#                         'result': result
#                     }
                    
#                     # 将新的字典添加到输出数据列表
#                     output_data.append(new_object)
    
#     # 将提取后的数据保存为一个新的 JSON 文件
#     with open(output_file, 'w', encoding='utf-8') as output_json:
#         json.dump(output_data, output_json, ensure_ascii=False, indent=4)

#     print(f"数据已成功提取并保存到 {output_file}")

# # 调用函数，输入文件夹路径和输出文件路径
# input_folder = folder_path
# extract_and_reformat_json(input_folder, output_file)

def check_compare(problems,model='gpt-4o', output_dir=output_dir):
    
    comparisons = []
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 生成唯一的文件名，包含时间戳和类别信息
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    former_cnt = 0
    comparable_cnt = 0 
    later_cnt = 0
    correct =0
    for idx,pair in enumerate(problems):
        prompt = createComparePrompt(
                problem1=pair['problem2'],
                answer1=pair['solution2'],
                problem2=pair['problem1'],
                answer2=pair['solution1']
        )
        ai_response = get_oai_completion(prompt, model)
        if ai_response:
                sections = ["Reasoning Steps","Conclusion"]
                steps,conclusion = parse_answer(ai_response, sections)
                result = -1
                last_result=pair['result']
                if "former" in conclusion:
                    result = "former is harder."
                    former_cnt += 1
                elif "comparable" in conclusion:
                    result = "comparable"
                    comparable_cnt+=1
                elif "later" in conclusion:
                    result = "later is harder."
                    later_cnt +=1
                if last_result=="former is harder." and result == "later is harder.":
                    correct+=1
                elif last_result=="later is harder." and result == "former is harder.":
                    correct+=1
                elif last_result=="comparable" and result == "comparable":
                    correct+=1
                else:
                    print("compare error!!!!")
                comparison_entry = {
                    'problem1':pair['problem1'],
                    'answer1':pair['solution1'],
                    'problem2':pair['problem2'],
                    'answer2':pair['solution2'],
                    'result': result,
                    'last_result':last_result,
                    "reasoning_step": steps
                }
                comparisons.append(comparison_entry)
        else:
            print(f"error.")
    
        # 为避免触发API速率限制，建议在请求之间稍作延迟
        time.sleep(1)  # 延迟1秒
    
        # 将比较结果保存为JSON文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparisons, f, ensure_ascii=False, indent=4)
            print(f"比较结果已保存到 {output_file}")
        except Exception as e:
            print(f"保存比较结果时出错: {e}")
        print("Total:",idx+1,",Correct:",correct)   

problems=load_compare_problems(output_file)
check_compare(problems)