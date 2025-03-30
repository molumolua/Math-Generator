import os
import json
from util.util import reject_sample
from process_train_data import process_train_data
from tqdm import tqdm
TEST_LIST=[f"\\boxed{i}" for i in range(-1,11)]
def filter_easy_problems(problems,section="complex_solution"):
    filter_hard_problems=[]
    for problem in tqdm(problems):
        FLAG=True
        for test_answer in TEST_LIST:
            if reject_sample(test_answer,problem[section],timeout=False):
                FLAG=False
                break
        if FLAG:
            filter_hard_problems.append(problem)
    return filter_hard_problems
def main():
    # file_path="/data2/xucaijun/Math-Generator/outputs/1-glm-generate-1.5b-reject.json"
    # result_path="/data2/xucaijun/Math-Generator/outputs/"

    # problems=[]
    # problems=[[] for i in range(11)]
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data_list = json.load(f)
    #     for data in data_list:
    #             problem=data
    #             problems[problem['correct_num']].append(data)
    # total_problems=[]
    # for hard_level in range(1,11):
    #     now_problems=filter_easy_problems(problems[hard_level],"complex_solution")
    #     result_path=f"./outputs/level-test/correct_num_{hard_level}.json"
    #     process_train_data(now_problems,output_path=result_path,prompt_type="qwen_math",sections=['complex_problem','complex_solution'])
    # total_problems=[]
    # for hard_level in range(1,5):
    #     file_path=f"./outputs/level-test/correct_num_{hard_level}.json"
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #          now_problems=json.load(f)
    #          total_problems.extend(now_problems)
    # result_path=f"./outputs/level-test/correct_num_12.json"
    # with open(result_path, 'w', encoding='utf-8') as output_json:
    #     json.dump(total_problems, output_json, ensure_ascii=False, indent=4)
    # print(f"len:{len(total_problems)}")
    file_path="/data2/xucaijun/Math-Generator/outputs/1-glm-generate-1.5b-reject.json"
    # result_path="/data/xucaijun/New/Math-Generator/outputs/open-r1-second_iter.json"

    problems=[]
    problems=[[] for i in range(11)]
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        for data in data_list:
                problem=data
                problems[problem['correct_num']].append(data)

    # file_path="./outputs/second_iter_fenbu.json"
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data_list = json.load(f)
    #     for data in data_list:
    #             problem=data
    #             problems[problem['correct_num']].append(data)
    # for i in range(11):
    #     result_path=f"./outputs/level-test/7b_correct_num_{i}.json"
    #     process_train_data(problems[i],output_path=result_path,prompt_type="qwen_math",sections=['complex_problem','complex_solution'])
    
    total_problems=sum([problems[6],problems[7],problems[8],problems[9]],[])
    result_path=f"./outputs/level-test/correct_num_13.json"
    process_train_data(total_problems,output_path=result_path,prompt_type="qwen_math",sections=['complex_problem','complex_solution'])

    # # total_problems=filter_easy_problems(total_problems)
    # result_path=f"./outputs/level-test/7b_correct_num_12.json"
    # process_train_data(total_problems,output_path=result_path,prompt_type="qwen_math",sections=['complex_problem','complex_solution'])

if __name__ =="__main__":
    main()