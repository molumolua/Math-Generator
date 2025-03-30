import os
import json
from prompt.prompt_design import createComplexQuestionProcessPrompt,createComplexQuestionPrompt
from data.data_loader import load_problems,load_simplify_problems
def process_train_data(data_list,output_path=None,prompt_type="generate_data",sections=["complex_problem","complex_solution"]):
    problems=[]
    for data in data_list:   
            if prompt_type=="generate_data":
                problem={
                    'messages':[
                        {
                            "role": "user",
                            "content": createComplexQuestionProcessPrompt(data['problem'],data['solution'])
                        },
                        {
                            "role": "assistant",
                            "content": "**Complexification Process**:\n{}\n**Complexified Problem**:\n{}\n\n**Complexified Solution**:\n{}\n".format(data['complexify_process'],data['original_problem'],data['original_solution'])
                        }
                    ]
                }
            elif prompt_type=="think":
                problem={
                    'messages':[
                        {
                            "role": "user",
                            "content": createComplexQuestionPrompt(data['problem'],data['solution'])
                        },
                        {
                            "role": "assistant",
                            "content": "<think>\n{}\n</think>\n\n**Complexified Problem**:\n{}\n\n**Complexified Solution**:\n{}\n".format(data['test_complex_think'],data['original_problem'],data['original_solution'])
                        }
                    ]
                }
            elif prompt_type=="test_data":
                problem={
                    'messages':[
                        {
                            "role": "user",
                            "content": data['complex_problem']
                        },
                        {
                            "role": "assistant",
                            "content": data['complex_solution']
                        }
                    ]
                }
            elif prompt_type=="qwen_math":
                problem={
                    'messages':[
                        {
                            "role": "system",
                            "content": "Please reason step by step, and put your final answer within \\boxed{{}}."
                        },
                        {
                            "role": "user",
                            "content": data[sections[0]]
                        },
                        {
                            "role": "assistant",
                            "content": data[sections[1]]
                        }
                    ]
                }
            else:
                raise ValueError("Not support prompt type.")
            problems.append(problem)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as output_json:
            json.dump(problems, output_json, ensure_ascii=False, indent=4)
    return problems
def main():
    file_path="/data2/xucaijun/Math-Generator/outputs/7b-generate-1.5b-reject.json"
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
    for i in range(11):
        result_path=f"./outputs/level-test/7b_correct_num_{i}.json"
        process_train_data(problems[i],output_path=result_path,prompt_type="qwen_math",sections=['complex_problem','complex_solution'])
    
    total_problems=sum([problems[1],problems[2],problems[3],problems[4],problems[5]],[])
    result_path=f"./outputs/level-test/7b_correct_num_11.json"
    process_train_data(total_problems,output_path=result_path,prompt_type="qwen_math",sections=['complex_problem','complex_solution'])

        # with open(result_path, 'w', encoding='utf-8') as output_json:
        #         json.dump(problems[i], output_json, ensure_ascii=False, indent=4)

    # file_path="/data/xucaijun/New/Math-Generator/outputs/tmp_2.json"
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data_list = json.load(f)
    #     for data in data_list:
    #         for problem in data:
    #             if problem['complex_problem'] != problem['original_problem']:
    #                 problems.append(problem)

    print(f"len:{len(problems)}")

    # file_path="/data/xucaijun/New/Math-Generator/deepseek-math/1/simplify_problem.json"
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data_list = json.load(f)
    #     for data in data_list:
    #         tmp_key=[]
    #         for problem in data:
    #             if problem['problem'] != problem['original_problem'] and problem['problem'] not in tmp_key:
    #                 tmp_key.append(problem['problem'])
    #                 problems.append(problem)

    # process_train_data(problems,output_path=result_path,prompt_type="qwen_math",sections=['complex_problem','complex_solution'])
    # problems = load_simplify_problems()
    # print(len(problems))
    # process_train_data(problems,output_path=result_path,prompt_type="qwen_math")
if __name__ =="__main__":
    main()