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
                            "content": "<think>\n{}\n</think>\n\n**Complexified Problem**:\n{}\n\n**Complexified Solution**:\n{}\n".format(data['complexify_process'],data['original_problem'],data['original_solution'])
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
    now_path="/data/xucaijun/New/Math-Generator/outputs/second_iter_deepseek_answer.json"
    result_path="/data/xucaijun/New/Math-Generator/outputs/second_iter_test.json"
    file_path = os.path.join(now_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        # problems=[ problem for problem in data_list if problem['value']==True]
        problems=[]
        for data in data_list:
            for problem in data:
                if problem['complex_problem'] != problem['original_problem']:
                    problems.append(problem)
        process_train_data(problems,output_path=result_path,prompt_type="qwen_math",sections=['complex_problem','complex_solution'])
    # problems = load_simplify_problems()
    # print(len(problems))
    # process_train_data(problems,output_path=result_path,prompt_type="qwen_math")
if __name__ =="__main__":
    main()