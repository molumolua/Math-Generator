import os
import json
from prompt.prompt_design import createComplexQuestionProcessPrompt
problems = []
now_path="/home/bingxing2/home/scx8q73/jobs/Math-Generator/outputs/filter_complex_question_process_1.5b_math.json"
result_path="/home/bingxing2/home/scx8q73/jobs/Math-Generator/outputs/train_data_filter_complex_question_process_1.5b_math.json"
file_path = os.path.join(now_path)
with open(file_path, 'r', encoding='utf-8') as f:
    data_list = json.load(f)
    for data in data_list:   
        problem={
            'messages':[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
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
        problems.append(problem)
with open(result_path, 'w', encoding='utf-8') as output_json:
    json.dump(problems, output_json, ensure_ascii=False, indent=4)