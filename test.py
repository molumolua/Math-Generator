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


import os
import json
from util import util, set_logger
from reject_sample import reject_sample
from prompt.openai_access import batch_get_chat_api
from util.util import parse_answer
from prompt.prompt_design import createAddProcessPrompt_2
import math
def pre_reject_fun(example):
    return createAddProcessPrompt_2(example['original_problem'],example['original_solution'],example['problem'],example['solution'])

def post_fun(example, reply):
    example['simplify_process'] = reply
def main(
        data_path="./outputs/outputs_backup.json",
        output_path='./outputs/outputs_fixed.json',
        batch_size=64):
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    data_list=data_list
    #preprocess problems
    problems=[]
    for data in data_list:
        problems.append(data)
    problems=problems
    #setup logger
    logger=set_logger.setup_logger()
    total_problems = len(problems)
    total_batch=math.ceil(total_problems / batch_size)
    logger.info(f"Loaded {total_problems} problems to add process.")
    output_list=[]
    for batch in range(total_batch):
        batch_problems = problems[batch * batch_size:(batch + 1) * batch_size]
        logger.info(f"Batch {batch + 1}, Starting add process.")
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
        
        output_list+=batch_problems
        with open(output_path, 'w', encoding='utf-8') as output_json:
            json.dump(output_list, output_json, ensure_ascii=False, indent=4)
        logger.info(f"Batch {batch + 1},Total {len(output_list)}/{min(len(problems),(batch+1)*batch_size)} has been left.")

if __name__ == "__main__":
    main()