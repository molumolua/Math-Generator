import os
import json
from util import util, set_logger
from reject_sample import reject_sample
from prompt.openai_access import batch_get_chat_api
from util.util import parse_answer
from prompt.prompt_design import createAddProcessPrompt
import math
def pre_reject_fun(example):
    return createAddProcessPrompt(example['problem'],example['solution'],example['original_problem'],example['original_solution'],example['simplify_process'])

def post_fun(example, reply):
    example['complexify_process'] = reply
def main(
        start_idx=13759,
        every_save=64,
        data_path="./outputs/outputs.json",
        output_name='./outputs/process_new/outputs_process',
        batch_size=64):
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    #preprocess problems
    problems=data_list[start_idx:]
    #setup logger
    logger=set_logger.setup_logger()
    total_problems = len(problems)
    total_batch=math.ceil(total_problems / batch_size)
    logger.info(f"Loaded {total_problems} problems to add process.")
    
    output_list=[]
    output_begin=start_idx
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
        if len(output_list)>=every_save or batch +1==total_batch:
            output_end=output_begin+len(output_list)
            with open(output_name+f".json", 'w', encoding='utf-8') as output_json:
                json.dump(output_list, output_json, ensure_ascii=False, indent=4)
            output_begin=output_end
            # output_list=[]
        logger.info(f"Batch {batch + 1},Total {len(output_list)}/{min(len(problems),(batch+1)*batch_size)} has been left.")

if __name__ == "__main__":
    main()