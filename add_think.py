import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import time
import logging
import json
import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from prompt.openai_access import batch_get_chat_api, get_oai_completion
from prompt.prompt_design import createAddThinkPrompt
from util.config import TRAINING_DATA_PATH, OUTPUT_PATH, TRAINING_DATA_PATH_AIME,MATH_DATA_PATH
from data.data_loader import load_simplify_problems
from util.util import reject_sample,extract_think_and_after
import openai
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import util, set_logger
def load_problems(data_name="MATH", iteration=0):
    problems = []
    if data_name == "MATH":
        if iteration is not None:
            now_path = MATH_DATA_PATH + "/{}".format(iteration)
        else:
            now_path = MATH_DATA_PATH  # test_data
            
    if os.path.isdir(now_path):
        for file_name in os.listdir(now_path):
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(now_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)  # Parse each line as a JSON object
                        problem = {
                            'original_problem':data.get('original_problem'),
                            'original_solution':data.get('original_solution'),
                            'problem': data.get('problem'),
                            'level': data.get('level'),
                            'solution': data.get('solution'),
                        }
                        problems.append(problem)
    return problems

def process_complex(problem, response,logger):
    problem['before_think'],problem['complex_process'] =extract_think_and_after(response)
    return problem


def add_think(model,tokenizer,logger,problems,stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>"],
         max_tokens=32768,
         output_path="/data/xucaijun/Math-Generator/outputs/newprompt_fliter_complex_question_process_deepseek.json",
         save=0
        #  model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B/full/sft"
        #  model_name_or_path="/data/xucaijun/DeepSeek-R1-Distill-Qwen-32B"
         ):

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        stop=stop_words,
        n=1
    )
    try:
        try_problems=problems
        # add think
        input_texts = [
                tokenizer.apply_chat_template(
                        [{"role": "user", "content":createAddThinkPrompt(example['problem'],example['solution'],example['original_problem'],example['original_solution'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                )
                for example in try_problems
        ]
        logger.info(input_texts[0])

        logger.info(f"Start add think.")
        generated_responses = model.generate(input_texts, sampling_params=sampling_params)
        generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]

        added_problems = [process_complex(problem,generated_response,logger) for problem,generated_response in zip(try_problems,generated_responses)]
        if save:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(added_problems, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error :{e}")
    return added_problems

def main():
    logger = set_logger.setup_logger()
    logger.info("Starting main processing loop.")
    model_name_or_path="/data/xucaijun/DeepSeek-R1-Distill-Qwen-32B"
     # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device="cuda",tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
    )
    problems=load_problems(iteration=1)
    logger.info("Model loaded successfully.")
    add_think(model,tokenizer,logger,problems,save=1)
if __name__ == "__main__":
    main()
