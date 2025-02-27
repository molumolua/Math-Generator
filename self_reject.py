import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5,6,7"
import time
import logging
import json
import re
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from prompt.openai_access import batch_get_chat_api, get_oai_completion
from prompt.prompt_design import createComparePrompt, createSimpleQuestionPromptV3, createAnsqerPrompt
from util.config import TRAINING_DATA_PATH, OUTPUT_PATH, TRAINING_DATA_PATH_AIME,MATH_DATA_PATH
from data.data_loader import load_simplify_problems
from util.util import reject_sample
import openai
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import util, set_logger
from self_filter import self_filter


def process_reject_sample(problem, response,logger):
    problem['output']=response
    try:
        if problem and problem['solution'] and response:
            result = reject_sample(response,problem['solution'])
            problem['value']=result
        else:
            logger.warning(f"Missing data for reject sample.")
            problem['value']=False
    except Exception as e:
        logger.error(f"Error in process_reject_sample.{e}")
        problem['value']=False
    return problem

def main(model,tokenizer,logger,stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>"],
         max_tokens=32768,
         device="cuda",
         output_path="/data/xucaijun/Math-Generator/outputs/math_output_deepseek.json",
        #  model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B/full/sft"
         model_name_or_path="/data/xucaijun/DeepSeek-R1-Distill-Qwen-32B"
         ):

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        stop=stop_words,
        n=1
    )
    problems=load_simplify_problems()
    problems=problems
    correct_num=0
    try:
        try_problems=problems
        # reject sample
        input_texts = [
                tokenizer.apply_chat_template(
                        [{"role": "user", "content": createAnsqerPrompt(problem['problem'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                )
                for problem in try_problems
        ]
        logger.info(input_texts[0])

        logger.info(f"Start reject sample.")
        generated_responses = model.generate(input_texts, sampling_params=sampling_params)
        generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]

        reject_sampled_problems = [process_reject_sample(problem,generated_response,logger) for problem,generated_response in zip(try_problems,generated_responses)]
 

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reject_sampled_problems, f, ensure_ascii=False, indent=4)
        return sum([reject_sampled_problem['value'] for reject_sampled_problem in reject_sampled_problems])
    except Exception as e:
        logger.error(f"Error :{e}")
        return 0

if __name__ == "__main__":
    logger = set_logger.setup_logger()
    logger.info("Starting main processing loop.")
    model_name_or_path="/data/xucaijun/DeepSeek-R1-Distill-Qwen-32B"
     # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device="cuda",tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
    )
    logger.info("Model loaded successfully.")
    correct_num=main(model,tokenizer,logger)
    self_filter(model,tokenizer,logger)
    print(f"correct num:{correct_num}")