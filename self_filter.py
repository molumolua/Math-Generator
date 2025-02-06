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
from prompt.prompt_design import createComparePrompt, createSimpleQuestionPromptV3, createAnsqerPrompt
from util.config import TRAINING_DATA_PATH, OUTPUT_PATH, TRAINING_DATA_PATH_AIME,MATH_DATA_PATH
from data.data_loader import load_simplify_problems
from util.util import reject_sample
import openai
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import util, set_logger


def process_reject_sample(problem, response,logger):
    try:
        if problem and problem['complex_solution'] and response:
            result = reject_sample(response,problem['complex_solution'])
            return result
        else:
            logger.warning(f"Missing data for reject sample.")
            return False
    except Exception as e:
        logger.error(f"Error in process_reject_sample.{e}")
        return False

def process_compare(problem, sections, response1,response2,logger):
    try:
        value=0
        if problem and response1 and response2:
            _,conclusion = util.parse_answer(response1, sections, logger)
            if "former" in conclusion:
                value+=1
            elif "later" in conclusion:
                value-=1
            _,conclusion = util.parse_answer(response2, sections, logger)
            if "later" in conclusion:
                value+=1
            elif "former" in conclusion:
                value-=1
            # logger.info(f"value:{value}")
            return value<0
        logger.warning(f"Invalid data for compare.")
        return False
    except Exception as e:
        logger.error(f"Error in process_compare: {e}")
        return False

def self_filter(model,tokenizer,logger,stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>"],
         max_tokens=32768,
         device="cuda",
         data_path="/data/xucaijun/Math-Generator/outputs/newprompt_complex_question_process_deepseek.json",
         output_path="/data/xucaijun/Math-Generator/outputs/newprompt_fliter_complex_question_process_deepseek.json",
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
    with open(data_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    problems=problems
    try:
        try_problems=problems
        # reject sample
        input_texts = [
                tokenizer.apply_chat_template(
                        [{"role": "user", "content": createAnsqerPrompt(problem['complex_problem'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                )
                for problem in try_problems
        ]
        logger.info(input_texts[0])

        logger.info(f"Start reject sample.")
        generated_responses = model.generate(input_texts, sampling_params=sampling_params)
        generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]

        reject_sampled_problems = [problem for problem,generated_response in zip(try_problems,generated_responses) if process_reject_sample(problem,generated_response,logger)]
        logger.info(f"{len(reject_sampled_problems)} problems pass reject sample.")
        logger.info(f" {len(try_problems)- len(reject_sampled_problems)} problems fail in reject sample.")

        input_texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": createComparePrompt(problem['original_problem'], problem['original_solution'], problem['complex_problem'], problem['complex_solution'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for problem in reject_sampled_problems
        ]
        logger.info(input_texts[0])
        logger.info(f"Start compare.")
        generated_responses = model.generate(input_texts, sampling_params=sampling_params)
        generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]

        #reversed compare
        reversed_input_texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": createComparePrompt(problem['complex_problem'], problem['complex_solution'], problem['original_problem'], problem['original_solution'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for problem in reject_sampled_problems
        ]
        logger.info(reversed_input_texts[0])
        logger.info(f"Start reversed compare.")
        reversed_generated_responses = model.generate(reversed_input_texts, sampling_params=sampling_params)
        reversed_generated_responses = [generated_response.outputs[0].text for generated_response in reversed_generated_responses]
                
        sections = [ "Reasoning Steps","Conclusion"]
        compared_problems = [problem for problem,generated_response,reversed_generated_response \
                                     in zip(reject_sampled_problems,generated_responses,reversed_generated_responses) if process_compare(problem,sections,generated_response,reversed_generated_response,logger)]
        logger.info(f" {len(compared_problems)} problems pass compare.")
        logger.info(f" {len(reject_sampled_problems)- len(compared_problems)} problems fail in compare.")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(compared_problems, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error :{e}")

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
    logger.info("Model loaded successfully.")
    self_filter(model,tokenizer,logger)
if __name__ == "__main__":
    main()
