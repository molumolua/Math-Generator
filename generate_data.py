import time
import json
from util import util, set_logger
import csv
import sys
import os
import random
import math
from tqdm import tqdm
from data.data_loader import load_problems
from prompt.prompt_design import createComplexQuestionProcessPrompt
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from filter_problems import filter_problems
from process_train_data import process_train_data


def main(stop_words = ["</s>", "<｜Assistant｜>", "<|endoftext|>","\n**Complexification Process**"],
         max_tokens=4096,
         max_try=3,
         enable_filter=True,
         use_chat_templete=False,
         device="cuda",
         input_path=None,
         output_path="./outputs/complex_question_process_deepseek.json",
         model_name_or_path="/data/xucaijun/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-32B/full/sft"):
    logger = set_logger.setup_logger()
    logger.info("Starting the process...")

    # Load problems
    logger.info("Loading problems...")
    if input_path:
        with open(input_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)
    else:
        problems = load_problems(iteration=None, min_level=1, max_level=5)
    problems= problems[:10]
    logger.info(f"Loaded {len(problems)} problems.")

    # Load vLLM model
    logger.info(f"Loading model from {model_name_or_path}...")
    model = LLM(model_name_or_path, device=device)
    if use_chat_templete:
        tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, trust_remote_code=True
        )
    logger.info("Model loaded successfully.")

    # Define sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        stop=stop_words,
        n=1
    )

    # Apply_chat_template
    output_list = []
    now_problems = problems
    for _ in range(max_try):
        if len(now_problems)==0:
            break
        logger.info(f"{len(now_problems)} problems for {_} th try generating data...")
        if use_chat_templete:
            input_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": createComplexQuestionProcessPrompt(problem['problem'], problem['solution'])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for problem in now_problems
            ]
        else:
            input_texts= [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+ \
                createComplexQuestionProcessPrompt(problem['problem'], problem['solution'])+ \
                "<|im_end|>\n<|im_start|>assistant\n"
                for problem in now_problems
            ]
        logger.info(input_texts[0])
            
        
        # Generate responses using vLLM
        logger.info("Generating responses...")
        generated_responses = model.generate(input_texts, sampling_params=sampling_params)
        generated_responses = [generated_response.outputs[0].text for generated_response in generated_responses]
        # Process the generated responses
        for problem, response in zip(now_problems, generated_responses):
            process, complex_problem, complex_solution = util.parse_answer(response, 
                                                                            ["Complexification Process", 
                                                                            "Complexified Problem", 
                                                                            "Complexified Solution"], 
                                                                            logger=logger)
            output_object = {
                "original_problem": problem['problem'],
                "original_solution": problem['solution'],
                "complex_problem": complex_problem,
                "complex_solution": complex_solution,
                "response": response,
                "Complexification Process": process
            }
            output_list.append(output_object)
        if enable_filter:
            output_list,now_problems=filter_problems(data_list=output_list,logger=logger)
        else:
            now_problems=[]

        # Save the output to a JSON file
        logger.info(f"Saving output to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as output_json:
            json.dump(output_list, output_json, ensure_ascii=False, indent=4)

    logger.info("Process completed.")


if __name__ == "__main__":
    main()
