from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from util import util, set_logger
import csv
import sys
import os
import json
import random
import sys
from tqdm import tqdm
from data.data_loader import load_problems
from prompt.prompt_design import createComplexQuestionPrompt
import math

def main(batch_size=64,
         device="cuda",
         output_path="./outputs/complex_question.json",
         model_name_or_path="/home/bingxing2/home/scx8q73/jobs/LLaMA-Factory-main/models/qwen2.5-Math-1.5b/full/sft"):
    logger = set_logger.setup_logger()
    logger.info("Starting the process...")

    # Load problems
    logger.info("Loading problems...")
    problems = load_problems(iteration=None, min_level=1, max_level=5)
    logger.info(f"Loaded {len(problems)} problems.")

    # Load model and tokenizer
    logger.info(f"Loading model from {model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    logger.info("Model and tokenizer loaded.")

    output_list = []
    total_batch = math.ceil(len(problems) / batch_size)
    logger.info(f"Total batches: {total_batch}")

    for batch_idx in range(total_batch):
        logger.info(f"Processing batch {batch_idx + 1}/{total_batch}...")

        # Batch process problems into message
        messages_batch = []
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(problems))
        now_problems = problems[start_idx:end_idx]

        for problem in now_problems:
            message = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": createComplexQuestionPrompt(problem['problem'], problem['solution'])
                }
            ]
            messages_batch.append(message)
        # 使用apply_chat_template批量处理
        texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        # 将处理后的文本转化为模型输入
        model_inputs = tokenizer(texts, return_tensors="pt",padding=True,truncation=True).to(device)
        
        # 生成文本
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 解码生成的文本
        generated_responses = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)


        # 输出生成的结果
        for problem, response in zip(now_problems, generated_responses):
            complex_problem, complex_solution = util.parse_answer(response, ["Complexified Problem", "Complexified Solution"], logger=logger)
            output_object = {
                "original_problem": problem['problem'],
                "original_solution": problem['solution'],
                "complex_problem": complex_problem,
                "complex_solution": complex_solution,
                "response":response
            }
            output_list.append(output_object)

        logger.info(f"Batch {batch_idx + 1}/{total_batch} processed.")
        # Save the output to a JSON file
        logger.info(f"Saving output to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as output_json:
            json.dump(output_list, output_json, ensure_ascii=False, indent=4)

    logger.info("Process completed.")

if __name__ == "__main__":
    main()
