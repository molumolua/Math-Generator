# main.py
import time
import json
from util import config, grader, graph, util
from util.util import remove_boxed
import csv
import sys
import os
import json
import random
import ast
import re
import logging
import sys
from tqdm import tqdm
from data.data_loader import load_problems
from prompt.openai_access import call_chatgpt,get_oai_completion
from util.config import TRAINING_DATA_PATH,OUTPUT_PATH
from prompt.prompt_design import createSimpleQuestionPrompt,createComparePrompt,createSimpleQuestionPromptV2
from prompt.new_prompt_design import createConstructPrompt,createProblemPrompt,createReConstructPrompt,createCheckPrompt
from util.graph import str2graph,findEraseProcess,constrcutDatafromMethod

MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if util.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def load_csv_rows(csv_file_path):
    """
    Generator that yields each row of the CSV as a dictionary.
    """
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row

def process_rows(csv_file_path, json_output_path, model="gpt-4o",start = 0,end=100):
    """
    Processes each row of the CSV, compares GPT's answer with the original answer,
    and writes matched results to a JSON file.
    """
    matched_answers = []
    cnt =0
    for row in load_csv_rows(csv_file_path):
        cnt +=1
        if cnt <= start:
            continue
        question = row['Question']
        original_answer = row['Answer']
        prompt = f"Problem: {question}\nProvide a detailed solution and Output the final number of the answer in the latex boxed format."

        print(f"{cnt},Processing ID {row['ID']}...")

        gpt_answer = get_oai_completion(prompt, model=model)
        
        if gpt_answer is not None:
            # Extract the numerical answer from GPT's response
            temp_ans = remove_boxed(util.last_boxed_only_string(gpt_answer))
            if temp_ans:
                if util.is_equiv(temp_ans,original_answer):
                    matched_answers.append({
                        'ID': row['ID'],
                        'Year': row['Year'],
                        'Problem Number': row['Problem Number'],
                        'problem': row['Question'],
                        'Original Answer': original_answer,
                        'solution': gpt_answer
                    })
                    print(f"Match found for ID {row['ID']}: {temp_ans}")
                    # Write matched answers to JSON
                    with open(json_output_path, 'w', encoding='utf-8') as jsonfile:
                        json.dump(matched_answers, jsonfile, indent=4)
                else:
                    print(f"No match for ID {row['ID']}: GPT Answer = {temp_ans}, Original Answer = {original_answer}")
            else:
                print(f"No numerical answer found in GPT response for ID {row['ID']}")
        else:
            print(f"No answer returned for ID {row['ID']}")

        # To respect API rate limits
        time.sleep(1)
        if cnt>=end:
            break
    # Write matched answers to JSON
    with open(json_output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(matched_answers, jsonfile, indent=4)
    print(f"Matched answers have been written to {json_output_path}")

if __name__ == "__main__":
    csv_file_path = 'AIME\AIME_Dataset_1983_2024.csv'       # Replace with your actual CSV file path
    json_output_path = 'AIME\matched_answers.json'  # Desired JSON output file
    process_rows(csv_file_path, json_output_path,model="gpt-4o",start=200,end=200)
