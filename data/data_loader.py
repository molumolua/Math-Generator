# data_loader.py

import os
import json
from util.config import TRAINING_DATA_PATH,TRAINING_DATA_PATH_AIME

def load_problems(iteration=0,search_keys=None,min_level=None,max_level=None):
    if iteration!=None:
        now_path = TRAINING_DATA_PATH +"/{}".format(iteration)
    else:
        now_path = TRAINING_DATA_PATH
    problems = []
    for category in os.listdir(now_path):
        category_path = os.path.join(now_path, category)
        if os.path.isdir(category_path):
            for file_name in os.listdir(category_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(category_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        problem={
                            'category': category,
                            'file_name': file_name,
                            'problem': data.get('problem'),
                            'level':data.get('level'),
                            'solution': data.get('solution'),
                            'last_problem':data.get("last_problem"),
                            'last_solution':data.get("last_solution")
                        }
                        key ={
                            'category': category,
                            'file_name': file_name
                        }
                        level=data.get('level')
                        if max_level:
                            if level and (not level[-1].isdigit() or int(level[-1])>max_level):
                                continue
                        if min_level:
                            if level and (not level[-1].isdigit() or int(level[-1])<min_level):
                                continue
                        if (not search_keys) or (key in search_keys):
                            problems.append(problem)
    return problems

def load_aime_problems(iteration=0,search_keys=None):
    now_path = TRAINING_DATA_PATH_AIME +"/{}".format(iteration)
    problems = []
    for file_name in os.listdir(now_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(now_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                # print(len(data_list))
                for data in data_list:   
                    problem={
                        'ID':data.get("ID"),
                        'problem': data.get('problem'),
                        'solution': data.get('solution'),
                        'last_problem':data.get("last_problem"),
                        'last_solution':data.get("last_solution")
                    }
                    key ={
                        'ID':data.get("ID"),
                    }
                    if (not search_keys) or (key in search_keys):
                        problems.append(problem)
    return problems
