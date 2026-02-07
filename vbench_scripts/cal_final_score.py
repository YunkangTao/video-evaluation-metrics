import io
import os
import json
import zipfile
import argparse
import shutil

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Vbench.scripts.constant import *

def submission_from_zip(model_name, zip_file):
    os.makedirs(model_name, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(model_name)
    upload_data = {}
    # load your score
    for file in os.listdir(model_name):
        if file.startswith('.') or file.startswith('__'):
            print(f"Skip the file: {file}")
            continue
        cur_file = os.path.join(model_name, file)
        if os.path.isdir(cur_file):
            for subfile in os.listdir(cur_file):
                if subfile.endswith(".json"):
                    with open(os.path.join(cur_file, subfile)) as ff:
                        cur_json = json.load(ff)
                        if isinstance(cur_json, dict):
                            for key in cur_json:
                                upload_data[key.replace('_',' ')] = cur_json[key][0]
        elif cur_file.endswith('json'):
            with open(cur_file) as ff:
                cur_json = json.load(ff)
                if isinstance(cur_json, dict):
                    for key in cur_json:
                        upload_data[key.replace('_',' ')] = cur_json[key][0]
        
        for key in TASK_INFO:
            if key not in upload_data:
                upload_data[key] = 0
    return upload_data

def submission_from_json(model_name: str, json_file: str) -> dict:
    """
    直接读取单一 JSON 文件的版本。
    """
    os.makedirs(model_name, exist_ok=True)

    # 可选：把原始 JSON 复制到指定目录
    dest_path = os.path.join(model_name, os.path.basename(json_file))
    if os.path.abspath(json_file) != os.path.abspath(dest_path):
        shutil.copyfile(json_file, dest_path)
    else:
        dest_path = json_file  # 已经在目标目录中

    with open(dest_path, "r", encoding="utf-8") as ff:
        cur_json = json.load(ff)

    upload_data = {}

    if isinstance(cur_json, dict):
        for key, value in cur_json.items():
            normalized_key = key.replace("_", " ")
            if isinstance(value, list) and value:
                upload_data[normalized_key] = value[0]
            else:
                upload_data[normalized_key] = value
    else:
        raise ValueError("JSON 文件的顶层结构需要是对象（dict）。")

    # 补齐 TASK_INFO 里要求的字段
    for task_key in TASK_INFO:
        if task_key not in upload_data:
            upload_data[task_key] = 0

    return upload_data

def get_nomalized_score(upload_data):
    # get the normalize score
    normalized_score = {}
    for key in TASK_INFO:
        min_val = NORMALIZE_DIC[key]['Min']
        max_val = NORMALIZE_DIC[key]['Max']
        normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = normalized_score[key] * DIM_WEIGHT[key]
    return normalized_score

def get_quality_score(normalized_score):
    quality_score = []
    for key in QUALITY_LIST:
        quality_score.append(normalized_score[key])
    quality_score = sum(quality_score)/sum([DIM_WEIGHT[i] for i in QUALITY_LIST])
    return quality_score

def get_semantic_score(normalized_score):
    semantic_score = []
    for key in SEMANTIC_LIST:
        semantic_score.append(normalized_score[key])
    semantic_score  = sum(semantic_score)/sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST ])
    return semantic_score

def get_final_score(quality_score,semantic_score):
    return (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Load submission file')
    parser.add_argument('--zip_file', type=str, required=True, help='Name of the zip file', default='evaluation_results.zip')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model', default='t2v_model')
    args = parser.parse_args()

    if args.zip_file.endswith('.zip'):
        upload_dict = submission_from_zip(args.model_name, args.zip_file)
    else:
        upload_dict = submission_from_json(args.model_name, args.zip_file)
    print(f"your submission info: \n{upload_dict} \n")
    normalized_score = get_nomalized_score(upload_dict)
    quality_score = get_quality_score(normalized_score)
    semantic_score = get_semantic_score(normalized_score)
    final_score = get_final_score(quality_score, semantic_score)
    print('+------------------|------------------+')
    print(f'|     quality score|{quality_score}|')
    print(f'|    semantic score|{semantic_score}|')
    print(f'|       total score|{final_score}|')
    print('+------------------|------------------+')
