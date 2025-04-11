import torch
import os
import random

import numpy as np
import math
from tqdm import tqdm
import json
from datasets import load_dataset, concatenate_datasets
from fusion.model.builder import load_pretrained_model
from fusion.mm_utils import get_model_name_from_path, process_images

from argparse import ArgumentParser

from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils import call_fusion_engine_df, fusion_image_processor
from utils.eval_utils import parse_multi_choice_response, parse_open_response

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_8bit', type=bool, default=False)

    args = parser.parse_args()
    # set_seed(args.seed)

    print('fusion_initializing...')
    processor = None
    call_model_engine = call_fusion_engine_df
    vis_process_func = fusion_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    subjects = [x for x in CAT_SHORT2LONG.values()]
    for subject in tqdm(subjects):
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    sub_dataset_list = get_chunk(sub_dataset_list, args.num_chunks, args.chunk_idx)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    # load model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, None, model_name)

    out_samples = []
    for sample in tqdm(dataset):
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        if sample['image']:
            image_tensor = process_images([sample['image']], vis_processors, model.config)[0]       
            sample['image'] = image_tensor  
        else:
            image_tensor = torch.zeros(5, 3, 336, 336)
            sample['image'] = image_tensor  

        with torch.no_grad():  
            response = call_model_engine(args, sample, model, tokenizer, processor)
            if sample['question_type'] == 'multiple-choice':
                parsed_pred = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
                out_sample = {
                    'id': sample['id'],
                    'question_type': sample['question_type'],
                    'answer': sample['answer'],
                    'response': response,
                    'parsed_pred': parsed_pred,
                    'index2ans': sample['index2ans'],
                    'question': sample['final_input_prompt'],
                }
            else:  
                parsed_pred = parse_open_response(response)
                out_sample = {
                    'id': sample['id'],
                    'question_type': sample['question_type'],
                    'answer': sample['answer'],
                    'response': response,
                    'parsed_pred': parsed_pred,
                    'question': sample['final_input_prompt'],
                }
            out_samples.append(out_sample) 

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, sample in enumerate(out_samples):
        ans_file.write(json.dumps(sample) + "\n")
    ans_file.close()


if __name__ == '__main__':
    main()
