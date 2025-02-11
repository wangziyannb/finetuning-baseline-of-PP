import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
import pickle

import tqdm
from transformers import LlamaTokenizer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_c4(samples, cutoff_len, tokenizer, seed,seq_len,size):
    if os.path.exists(f"data/c4_{seed}_{seq_len}.json"):
        dataset = load_dataset("json", data_files=f"data/c4_{seed}_{seq_len}.json")
        if len(dataset['train']) == samples:
            print("load c4 from {}".format(f"data/c4_{seed}_{seq_len}.json"))
            return dataset
    with open(f'sampled_dataset_seed{seed}_seqlen{seq_len}_size20000.pkl', 'rb') as file:
        dataset = pickle.load(file)
    print(f"Sampling {samples} data from sampled_dataset_seed{seed}_seqlen{seq_len}_size20000.pkl")
    subdata, history = [], []
    for i in tqdm.tqdm(range(samples)):
        subdata.append({"inputs": dataset[i]['text']})
    with open(f"data/c4_{seed}_{seq_len}.json", 'w') as f:
        f.writelines(json.dumps(subdata))
    return load_dataset("json", data_files=f"data/c4_{seed}_{seq_len}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sub-dataset')
    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--seq_len', type=int, help='length of cut_off seq')
    parser.add_argument('--size', type=int, help='sampel size')
    args = parser.parse_args()
    set_random_seed(args.seed)
    # base_model = 'baffo32/decapoda-research-llama-7B-hf'
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    data = load_dataset(
        'json',
        data_files={'train': 'c4-train.00000-of-01024.json.gz'},
        split='train'
    )

    data = data.shuffle(seed=args.seed)


    def filter_by_length(example):
        tokenized_length = len(tokenizer(example['text'], add_special_tokens=False).input_ids)
        return tokenized_length > args.seq_len


    filtered_data = data.filter(filter_by_length)

    if len(filtered_data) < args.size:
        print(f"not enough for {args.size} items")
    else:
        data_sampled = filtered_data.select(range(args.size))

        with open(f'sampled_dataset_seed{args.seed}_seqlen{args.seq_len}_size{args.size}.pkl', 'wb') as file:
            pickle.dump(data_sampled, file)
    get_c4(args.size, args.seq_len, tokenizer, args.seed, args.seq_len, args.size)
