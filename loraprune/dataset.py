import os.path
import pickle

from datasets import load_dataset
import json
import random
import tqdm


def get_c4(samples, cutoff_len, tokenizer, seed):
    if os.path.exists(f"data/c4_{seed}_{cutoff_len}.json"):
        dataset = load_dataset("json", data_files=f"data/c4_{seed}_{cutoff_len}.json")
        if len(dataset['train']) == samples:
            print("load c4 from {}".format(f"data/c4_{seed}_{cutoff_len}.json"))
            return dataset

    # dataset = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    with open(f'sampled_dataset_seed{seed}_seqlen{cutoff_len}_size20000.pkl', 'rb') as file:
        print(f"load dataset: 'sampled_dataset_seed{seed}_seqlen{cutoff_len}_size20000.pkl'")
        dataset = pickle.load(file)

    print("Sampling {} data from c4".format(samples))
    subdata, history = [], []
    for _ in tqdm.tqdm(range(samples)):
        while True:
            i = random.randint(0, len(dataset) - 1)
            trainenc = tokenizer(dataset[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > cutoff_len and i not in history:
                history.append(i)
                break
        subdata.append({"inputs": dataset[i]['text']})
    with open(f'data/c4_{seed}_{cutoff_len}.json', 'w') as f:
        f.writelines(json.dumps(subdata))
    return load_dataset("json", data_files=f"data/c4_{seed}_{cutoff_len}.json")


def get_bookcorpus(samples, cutoff_len, tokenizer):
    if os.path.exists("data/bookcorpus.json"):
        dataset = load_dataset("json", data_files="data/bookcorpus.json")
        if len(dataset) == samples:
            print("load bookcorpus from".format("data/bookcorpus.json"))
            return dataset

    dataset = load_dataset('bookcorpus', split='train')
    print("Sampling {} data from bookcorpus".format(samples))
    # dataset = "".join(dataset['text'])
    subdata, history = [], []
    for _ in tqdm.tqdm(range(samples)):
        stop = False
        while not stop:
            i = random.randint(0, len(dataset) - 2)
            if i in history:
                continue
            history.append(i)
            current_text = dataset[i]['text']
            sh = []
            for j in range(i + 1, len(dataset) - 1):
                sh.append(j)
                if j in history:
                    break
                current_text += dataset[j]['text']
                trainenc = tokenizer(current_text, return_tensors='pt')
                if trainenc.input_ids.shape[1] > cutoff_len:
                    stop = True
                    history.extend(sh)
                    break
        subdata.append({"inputs": current_text})
    with open('data/bookcorpus.json', 'w') as f:
        f.writelines(json.dumps(subdata))
    return load_dataset("json", data_files="data/bookcorpus.json")


get_dataset = {'c4': get_c4,
               'bookcorpus': get_bookcorpus}
