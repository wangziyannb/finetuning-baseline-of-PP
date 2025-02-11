import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def load(base_model: str = 'llama2-7b', ckpt: str = ''):

    pruned_dict = torch.load(ckpt, map_location='cpu')
    model = pruned_dict['model']

    if device == "cuda":
        model.half()
        model = model.cuda()

    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    return model
