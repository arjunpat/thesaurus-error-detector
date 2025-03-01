import os

import torch
from api_keys import HF_ACCESS_TOKEN
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BASE_DIRECTORY = "/scratch/users/erjones/models/postprocessed_models"
SUPPORTED_MODELS = ["7B-chat", "70B-chat", "7B", "13B-chat"]


def load_llama(model_name, device: torch.device):
    assert model_name.startswith("llama")
    llama_type = model_name[len("llama-") :]
    assert llama_type in SUPPORTED_MODELS
    if llama_type == "70B-chat":
        basedir = "/data/erjones"
    else:
        basedir = BASE_DIRECTORY
    model_name_or_path = os.path.join(basedir, llama_type)
    config = AutoConfig.from_pretrained(model_name_or_path)
    use_fast_tokenizer = "LlamaForCausalLM" not in config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left"
    )
    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.bos_token_id = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, device_map=device
    )
    return model, tokenizer


def load_llama3(model_name: str, device: torch.device | str = "auto"):
    model_id = f"meta-llama/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_ACCESS_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, token=HF_ACCESS_TOKEN
    )
    return model, tokenizer


MODELNAME2PATH = {
    "mistral-7B-instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral-8x7b-instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


def load_mistral(model_name, device: torch.device):
    assert model_name in MODELNAME2PATH
    model_name_or_path = MODELNAME2PATH[model_name]
    # config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, padding_side="left"
    )
    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.bos_token_id = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,  # device_map="cuda:0"
        device_map=device,
    )
    return model, tokenizer
