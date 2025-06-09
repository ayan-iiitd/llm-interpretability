import gc
import json
import re
import warnings

import bitsandbytes as bnb
from icecream import ic
from rich import print as pp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from captum.attr import (
    FeatureAblation,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
)

# Ignore warnings due to transformers library
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")
available_gpu_memory = 0

def get_total_available_gpu_memory_gb():
    """
    Calculates the sum of currently available (free) memory across all GPUs in gigabytes.

    Returns:
        A float representing the total available GPU memory in GB.
        Returns 0.0 if no GPUs are available or CUDA is not installed.
    """
    
    total_available_gb = 0.0
    
    if torch.cuda.is_available():
    
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s).")
    
        for i in range(num_gpus):
    
            try:
                # Switch to the current GPU context to get its memory info accurately
                torch.cuda.set_device(i)
                
                # Get free and total memory for the current device (in bytes)
                free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(i)
                
                available_gb = free_memory_bytes / (1024**2)  # Convert bytes to GB
                total_gb = total_memory_bytes / (1024**2) # Convert bytes to GB
                
                print(f"GPU {i} ({torch.cuda.get_device_name(i)}): "
                      f"{available_gb:.2f} MB free / {total_gb:.2f} MB total")
                total_available_gb += available_gb
    
            except Exception as e:
                print(f"Could not get memory info for GPU {i}: {e}")
    
    else:
        print("CUDA is not available. No GPUs found.")
    
    return total_available_gb


def load_model(model_name, bnb_config):
    
    n_gpus = torch.cuda.device_count()
    max_memory = "16000MB"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto",  
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_bnb_config():
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def captum_res(sample):
    
    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    seq_attr = dict()

    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)
    fa = FeatureAblation(model)
    llm_attr = LLMAttribution(fa, tokenizer)
    lig = LayerIntegratedGradients(model, model.model.embed_tokens)
    llm_grattr = LLMGradientAttribution(lig, tokenizer)

    seq_attr['some_data'] = sample['full_text']
    seq_attr['entities'] = sample['entities'][0]

    promptp1 = ''
    promptp2 = ''

    eval_prompt = promptp1 + sample['full_text'] + promptp2
 
    skip_tokens = [1]
    inp = TextTokenInput(
        eval_prompt, 
        tokenizer,
        skip_tokens = skip_tokens,
    )

    target = sample['entities'][0]

    attr_res = llm_attr.attribute(inp, target = target, skip_tokens = skip_tokens)

    seq_attr['input_tokens'] = attr_res.input_tokens
    seq_attr['output_tokens'] = attr_res.output_tokens
    seq_attr['input_sensitivity_seq'] = attr_res.seq_attr_dict
    seq_attr['input_sensitivity_per_output_token'] = attr_res.token_attr

    inp = TextTemplateInput(
            template = eval_prompt[:eval_prompt.find(target)] + '{}' + eval_prompt[eval_prompt.find(target) + len(target):], 
            values = [target],
        )

    attr_res = llm_attr.attribute(inp,
                                    target = target,
                                    skip_tokens = skip_tokens)
    seq_attr['input_sensitivity_entity'] = attr_res.seq_attr_dict
    seq_attr['input_sensitivity_entity_per_output_token'] = attr_res.token_attr

    inp = TextTokenInput(
        eval_prompt,
        tokenizer,
        skip_tokens = skip_tokens,
    )

    attr_res = llm_grattr.attribute(inp,
                                    target = target,
                                    skip_tokens = skip_tokens)
    seq_attr['input_sensitivity_gradient'] = attr_res.seq_attr_dict
    seq_attr['input_sensitivity_gradient_per_output_token'] = attr_res.token_attr

    del bnb_config
    del attr_res
    del inp
    del model
    del tokenizer
    del fa
    del llm_attr
    del lig
    del llm_grattr
    gc.collect()
    torch.cuda.empty_cache()

    # pp(seq_attr)

    return seq_attr