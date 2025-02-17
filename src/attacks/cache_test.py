import torch
from src.attacks.redgreen_test import find_common_prefix, find_sublist_idx, CacheLogit, generate_completion, identify_fruit
from src.utils import generate_random_prefix
from src.watermarks.no_watermark import No_watermark
import re
import numpy as np
import random


def get_cacheTest_phase1(model_name):
    
    model_name = model_name.replace("/", "_")
    
    model_data = {
        "mistralai_Mistral-7B-Instruct-v0.1": {
            "word_list": ["apples", "pears"],
            "example": "strawberries",
            "format": ""
        },
        "meta-llama_Llama-2-13b-chat-hf": {
            "word_list": ["peaches", "plums"],
            "example": "strawberries",
            "format": "Sure!"
        },
        "meta-llama_Llama-2-70b-chat-hf": {
            "word_list": ["cherries", "apricots"],
            "example": "strawberries",
            "format": "Sure!"
        }, 
        "meta-llama_Llama-2-7b-chat-hf": { 
            "word_list": ["peaches", "plums"],
            "example": "strawberries",
            "format": "Sure!"
        },
        "meta-llama_Meta-Llama-3-8B-Instruct": { 
            "word_list": ["cherries", "apricots"],
            "example": "strawberries",
            "format": "Sure!"
        },
        "BAAI_Infinity-Instruct-3M-0625-Yi-1.5-9B": {
            "word_list": ["cherries", "apricots"],
            "example": "strawberries",
            "format": "Sure!"
        },
        "Qwen_Qwen2-7B-Instruct": {
            "word_list": ["cherries", "apples"],
            "example": "strawberries",
            "format": ""
        },
    }

    if model_name in model_data:
        return model_data[model_name]
    else:
        return None
    

    
def generate_data_cache_test(watermark, temperature, model_name, n_samples: int, use_cache: bool, fast: bool):
    
    prefix = "I ate"
    
    if use_cache:
        phase1_model = No_watermark(watermark.model, watermark.tokenizer)
    else:
        phase1_model = watermark
    
    word_list, example, format = get_cacheTest_phase1(model_name).values()
    is_phase1 = True
    
    logit_cache = CacheLogit(word_list=word_list)
    
    while is_phase1:
        
        phase1_prob = [0] * len(word_list)
        
        k = generate_random_prefix()
        prompt = f"Complete the sentence \"{prefix} {k}\" using only and exacty a random word from the list: {word_list}.  Answer in this speific format: {format} {prefix} {k} {example}. (here I chose an other fruit for the sake of the example, you have to choose among {word_list})."

        prompts = [prompt] * n_samples
        
        logit_cache.store_prompt_variable(prefix, k)
        
        out = [generate_completion(prompt, phase1_model, temperature, fast, logit_cache) for prompt in prompts]

        success = True
        for response in out:
            fruit_number = identify_fruit(response, word_list)

            safeguard = 0
            while fruit_number is None:
                print("Parsing error - Rejection sampling")
                response = generate_completion(prompt, phase1_model, temperature, fast, logit_cache)

                fruit_number = identify_fruit(response, word_list)
                safeguard += 1
                if safeguard > 10:
                    success = False
                    break
            if success: 
                phase1_prob[fruit_number] += 1   
                is_phase1 = False 
            else: # Failure of Phase1
                is_phase1 = True 
                break
            
    # Phase2
            
    phase2_prob = [0] * len(word_list)
    logit_cache = CacheLogit(word_list=word_list)
    
    prompts = [prompt] * n_samples
        
    logit_cache.store_prompt_variable(prefix, k)
    
    out = [generate_completion(prompt, watermark, temperature, fast, logit_cache) for prompt in prompts]

    for response in out:
        fruit_number = identify_fruit(response, word_list)

        safeguard = 0
        while fruit_number is None:
            print("Parsing error - Rejection sampling")
            response = generate_completion(prompt, watermark, temperature, fast, logit_cache)

            fruit_number = identify_fruit(response, word_list)
            safeguard += 1
            
            if safeguard > 50:
                fruit_number = random.randint(0, len(word_list) - 1)
                break
            
        phase2_prob[fruit_number] += 1   

    return phase1_prob, phase2_prob