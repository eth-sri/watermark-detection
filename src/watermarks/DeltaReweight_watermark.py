from typing import List
import torch
import torch.nn.functional as F
from src.watermarks.KTH_watermark import DummyGenerationOutput
import random

class DeltaReweight_watermark():
    
    def __init__(self, model, tokenizer, context_size: int, seeds: List = [0], use_cache: bool = True, disable_watermark_every: int = 0):
        
        self.model = model
        self.tokenizer = tokenizer
        self.seeds = seeds
        self.context_size = context_size
        self.cache = {}
        self.use_cache = True
        self.randomize_every = disable_watermark_every
        self.count_request = 0
        
    def generate(self, model_inputs, temperature, max_new_tokens) -> List:
        
        outputs = []
            
        generation_output = self.generate_key(
            model_inputs, temperature, max_new_tokens, random.randint(0, len(self.seeds) - 1)
        )
        outputs.append(generation_output)
            
        return outputs
    
    def generate_key(self, model_inputs, temperature, max_new_tokens, key_number):
        
        key = self.seeds[key_number]
        
        self.count_request += 1
        
        if self.randomize_every > 0 and self.count_request % self.randomize_every == 0:
            print("DISABLED WATERMARK")
            generation_output = self.normal_generate(model_inputs, max_new_tokens, temperature)
        else:
            generation_output = self.deltareweight_generate(model_inputs, max_new_tokens, temperature, key)
         
        return generation_output
    
    def normal_generate(self, prompts, max_new_tokens, temperature):
        
        generation_output = self.model.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=temperature,
        )

        generation_output.logits = generation_output.scores

        return generation_output

    
    def deltareweight_generate(self, prompts, max_new_tokens, temperature, key):
        
        device = self.model.device
        inputs = prompts.to(device)
        attn = torch.ones_like(inputs).to(device)
        past = None
        logits = []
        past_tokens = []

        for i in range(max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.model(inputs)

            probs = F.softmax(output.logits[:, -1] / temperature, dim=-1)
            
            ctx = tuple(past_tokens[-self.context_size:])
            if ctx in self.cache:
                token = torch.multinomial(probs, 1)
            else:
                token = self.reweight_sample_probs(probs, past_tokens, key)
                
                if self.use_cache:
                    self.cache[ctx] = token
         
            inputs = torch.cat([inputs, token], dim=-1).to(device)

            past_tokens.append(token.item())
            past = output.past_key_values
            attn = torch.cat([attn, torch.ones((attn.shape[0], 1)).to(device)], dim=-1)
            logits.append(output.logits[:, -1].cpu())

            if token.item() == self.tokenizer.eos_token_id:
                break

        generation_output = DummyGenerationOutput(inputs, logits)
        
        return generation_output
    
    
    def reweight_sample_probs(self, probs, past_tokens, key):
        
        reweight_key = hash(tuple(past_tokens[-self.context_size:])) + key
        
        with torch.random.fork_rng():
            torch.random.manual_seed(reweight_key)
            sampled_token = torch.multinomial(probs, 1)
        
        return sampled_token
    
    def reset_cache(self):
        self.cache = {}