from typing import List
import torch
import torch.nn.functional as F
from src.watermarks.KTH_watermark import DummyGenerationOutput
import random

class DipMark_watermark():
    
    def __init__(self, model, tokenizer, context_size: int, seeds: List = [0], alpha = 0.5, use_cache: bool = True, disable_watermark_every: int = 0):
        
        self.model = model
        self.tokenizer = tokenizer
        self.seeds = seeds
        self.context_size = context_size
        self.alpha = alpha
        self.cache = {}
        self.use_cache = use_cache
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
            generation_output = self.dipmark_generate(model_inputs, max_new_tokens, temperature, key)
         
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

    
    def dipmark_generate(self, prompts, max_new_tokens, temperature, key):
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
                permuted_probs = probs
            else:
                permuted_probs = self.permute_probs(probs, past_tokens, key)
                
                if self.use_cache:
                    self.cache[ctx] = permuted_probs
         
            token = torch.multinomial(permuted_probs, 1)
            inputs = torch.cat([inputs, token], dim=-1).to(device)

            past_tokens.append(token.item())
            past = output.past_key_values
            attn = torch.cat([attn, torch.ones((attn.shape[0], 1)).to(device)], dim=-1)
            logits.append(output.logits[:, -1].cpu())

            if token.item() == self.tokenizer.eos_token_id:
                break

        generation_output = DummyGenerationOutput(inputs, logits)
        
        return generation_output
        
    def permute_probs(self, probs, past_tokens, key):
        permutation_key = hash(tuple(past_tokens[-self.context_size:]))+ key

        # Generate permutation based on the key
        with torch.random.fork_rng():
            torch.random.manual_seed(permutation_key)
            idx = torch.randperm(probs.size(-1))

        permuted_probs = permute_and_reweight_probs(probs, idx, self.alpha, self.model.device)
        
        return permuted_probs
    
    def reset_cache(self):
        self.cache = {}
    
def permute_and_reweight_probs(probs, permutation, alpha, device):
    permuted_probs = probs[:, permutation]

    cdf = permuted_probs.cumsum(dim=-1)
    
    F = torch.maximum(cdf - alpha, torch.zeros_like(cdf)) + torch.maximum(cdf - (1 - alpha), torch.zeros_like(cdf))
    F = F[0]
    F = torch.concat([torch.tensor([0], device=device), F], dim=-1)
    reweighted_probs = F[1:] - F[:-1]
    reweighted_probs = torch.maximum(reweighted_probs, torch.zeros_like(reweighted_probs)) # Ensure non-negativity due to numerical errors
    
    # Shuffle back to original order
    reweighted_probs = reweighted_probs[torch.argsort(permutation)]
    
    return reweighted_probs.view(1, -1)