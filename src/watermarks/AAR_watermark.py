# Code forked from https://github.com/THU-BPM/MarkLLM/blob/main/watermark/exp_gumbel/exp_gumbel.py
import torch.nn.functional as F
from src.watermarks.KTH_watermark import DummyGenerationOutput
from typing import List
import torch
import numpy as np
import scipy
from math import log

class AARWatermark:
    def __init__(
        self,
        model,
        tokenizer,
        context_size: int = 3,
        lambd: float = 3,
        seed: int = 0
    ):

        self.tokenizer = tokenizer
        self.model = model

        self.context_size = context_size
        self.lambd = lambd
        
        self.eps = 1e-3
        
        # Utils
        vocab_size = tokenizer.vocab_size
        self.generator = torch.Generator(device=model.device).manual_seed(seed)
        self.uniform = torch.clamp(torch.rand((vocab_size * self.context_size, vocab_size), 
                                         generator=self.generator, dtype=model.dtype, device=model.device), min=self.eps)
        self.gumbel = (-torch.log(torch.clamp(-torch.log(self.uniform), min=self.eps))).to(self.model.device)
        print("Initialized AAR")

    def generate(self, model_inputs, temperature, max_new_tokens) -> List:
        outputs = []
        
        generation_output = self.generate_key(
            model_inputs, temperature, max_new_tokens,
        )

        outputs.append(generation_output)
        return outputs
    
    def generate_key(self, model_inputs, temperature, max_new_tokens, key_number= None):
        
        generation_output = self.aarson_generate(model_inputs, temperature, max_new_tokens)
        
        return generation_output

    def aarson_generate(self, model_inputs, temperature, max_new_tokens):
        
        # Initialize
        inputs = model_inputs
        attn = torch.ones_like(model_inputs)
        past = None
        logits = []

        entropies = []

        # Generate tokens
        for i in range(max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                    output_gumbel, use_argmax = self._logit_processor(input_ids=inputs, scores=output.logits, temperature=temperature, entropies=entropies)
                else:
                    output = self.model(inputs)
                    output_gumbel, use_argmax = self._logit_processor(input_ids=inputs, scores=output.logits, temperature=temperature, entropies=entropies)

            # Sample token
            probs = F.softmax(output.logits[:, -1] / temperature, dim=-1)
            if use_argmax:
                token = self.watermark_logits_argmax(inputs, output_gumbel)
            else:
                token = torch.multinomial(probs, 1)
            
            # Update past
            past = output.past_key_values

            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
            
            selected_prob = probs[0, token].item()
            entropies.append(-np.log(selected_prob))
            
            logits.append(output.logits[:, -1].cpu())   
            
            if token.item() == self.tokenizer.eos_token_id:
                break
            
        generation_output = DummyGenerationOutput(inputs, logits)


        return generation_output
    
    def _logit_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, temperature: float, entropies: List[float]) -> torch.FloatTensor:

        if input_ids.shape[-1] < self.context_size:
            return scores, False
        
        if self.lambd != 0 and len(entropies) < self.context_size:
            return scores, False
        
        current_context_entropy = np.sum(entropies[-self.context_size:])
        if current_context_entropy < self.lambd:
            return scores, False
        
        prev_token = torch.sum(input_ids[:, -self.context_size:], dim=-1)  # (batch_size,)
        gumbel = self.gumbel[prev_token]  # (batch_size, vocab_size)
        return scores[..., :gumbel.shape[-1]] / temperature + gumbel, True
    
    def watermark_logits_argmax(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.LongTensor:
        """
        Applies watermarking to the last token's logits and returns the argmax for that token.
        Returns tensor of shape (batch,), where each element is the index of the selected token.
        """
        
        # Get the logits for the last token
        last_logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Get the argmax of the logits
        last_token = torch.argmax(last_logits, dim=-1).unsqueeze(-1)  # (batch,)
        return last_token
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]

        seq_len = len(encoded_text)
        score = 0
        for i in range(self.context_size, seq_len):
            prev_tokens_sum = torch.sum(encoded_text[i - self.context_size:i], dim=-1)
            token = encoded_text[i]
            u = self.uniform[prev_tokens_sum, token]
            score += log(1 / (1 - u))
        
        p_value = scipy.stats.gamma.sf(score, seq_len - self.context_size, loc=0, scale=1)
        
        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = bool(p_value < 0.01)

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": p_value}
        else:
            return (is_watermarked, p_value)