from src.kgw.extended_watermark_processor import WatermarkLogitsProcessor
from transformers import LogitsProcessorList
from typing import List
import random

def get_logit_processor(gamma, delta, seeding_scheme, device, tokenizer):
    vocab = list(tokenizer.get_vocab().values())
    return WatermarkLogitsProcessor(
        vocab=vocab,
        gamma=gamma,
        delta=delta,
        seeding_scheme=seeding_scheme,
        device=device,
        tokenizer=tokenizer,
    )


class KGW_watermark:
    def __init__(
        self,
        model,
        tokenizer,
        delta,
        gamma=0.25,
        context_size=3,
        seeding_scheme="lefthash",
        seeds: List = [0],
        disable_watermark_every: int = 0
    ):
        self.gamma = gamma
        self.delta = delta
        self.context_size = context_size
        self.seeding_scheme = seeding_scheme
        self.model = model
        self.tokenizer = tokenizer
        self.seeds = seeds
        self.randomize_every = disable_watermark_every
        self.count_request = 0

    def generate(self, model_inputs, temperature, max_new_tokens) -> List:
        outputs = []

        generation_output = self.generate_key(
            model_inputs, temperature, max_new_tokens, random.randint(0, len(self.seeds) - 1)
        )

        outputs.append(generation_output)
        return outputs
    
    def generate_key(self, model_inputs, temperature, max_new_tokens, key_number, disable_watermark: bool = False, force_watermark: bool = False):

        assert not (disable_watermark and force_watermark), "Cannot disable and force watermark at the same time"

        key = self.seeds[key_number]

        if self.delta == 0:
            logit_processor = None
        else:
            logit_processor = get_logit_processor(
                self.gamma,
                self.delta,
                self.seeding_scheme,
                self.model.device,
                self.tokenizer,
            )
            
        # Disable logit processor every nth request
        self.count_request += 1
        if self.randomize_every > 0 and self.count_request % self.randomize_every == 0:
            logit_processor = None
            print("DISABLED WATERMARK")
            
        # Force arguments
        if disable_watermark:
            logit_processor = None
            print("FORCE DISABLED WATERMARK")
            
        if force_watermark:
            logit_processor = get_logit_processor(
                self.gamma,
                self.delta,
                self.seeding_scheme,
                self.model.device,
                self.tokenizer,
            )
            print("FORCE ENABLED WATERMARK")

        if logit_processor is not None:
            logit_processor.hash_key = key
            logit_processor.context_width = self.context_size
            logit_processors = [logit_processor]
        else:
            logit_processors = []

        generation_output = self.model.generate(
            model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=LogitsProcessorList(logit_processors),
            temperature=temperature,
        )

        generation_output.logits = generation_output.scores

        return generation_output
