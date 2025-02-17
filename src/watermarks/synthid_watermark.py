from synthid_text import synthid_mixin
import transformers
from typing import List
import torch
import enum

class ModelName(enum.Enum):
    GPT2 = "gpt2"
    GEMMA_2B = "google/gemma-2b-it"
    GEMMA_7B = "google/gemma-7b-it"
    

def load_model(
    model_name: ModelName,
    enable_watermarking: bool = False,
    disable_cache: bool = True,
) -> transformers.PreTrainedModel:
    match model_name:
        case ModelName.GPT2:
            model_cls = (
                synthid_mixin.SynthIDGPT2LMHeadModel
                if enable_watermarking
                else transformers.GPT2LMHeadModel
            )
            model = model_cls.from_pretrained(model_name.value, device_map="auto")
        case ModelName.GEMMA_2B | ModelName.GEMMA_7B:
            model_cls = (
                synthid_mixin.SynthIDGemmaForCausalLM
                if enable_watermarking
                else transformers.GemmaForCausalLM
            )
            model = model_cls.from_pretrained(
                model_name.value,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=False
            )
    return model

class SynthIDWatermark:
    def __init__(
        self,
        model_name,
        use_watermark: bool = True,
    ):
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        
        model_name = ModelName(model_name)
        
        self.model = load_model(model_name, enable_watermarking=use_watermark)
    

    def generate(self, model_inputs, temperature, max_new_tokens) -> List:
        outputs = []


        generation_output = self.generate_key(
            model_inputs, temperature, max_new_tokens, None
        )

        outputs.append(generation_output)
        return outputs
    
    def generate_key(self, model_inputs, temperature, max_new_tokens, key_number):

        generation_output = self.model.generate(
            model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            do_sample=True,
            return_dict_in_generate=True,
            output_logits=True,
            temperature=temperature,
        )
        return generation_output
