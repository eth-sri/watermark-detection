from typing import List

class No_watermark():
    
    def __init__(self, model, tokenizer):
        
        self.model = model
        self.tokenizer = tokenizer
        
    def generate(self, model_inputs, temperature, max_new_tokens) -> List:
        
        outputs = []
        
        generation_output = self.generate_key(model_inputs, temperature, max_new_tokens)
        outputs.append(generation_output)
            
        return outputs
    
    def generate_key(self, model_inputs, temperature, max_new_tokens, **kwargs):
        
        generation_output = self.normal_generate(model_inputs, max_new_tokens, temperature)
         
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