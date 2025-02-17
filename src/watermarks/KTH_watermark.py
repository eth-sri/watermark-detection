import torch
from typing import List
import random


def gumbel_key_func(generator, n, vocab_size, eff_vocab_size=None):
    if eff_vocab_size is None:
        eff_vocab_size = vocab_size

    pi = torch.arange(eff_vocab_size)
    xi = torch.rand((n, eff_vocab_size), generator=generator)

    return xi, pi


def gumbel_sampling(probs, pi, xi):
    return torch.argmax(xi ** (1 / torch.gather(probs, 1, pi)), axis=1).unsqueeze(-1)


class KTH_watermark:
    def __init__(
        self,
        model,
        tokenizer,
        key_length: int,
        seeds: List = [0],
        disable_watermark_every: int = 0,
        percent_disable: float = 0,
    ):
        self.key_length = key_length
        self.model = model
        self.tokenizer = tokenizer
        self.seeds = seeds
        self.randomize_every = disable_watermark_every
        self.count_request = 0

        self.percent_disable = percent_disable

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
            generation_output = self.normal_generate(
                model_inputs, max_new_tokens, temperature
            )
        else:
            generation_output = self.kth_generate(
                self.model,
                model_inputs,
                self.key_length,
                max_new_tokens,
                [key],
                temperature,
            )

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

    def kth_generate(
        self,
        model,
        prompts,
        n_key,
        max_new_tokens,
        seeds,
        temperature,
        random_offset=True,
    ):
        key_func = gumbel_key_func
        sampler = gumbel_sampling
        vocab_size = model.get_output_embeddings().weight.shape[0]

        batch_size = len(prompts)

        generator = torch.Generator()
        xis, pis = [], []
        for seed in seeds:
            generator.manual_seed(int(seed))
            xi, pi = key_func(generator, n_key, vocab_size)
            xis.append(xi.unsqueeze(0))
            pis.append(pi.unsqueeze(0))
        xis = torch.vstack(xis)
        pis = torch.vstack(pis)

        # deliberately not controlling this randomness with the generator
        if random_offset:
            offset = torch.randint(n_key, size=(batch_size,))
        else:
            offset = torch.zeros(size=(batch_size,), dtype=torch.int64)
        inputs = prompts.to(model.device)
        attn = torch.ones_like(inputs)
        past = None

        logits = []

        for i in range(max_new_tokens):
            with torch.no_grad():
                if past:
                    output = model(
                        inputs[:, -1:], past_key_values=past, attention_mask=attn
                    )
                else:
                    output = model(inputs)

            probs = torch.nn.functional.softmax(
                output.logits[:, -1] / temperature, dim=-1
            ).cpu()

            # Random disabling of watermark
            disable = random.random() < self.percent_disable

            if disable:
                print("DISABLED WATERMARK ON TOKEN", i)
                tokens = torch.multinomial(probs, 1).to(model.device)
            else:
                tokens = sampler(
                    probs,
                    pis,
                    xis[torch.arange(batch_size), (offset.squeeze() + i) % n_key],
                ).to(model.device)
            inputs = torch.cat([inputs, tokens], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            logits.append(output.logits[:, -1].cpu())

            if tokens.item() == self.tokenizer.eos_token_id:
                break

        generation_output = DummyGenerationOutput(inputs, logits)

        return generation_output


class DummyGenerationOutput:
    def __init__(self, sequences, logits):
        self.sequences = sequences
        self.logits = logits
