from tqdm import tqdm
import torch
import numpy as np
import random
from src.utils import logit
from scipy.stats import mood
from typing import List
import re


def get_model_info(model_name):
    # Define the data based on the table with updated model names

    model_name = model_name.replace("/", "_")

    model_data = {
        "meta-llama_Llama-2-7b-chat-hf": {
            "word_list": ["mangoes", "pineapples", "papayas", "kiwis"],
            "example": "strawberries",
            "format": "Sure!",
        },
        "meta-llama_Llama-2-13b-chat-hf": {
            "word_list": ["peaches", "plums", "cherries", "apricots"],
            "example": "strawberries",
            "format": "Sure!",
        },
        "meta-llama_Llama-2-70b-chat-hf": {
            "word_list": ["mangoes", "pineapples", "papayas", "kiwis"],
            "example": "strawberries",
            "format": "Sure!",
        },
        "mistralai_Mistral-7B-Instruct-v0.1": {
            "word_list": ["strawberries", "blueberries", "raspberries", "blackberries"],
            "example": "apples",
            "format": "",
        },
        "meta-llama_Meta-Llama-3-8B-Instruct": {
            "word_list": ["strawberries", "blueberries", "raspberries", "blackberries"],
            "example": "apples",
            "format": "Sure!",
        },
        "BAAI_Infinity-Instruct-3M-0625-Yi-1.5-9B": {
            "word_list": ["mangoes", "pineapples", "papayas", "kiwis"],
            "example": "strawberries",
            "format": "Sure!",
        },
        "Qwen_Qwen2-7B-Instruct": {
            "word_list": ["peaches", "plums", "cherries", "apricots"],
            "example": "strawberries",
            "format": "",
        },
        "google_gemma-7b-it": {
            "word_list": ["apples", "bananas", "oranges", "pears"],
            "example": "strawberries",
            "format": "Sure!",
        },
    }

    # Check if the model name exists in the data
    if model_name in model_data:
        return model_data[model_name]
    else:
        raise ValueError(f"Model name {model_name} not found in the data")


def find_sublist_idx(response: List[int], candidate_answer: List[int]):
    n = len(candidate_answer)
    for k in range(len(response)):
        if response[k] == candidate_answer[0]:
            if response[k : k + n] == candidate_answer:
                return k
    return -1


def find_common_prefix(l: List[List[int]]):
    k = 0
    for i in range(len(l[0])):
        for subl in l:
            if subl[i] != l[0][i]:
                return k
        k += 1
    return k


def identify_fruit(text, candidates):
    found = []  # This will hold tuples of (candidate, count)

    for i, candidate in enumerate(candidates):
        count = text.count(candidate)
        if count > 0:
            found.append((i, count))

    # We need exactly one candidate to be present
    if len(found) == 1:
        i, count = found[0]
        # And that candidate must appear exactly once
        if count == 1:
            return i
    return None


class CacheLogit:
    def __init__(self, word_list):
        self.cache = {}
        self.word_list = word_list

    def store_prompt_variable(self, prefix, k):
        self.prompt_variable = (prefix, k)

    def store(self, logits, key):
        if key in self.cache:
            return self.cache[key]
        else:
            self.cache[key] = logits
            return logits

    def get(self, key):
        if key in self.cache:
            probs = self.cache[key]

            probs = probs / np.sum(probs)  # Normalize

            # Sample a fruit
            fruit = np.random.choice(len(self.word_list), p=probs)

            return self.word_list[fruit]

        else:
            return None

    def is_cached(self, key):
        return key in self.cache

    def parser(
        self, watermark, generation_output, temperature, output, response, prompt
    ):
        output = output.tolist()

        prefix, k = self.prompt_variable

        word_list = self.word_list

        candidates = [
            watermark.tokenizer.encode(f"{prefix} {k} {word}.")[2:]
            for word in word_list
        ]
        n_prefix = find_common_prefix(candidates)

        tokens_of_interest = [candidate[n_prefix] for candidate in candidates]

        idxs = [find_sublist_idx(output, candidate) for candidate in candidates]

        number_of_repetition = [len(re.findall(word, response)) for word in word_list]

        # If only -1 in idxs; retry
        if (np.array(idxs) == -1).all():
            pass
        elif (
            np.sum(number_of_repetition) >= 2
        ):  # We don't want the model to first pick his choice and then write it
            pass
        else:
            output_idx = max(idxs) + n_prefix

            logits = generation_output.logits
            logits = torch.stack(logits, dim=1) / temperature
            probs = torch.softmax(logits, dim=-1)

            probs = probs[0].cpu().detach().numpy()
            self.store(probs[output_idx, tokens_of_interest], prompt)


def generate_completion(
    prompt: str,
    watermark,
    temperature: float,
    fast: bool,
    logit_cache,
    max_new_tokens: int = 65,
):
    if fast and logit_cache.is_cached(prompt):
        return logit_cache.get(prompt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    messages = [{"role": "user", "content": prompt}]
    encoded_inputs = watermark.tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    model_inputs = encoded_inputs.to(device)
    generation_outputs = watermark.generate(
        model_inputs, temperature=temperature, max_new_tokens=max_new_tokens
    )
    generation_output = generation_outputs[0]
    output = generation_output.sequences[0, model_inputs.shape[-1] :]
    response = watermark.tokenizer.decode(output, skip_special_tokens=True)
    
    print(response)

    if fast:
        logit_cache.parser(
            watermark, generation_output, temperature, output, response, prompt
        )

    return response


def compute_red_green_test(
    watermark,
    temperature: float,
    word_list: List[str],
    context: int,
    example: str,
    format: str,
    n_samples: int,
    fast: bool,
):
    prefixes = [
        "I ate",
        "I chose",
        "I picked",
        "I selected",
        "I took",
        "I went for",
        "I settled on",
        "I got",
        "I gathered",
        "I harvested",
    ]

    k_array = [
        (prefix, int(str(i) * context)) for prefix in prefixes for i in range(1, 10)
    ]

    return _compute_red_green_test(
        watermark=watermark,
        temperature=temperature,
        word_list=word_list,
        k_array=k_array,
        example=example,
        format=format,
        n_samples=n_samples,
        fast=fast,
    )


def _compute_red_green_test(
    watermark,
    temperature: float,
    word_list: List[str],
    k_array: List[str],
    example: str,
    format: str,
    n_samples: int,
    fast: bool,
):
    data = []

    logit_cache = CacheLogit(word_list=word_list)

    for i in tqdm(range(len(k_array))):
        temp = [1] * len(word_list)  # intialized at 1 for stability

        prefix, k = k_array[i]

        logit_cache.store_prompt_variable(prefix, k)

        prompt = f'Complete the sentence "{prefix} {k}" using only and exacty a random word from the list: {word_list}.  Answer in this speific format: {format} {prefix} {k} {example}. (here I chose an other fruit for the sake of the example, you have to choose among {word_list})'
        prompts = [prompt] * n_samples

        out = [
            generate_completion(prompt, watermark, temperature, fast, logit_cache)
            for prompt in prompts
        ]

        for response in out:
            fruit_number = identify_fruit(response, word_list)

            safeguard = 0
            while fruit_number is None:
                print("Parsing error - Rejection sampling")
                response = generate_completion(
                    prompt, watermark, temperature, fast, logit_cache
                )

                fruit_number = identify_fruit(response, word_list)
                safeguard += 1
                if safeguard > 10:
                    fruit_number = random.randint(0, len(word_list) - 1)
                    print(f"Parsing failed {safeguard} times in a row, random choice")
                    break

            temp[fruit_number] += 1

        data.append([x / (n_samples + len(word_list)) for x in temp])
        print(f"Data: {data[-1]}")

    return data


def test_kgw_detection(data: np.array, num_permutations: int):
    data = data.reshape(-1, 9, 4)

    check = np.mean(data, axis=(0, 1)) > 0.8
    if check.any():
        print(
            "Warning: Some fruits have a high probability of being chosen -- Results might be incorrect"
        )
        print(f"Probabilities: {np.mean(data, axis=(0, 1))}")
        print("To adjust this, edit the fruit list in the get_model_info function")

    data = logit(data)

    # Select the token x to look at
    weight = np.sum(data, axis=(0, 1))

    chosen = np.argmax(weight)

    data = data - np.median(data, axis=1, keepdims=True) * 0

    # Calculate the observed statistic
    observed_statistics = statistic(data, chosen)
    statistics = np.zeros(num_permutations)

    # Permutation loop
    for i in range(num_permutations):
        # Permute the entire dataset
        permuted_data = np.random.permutation(data.reshape(-1, 4)).reshape(-1, 9, 4)
        res = statistic(permuted_data, chosen)
        statistics[i] = res

    p_value = np.mean(statistics >= observed_statistics)

    return observed_statistics, statistics, p_value


def statistic(data: np.array, chosen: int):
    data = data.reshape(-1, 9, 4)  # Axis 1 is the last digit axis
    result_array = data[:, :, chosen]

    median = np.median(result_array, axis=1)
    std = np.median(np.std(result_array, axis=0))

    r = 1.96
    red = result_array.T - median < -r * std
    green = result_array.T - median > r * std

    red_score = np.sum(red, axis=1)
    green_score = np.sum(green, axis=1)

    max_red = np.max(red_score)
    min_green = np.min(green_score)
    max_green = np.max(green_score)
    min_red = np.min(red_score)

    max_common = np.max([max_red, max_green])
    min_common = np.max([min_red, min_green])

    return max_common - min_common


def estimate_context_size(data: np.array, significance: float = 0.05):
    n_context, n_perturb, n_t, n_choices = data.shape

    weights = np.sum(data, axis=(0, 1, 2))
    choice = np.argmax(weights)

    data = logit(data)

    # starting from smallest_context:
    chosen_contexts = []
    for t in range(n_t):
        for context in range(1, n_context + 1):
            if context == 1:
                samples1 = data[context - 1, :, t, choice]
            else:
                samples2 = data[context - 1, :, t, choice]
                res = mood(samples1, samples2, alternative="greater")

                if res.pvalue < significance:
                    chosen_contexts.append(context)
                    break

    if len(chosen_contexts) == 0:
        return -1
    else:
        return chosen_contexts


def estimate_context_size(
    watermark, temperature, word_list, contexts, example, format, n_samples, fast
):
    significance = 0.05

    data = compute_context_estimation_data(
        watermark=watermark,
        temperature=temperature,
        word_list=word_list,
        contexts=contexts,
        example=example,
        format=format,
        n_samples=n_samples,
        fast=fast,
    )

    n_context, n_perturb, n_t, n_choices = data.shape

    weights = np.sum(data, axis=(0, 1, 2))
    choice = np.argmax(weights)

    data = logit(data)

    # starting from smallest_context:
    chosen_contexts = []
    for t in range(n_t):
        for context in range(1, n_context + 1):
            if context == 1:
                samples1 = data[context - 1, :, t, choice]
            else:
                samples2 = data[context - 1, :, t, choice]
                res = mood(samples1, samples2, alternative="greater")

                if res.pvalue < significance:
                    chosen_contexts.append(context)
                    break

    if len(chosen_contexts) == 0:
        return -1
    else:
        return chosen_contexts


def compute_context_estimation_data(
    watermark,
    temperature: float,
    word_list: List[str],
    contexts: List[int],
    example: str,
    format: str,
    n_samples: int,
    fast: bool,
):
    prefix = "I ate"

    k_array = [
        (prefix, int(str(perturbation) + str(t2) * context))
        for context in contexts
        for perturbation in range(1, 10)
        for t2 in [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]

    data = _compute_red_green_test(
        watermark=watermark,
        temperature=temperature,
        word_list=word_list,
        k_array=k_array,
        example=example,
        format=format,
        n_samples=n_samples,
        fast=fast,
    )

    data = np.array(data)
    data = data.reshape(len(contexts), 9, 9, len(word_list))

    return data
