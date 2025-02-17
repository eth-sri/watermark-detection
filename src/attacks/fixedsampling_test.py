import os
from tqdm import tqdm
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import random
from collections import Counter


def generate_fixedsampling_test_data(watermark, temperature, n_samples, max_new_tokens):
    """Generate data for the Fixed-Sampling detection test."""
    
    prompt = "This is the story of"
    prompt = watermark.tokenizer.encode(prompt, return_tensors='pt', max_length=2048)
    prompt = prompt.to(watermark.model.device)
    
    responses = []
    response_ids = {}
    
    max_id = 0
                
    for _ in tqdm(range(n_samples)):
        
        generation_output = watermark.generate_key(
                        prompt, 
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        key_number=0)
            
        response = watermark.tokenizer.decode(generation_output.sequences[0])

        if response not in response_ids:
            response_ids[response] = max_id
            max_id += 1
        id = response_ids[response]        

        responses.append(id)
        
    return responses
            
def _rarefaction_curve(data, num_samples=1000, trials=500):

    max_samples = min(num_samples, len(data))
    unique_counts = np.zeros(max_samples)
    
    # Perform multiple trials to average the curve
    for _ in range(trials):
        np.random.shuffle(data)
        seen_sentences = set()
        cumulative_uniques = []

        for i in range(1, max_samples + 1):
            seen_sentences.add(data[i - 1])
            cumulative_uniques.append(len(seen_sentences))
        
        unique_counts += np.array(cumulative_uniques)

    unique_counts /= trials
    return unique_counts
            
            
def test_stanford(data):
    
    rarefaction = _rarefaction_curve(data, num_samples=len(data))
    return stats.mannwhitneyu(rarefaction, np.arange(len(rarefaction))).pvalue

def rarefaction(x, n_key):
    return n_key * (1 - (1 - 1/n_key)**x)

def fit_model(data, x, n_bootstraps=100):
    bootstrap_params = []
    for _ in range(n_bootstraps):
        # Resample the data
        resampled_data = _rarefaction_curve(data)
        try:
            # Fit model on resampled data
            popt, _ = curve_fit(rarefaction, x, resampled_data)
            bootstrap_params.append(popt)
        except RuntimeError:
            # Handle cases where the model fitting fails
            continue

    bootstrap_params = np.array(bootstrap_params)
    # Estimate the parameter confidence intervals
    ci_lower = np.percentile(bootstrap_params, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_params, 97.5, axis=0)

    return ci_lower, ci_upper, np.mean(bootstrap_params, axis=0)


def estimate_key_size(data):
    
    _,_,key_size = fit_model(data, np.arange(len(data)))
    
    return key_size