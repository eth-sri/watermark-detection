from pydantic import BaseModel
from strenum import StrEnum
from src.attacks.redgreen_test import (
    compute_red_green_test,
    test_kgw_detection,
    get_model_info,
    estimate_context_size
)
from src.attacks.fixedsampling_test import (
    generate_fixedsampling_test_data,
    test_stanford,
    estimate_key_size
)
from src.attacks.cache_test import (
    generate_data_cache_test
)
from src.delta_estimation import estimate_delta_grad
import numpy as np
from scipy.stats import fisher_exact

class TestType(StrEnum):
    red_green = "red_green"
    fixed_key = "fixed_key"
    cache = "cache"
    
    def get_config(self, config):
        if self == TestType.red_green:
            return RedGreenTestConfiguration.parse_obj(config)
        elif self == TestType.fixed_key:
            return FixedSamplingTestConfiguration.parse_obj(config)
        elif self == TestType.cache:
            return CacheTestConfiguration.parse_obj(config)
        else:
            raise NotImplementedError


class TestConfiguration(BaseModel):
    pass

    def launch(self, watermark, configuration):
        raise NotImplementedError


class RedGreenTestConfiguration(TestConfiguration):
    context: int  
    n_samples: int
    
    estimate_delta: bool = False  # If True, will estimate the delta value
    estimate_context_size: bool = False  # If True, will estimate the context size
    
    fast: bool = False  # If True, will only generate one sample and simulate the red-green test

    def launch(self, watermark, configuration):
        word_list, example, format = get_model_info(configuration.model_name).values()

        data = compute_red_green_test(
            watermark=watermark, 
            temperature=configuration.temperature, 
            word_list=word_list, 
            context=self.context, 
            example=example, 
            format=format,
            n_samples=self.n_samples,
            fast=self.fast
        )
        
        data = np.array(data)
        
        _,_, p_value = test_kgw_detection(data, num_permutations=1000)
        
        print(f"P-value: {p_value}")
        
        if self.estimate_delta:
            delta = estimate_delta_grad(data)
            print("Estimated delta: ", delta)
            
        if self.estimate_context_size:
            chosen_context = estimate_context_size(
                watermark=watermark,
                temperature=configuration.temperature,
                word_list=word_list,
                contexts=[1,2,3,4,5],
                example=example,
                format=format,
                n_samples=self.n_samples,
                fast=self.fast
            )
            print("Estimated context size: ", np.median(chosen_context))

        return p_value

class FixedSamplingTestConfiguration(TestConfiguration):
    n_samples: int
    max_new_tokens: int
    
    estimate_key_size: bool = False 
    
    
    def launch(self, watermark, configuration):
        
        data = generate_fixedsampling_test_data(
            watermark=watermark,
            temperature=configuration.temperature,
            n_samples=self.n_samples,
            max_new_tokens=self.max_new_tokens,
        )
        
        data = np.array(data)
        
        pvalue = test_stanford(data)
        
        print("Test result", pvalue)
        
        if self.estimate_key_size:
            key_size = estimate_key_size(data)
            print("Estimated key size: ", key_size)
        
        return pvalue
        
class CacheTestConfiguration(TestConfiguration):
    n_samples: int
    fast: bool
    use_cache: bool
    
    def launch(self, watermark, configuration):
        
        phase1, phase2 = generate_data_cache_test(watermark, configuration.temperature, configuration.model_name, self.n_samples, self.use_cache, self.fast)
        
        res = fisher_exact([phase2, phase1])
        p_value = res.pvalue
        
        print(f"Test results: {p_value}")
        
        return p_value