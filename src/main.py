import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.configs.utils import load_recursive
from src.config import MainConfiguration

def parse_args():
    parser = argparse.ArgumentParser(description="Black-box detection of LLM watermarks")

    parser.add_argument("--config", type=str, help="Path to the configuration file")

    return parser.parse_args()


def get_watermarked_model(configuration: MainConfiguration):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(configuration.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(configuration.model_name)
    disable_watermark_every = configuration.disable_watermark_every
    
    watermark = configuration.watermark_config.get_watermarked_model(
        model=model,
        tokenizer=tokenizer,
        disable_watermark_every=disable_watermark_every,
    )
    
    return watermark

if __name__ == "__main__":
    args = parse_args()
    
    config = load_recursive(args.config)
    configuration = MainConfiguration.model_validate(config)
    
    watermark = get_watermarked_model(configuration)
    
    configuration.test_config.launch(watermark, configuration)