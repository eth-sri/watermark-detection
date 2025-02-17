import yaml
from typing import Dict
import os

def load_recursive(path: str) -> Dict:
    """Simple recursive loader for yaml files"""
    
    data = yaml.safe_load(open(path, "r"))
    
    for key, value in data.items():
        if isinstance(value, str) and os.path.isfile(value) and value.endswith(".yaml"):
            data[key] = load_recursive(value)
            
    return data
            