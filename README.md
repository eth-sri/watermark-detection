# Black-Box Detection of Language Model Watermarks


## Requirements

To install requirements:

```setup
conda env create --file=env.yaml
```

Depending on your GPU setup this might be subject to change.

## Running the test

### Open source models

For configuring tests, please refer to *src/config.py* and the files in *src/configs*. Note that example configurations are available in the *configs* folder.

For the Red-Green test and the Cache-Augmented test, the *fast* option should be disabled for most watermarks (only tested with KGW).

For the Cache-Augmented test, the cache is enforced on all watermarks by default. To enable or disable the cache, use the *use_cache* argument.

To run the different tests with the provided configuration, please do

```
python src/main.py --config configs/main/red_green_llama.yaml
python src/main.py --config configs/main/fixed_llama.yaml
python src/main.py --config configs/main/cache_llama.yaml
```

### Closed models

The notebook *claused_models.ipynb* contains all the code needed to perform each test in a self-contained manner. You need to provide API keys for these models to work. 
Only implentation for the OpenAI, Gemini and Claude API is provided. To ensure compatibility with other model providers, please refer to *src/closed_models.py*.


### Additional information

SynthID-text watermark is only implemented for Gemma models.