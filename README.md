# CAP6614 Group Project: Replicating ShortGPT

**Team 7 - Efficient AI, Spring 2026, UCF**

Reproducing the results from [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853) (Men et al., 2024).

ShortGPT measures how much each transformer layer changes the hidden state passing through it (Block Influence metric), ranks layers by importance, and removes the least important ones without any retraining.

## Team

- **Adrian Teodorescu** -- Lead: coordination, docs, qualitative testing
- **Tristan Sherzer** -- Implementation: BI metric, layer removal, model loading
- **Kensley Cadet** -- Data & evaluation: dataset prep, baseline PPL, reproduction
- **Morgan Skinner** -- Benchmarking & extension: inference speed, Phi-2/TinyLlama

## Files

- **`shortgpt_pipeline.ipynb`** -- Main notebook. Runs the full pipeline from model loading through pruning to evaluation. Start here.
- **`shortgpt_utils.py`** -- Importable utility functions. Use this to call functions from your own notebook or script without copy-pasting.
- **`requirements.txt`** -- Python dependencies.

## Quick Start (Google Colab)

1. Upload `shortgpt_pipeline.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Make sure you have a [HuggingFace account](https://huggingface.co/join) and have accepted the [Llama-2 license](https://huggingface.co/meta-llama/Llama-2-7b-hf)
3. Set your runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
4. Run cells top to bottom - the HF login cell will prompt for your token

## Quick Start (Local)

```bash
pip install -r requirements.txt
jupyter notebook shortgpt_pipeline.ipynb
```

Requires a CUDA GPU. Works with any GPU that has at least 6GB VRAM in 4-bit mode.

## What the Notebook Does

The notebook runs the complete ShortGPT pipeline in order:

1. **Load model** - Llama-2-7B in 4-bit NF4 quantization (~5GB VRAM)
2. **Load calibration data** - WikiText-2 (128 samples x 512 tokens)
3. **Compute Block Influence** - measures how much each layer changes the hidden state. BI = 1 - cosine_similarity(layer_input, layer_output). Low BI = redundant layer.
4. **Visualize** - bar chart of BI scores per layer (reproduces Figure 3 from the paper)
5. **Baseline perplexity** - evaluate the unpruned model on WikiText-2 test set
6. **Prune** - remove the 8 lowest-BI layers (~25% of 32 total)
7. **Evaluate pruned model** - perplexity + text generation quality check
8. **Summary** - prints a results table with before/after comparison

## Using shortgpt_utils.py

If you want to use the functions from your own code instead of running the full notebook:

```python
from shortgpt_utils import (
    load_model_4bit,
    get_wikitext2_calibration,
    compute_bi_scores,
    remove_layers,
    evaluate_perplexity,
)

# Load model
model, tokenizer = load_model_4bit("meta-llama/Llama-2-7b-hf")

# Load calibration data
cal_data, test_enc = get_wikitext2_calibration(128, 512, tokenizer)

# Compute BI scores
bi_scores = compute_bi_scores(model, cal_data)

# Prune 8 layers
removed = remove_layers(model, bi_scores, n_prune=8)

# Evaluate
ppl = evaluate_perplexity(model, test_enc)
```

### Adapting for a different model (Phi-2, TinyLlama, etc.)

Just change the model ID:

```python
model, tokenizer = load_model_4bit("microsoft/phi-2")
# or
model, tokenizer = load_model_4bit("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

The rest of the pipeline (BI computation, pruning, evaluation) works for any model that uses the same `model.model.layers` structure as Llama - this includes Phi-2 and TinyLlama. Layer count is detected automatically.

## Paper Reference

```
Men, X., Xu, M., Zhang, Q., Wang, B., Lin, H., Lu, Y., Han, X., & Chen, W. (2024).
ShortGPT: Layers in Large Language Models are More Redundant Than You Expect.
arXiv preprint arXiv:2403.03853.
```
