"""
ShortGPT utility functions - CAP6614 Team 7

Shared helper functions for the ShortGPT pipeline. Import these into
notebooks or scripts instead of copy-pasting code around.

Usage:
    from shortgpt_utils import (
        load_model_4bit,
        get_wikitext2_calibration,
        block_influence,
        compute_bi_scores,
        remove_layers,
        evaluate_perplexity,
    )
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Model Loading ---


def load_model_4bit(model_id="meta-llama/Llama-2-7b-hf"):
    """Load a HuggingFace causal LM in 4-bit NF4 quantization.
    Returns (model, tokenizer) tuple.
    """
    # NF4 quantization config - double quant saves a bit more memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Some models don't have a pad token set, so use EOS as a fallback
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",          # let accelerate figure out device placement
        torch_dtype=torch.bfloat16,
    )
    model.eval()  # we're only doing inference, no training
    return model, tokenizer


# --- Calibration Data ---


def get_wikitext2_calibration(nsamples, seqlen, tokenizer, seed=42):
    """Load WikiText-2 and return random calibration samples + test encoding.

    We tokenize the whole training set into one big sequence, then sample
    random windows of length seqlen from it. This is the standard approach
    used by SparseGPT, WANDA, and ShortGPT.

    Returns (calibration_samples, test_encoding) where calibration_samples
    is a list of (1, seqlen) tensors.
    """
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concat all text into one long string, then tokenize
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Sample random windows from the training tokens
    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        samples.append(trainenc.input_ids[:, i : i + seqlen])

    return samples, testenc


# --- Block Influence ---


def block_influence(x_in, x_out):
    """Compute per-token Block Influence between a layer's input and output.
    BI = 1 - cosine_similarity(input, output).
    Returns a (B*S,) tensor of BI scores in [0, 1].
    """
    B, S, D = x_in.shape

    # Flatten batch and sequence dims so we can compute cosine sim per token
    flat_in = x_in.reshape(-1, D).float()
    flat_out = x_out.reshape(-1, D).float()

    sim = F.cosine_similarity(flat_in, flat_out, dim=-1)
    # If a token has zero-norm hidden state (rare edge case), default to 0.5
    sim = sim.nan_to_num(nan=0.5)
    return 1.0 - sim


def compute_bi_scores(model, calibration_data, num_layers=None):
    """Compute average Block Influence score for every layer.

    Runs all calibration samples through the model with output_hidden_states=True.
    hidden_states[i] is the input to layer i, hidden_states[i+1] is the output.

    Returns a list of float BI scores, one per layer.
    """
    if num_layers is None:
        num_layers = len(model.model.layers)

    # Accumulate BI across all tokens from all samples
    bi_scores = [0.0] * num_layers
    total_tokens = 0

    with torch.no_grad():
        for inp in tqdm(calibration_data, desc="Computing BI scores"):
            outputs = model(
                input_ids=inp.to(DEVICE),
                output_hidden_states=True,  # gives us hidden states at every layer boundary
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors
            total_tokens += inp.shape[1]

            for i in range(num_layers):
                bi = block_influence(hidden_states[i], hidden_states[i + 1])
                bi_scores[i] += bi.sum().item()

    # Average over all tokens
    bi_scores = [s / total_tokens for s in bi_scores]
    return bi_scores


# --- Layer Removal ---


def remove_layers(model, bi_scores, n_prune):
    """Remove the n_prune layers with the lowest BI scores.

    Three things we need to handle carefully:
    1. Delete in reverse index order so earlier indices don't shift
    2. Re-index self_attn.layer_idx on remaining layers (KV cache breaks otherwise)
    3. Update model.config.num_hidden_layers so save/load works right

    Returns a sorted list of the removed layer indices (original numbering).
    """
    # Pick the layers with the smallest BI (most redundant)
    layers_to_remove = np.argsort(bi_scores)[:n_prune].tolist()

    # Delete from highest index to lowest so we don't mess up the ordering
    for idx in sorted(layers_to_remove, reverse=True):
        del model.model.layers[idx]

    # Fix the layer_idx attribute on each remaining layer's attention module.
    # If we skip this, the KV cache will try to access the wrong positions.
    for new_idx, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = new_idx

    # Keep the config in sync with the actual layer count
    model.config.num_hidden_layers = len(model.model.layers)

    return sorted(layers_to_remove)


# --- Perplexity Evaluation ---


def evaluate_perplexity(model, test_encoding, seqlen=512):
    """Calculate perplexity on a test set using non-overlapping windows.
    Lower PPL = better. Returns a float.
    """
    test_ids = test_encoding.input_ids
    n_tokens = test_ids.shape[1]
    nlls = []  # negative log likelihoods per window

    with torch.no_grad():
        for i in tqdm(range(0, n_tokens - seqlen, seqlen), desc="Evaluating PPL"):
            batch = test_ids[:, i : i + seqlen].to(DEVICE)
            outputs = model(input_ids=batch, use_cache=False, return_dict=True)

            # Standard next-token prediction: shift logits left by 1
            # Cast to float32 before CE to avoid precision blow-up in fp16/bf16
            logits = outputs.logits[:, :-1, :].float().contiguous()
            labels = batch[:, 1:].contiguous()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="mean",
            )
            nlls.append(loss.item())

    # PPL = exp(average NLL)
    return float(np.exp(np.mean(nlls)))
