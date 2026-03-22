import time
import torch
import numpy as np
from typing import Dict, List


def benchmark_inference_speed(
        model,
        tokenizer,
        prompt: str = "Provide step-by-step instructions for making chocolate chip cookies.",
        max_new_tokens: int = 100,
        num_runs: int = 3
) -> Dict[str, float]:
    """
    Args:
        model: Loaded HF model
        tokenizer: tokenizer
        prompt: Text prompt for generation
        max_new_tokens: How many tokens to generate
        num_runs: Number of benchmark runs

    """
    print(f"\n Benchmarking: {max_new_tokens} token generation ")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    speeds = []
    latencies = []

    for run in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                use_cache=True
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        total_tokens = outputs.shape[1] - input_len
        elapsed = end_time - start_time
        tokens_per_sec = total_tokens / elapsed
        ms_per_token = (elapsed / total_tokens) * 1000
        speeds.append(tokens_per_sec)
        latencies.append(ms_per_token)

        print(f"  Run {run + 1}: {tokens_per_sec:.2f} tokens/sec ({ms_per_token:.2f} ms/token)")

    return {
        'tokens_per_sec_mean': np.mean(speeds) if speeds else 0,
        'tokens_per_sec_std': np.std(speeds) if speeds else 0,
        'ms_per_token_mean': np.mean(latencies) if latencies else 0,
        'ms_per_token_std': np.std(latencies) if latencies else 0,
        'total_tokens_generated': max_new_tokens
    }


def measure_vram_usage(model, tokenizer, prompt: str, max_new_tokens: int = 100):

    if not torch.cuda.is_available():
        print("CUDA not available, skipping VRAM measurement")
        return {
            'vram_before_gb': 0,
            'vram_peak_gb': 0,
            'vram_after_gb': 0,
            'cuda_available': False
        }

    print(f"\nVRAM Usage ({max_new_tokens} tokens)")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1024 ** 3
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1024 ** 3
    mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"  VRAM before: {mem_before:.2f} GB")
    print(f"  VRAM peak:   {mem_peak:.2f} GB")
    print(f"  VRAM after:  {mem_after:.2f} GB")

    return {
        'vram before gb': mem_before,
        'vram peak gb': mem_peak,
        'vram after gb': mem_after
    }


def quick_benchmark(model, tokenizer, model_name="Model"):
    print(f"Quick Benchmark: {model_name}")
    num_layers = len(model.model.layers)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Layers: {num_layers}")
    print(f"Parameters: {num_params:.2f}B")

    test_prompt = "List five tips for improving productivity while working from home."

    results = {}
    for token_len in [50, 100]:
        speed = benchmark_inference_speed(
            model, tokenizer,
            prompt=test_prompt,
            max_new_tokens=token_len,
            num_runs=2  #Can increase
        )
        results[f'speed_{token_len}'] = speed

        vram = measure_vram_usage(
            model, tokenizer,
            prompt=test_prompt,
            max_new_tokens=token_len
        )
        results[f'vram_{token_len}'] = vram


    print("Summary")
    print(f"Model: {num_layers} layers, {num_params:.2f}B params")
    if 'speed_50' in results and 'tokens_per_sec_mean' in results['speed_50']:
        print(f"Speed (50 tokens): {results['speed_50']['tokens_per_sec_mean']:.1f} tokens/sec")
    else:
        print(f"Speed (50 tokens): N/A")

    if 'speed_100' in results and 'tokens_per_sec_mean' in results['speed_100']:
        print(f"Speed (100 tokens): {results['speed_100']['tokens_per_sec_mean']:.1f} tokens/sec")
    else:
        print(f"Speed (100 tokens): N/A")

    if 'vram_100' in results and results['vram_100'].get('cuda_available', False):
        print(f"Peak VRAM (100 tokens): {results['vram_100']['vram_peak_gb']:.2f} GB")
    else:
        print(f"Peak VRAM (100 tokens): N/A (CPU mode)")
    return results
