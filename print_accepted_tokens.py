#!/usr/bin/env python3
import json
import math
import statistics
from pathlib import Path

# Mapping from shorthand to full model name (for file path construction)
MODELS = {
    "68m": "llama-68m",
    "1b": "Llama-3.2-1B-Instruct",
    "7b": "Llama-2-7b-hf",
    "8b": "Llama-3.1-8B-Instruct",
    "13b": "Llama-2-13b-hf",
    "70b": "Meta-Llama-3.1-70B-Instruct"
}

def make_path(decoding, models):
    # decoding is one of "base", "spec", or "pipe"
    prefix = {"base": "greedy", "spec": "chain", "pipe": "async-chain"}[decoding]
    suffix = {"base": "", "spec": "_8", "pipe": "_0"}[decoding]
    return f"{prefix}_" + "-".join(MODELS[m] for m in models) + f"{suffix}.json"

# [dataset, base_model, folder_path, [(method, display_models, decoding, model_list[, speedup, citation])]]
EXPERIMENTS = [
    ["CNN/DM", "7b", "cnn_dailymail", [
        ("Autoregressive", "LLaMA2-7B", "base", ["7b"]),
        ("Speculative", "68M, 7B", "spec", ["68m", "7b"]),
        ("LayerSkip", "LLaMA2-7B", None, None, 1.86, "elhoushi2024layer"),
        ("PipeSpec", "68M, 7B", "pipe", ["68m", "7b"])
    ]],
    ["CNN/DM", "13b", "cnn_dailymail", [
        ("Autoregressive", "LLaMA2-13B", "base", ["13b"]),
        # Exclude two-model async results
        # ("Speculative", "68M, 13B", "spec", ["68m", "13b"]),
        ("Speculative", "68M, 7B, 13B", "spec", ["68m", "7b", "13b"]),
        ("Draft\\&Verify", "LLaMA2-13B", None, None, 1.56, "zhang2023draft"),
        ("LayerSkip", "LLaMA2-13B", None, None, 1.81, "elhoushi2024layer"),
        ("PipeSpec", "68M, 13B", "pipe", ["68m", "13b"]),
        ("PipeSpec", "7B, 13B", "pipe", ["7b", "13b"]),
        ("PipeSpec", "68M, 7B, 13B", "pipe", ["68m", "7b", "13b"])
    ]],
    ["XSum", "7b", "xsum", [
        ("Autoregressive", "LLaMA2-7B", "base", ["7b"]),
        ("Speculative", "68M, 7B", "spec", ["68m", "7b"]),
        ("LayerSkip", "LLaMA2-7B", None, None, 1.54, "elhoushi2024layer"),
        ("PipeSpec", "68M, 7B", "pipe", ["68m", "7b"])
    ]],
    ["XSum", "13b", "xsum", [
        ("Autoregressive", "LLaMA2-13B", "base", ["13b"]),
        ("Speculative", "68M, 13B", "spec", ["68m", "13b"]),
        ("Speculative", "68M, 7B, 13B", "spec", ["68m", "7b", "13b"]),
        ("Draft\\&Verify", "LLaMA2-13B", None, None, 1.43, "zhang2023draft"),
        ("LayerSkip", "LLaMA2-13B", None, None, 1.48, "elhoushi2024layer"),
        ("PipeSpec", "68M, 13B", "pipe", ["68m", "13b"]),
        ("PipeSpec", "7B, 13B", "pipe", ["7b", "13b"]),
        ("PipeSpec", "68M, 7B, 13B", "pipe", ["68m", "7b", "13b"])
    ]],
    ["HumanEval", "13b", "human_eval", [
        ("Autoregressive", "LLaMA2-13B", "base", ["13b"]),
        ("Speculative", "68M, 13B", "spec", ["68m", "13b"]),
        ("Speculative", "68M, 7B, 13B", "spec", ["68m", "7b", "13b"]),
        ("Draft\\&Verify", "CodeLLaMA2-13B", None, None, 1.46, "zhang2023draft"),
        ("LayerSkip", "LLaMA2-13B", None, None, 1.66, "elhoushi2024layer"),
        ("PipeSpec", "68M, 13B", "pipe", ["68m", "13b"]),
        ("PipeSpec", "7B, 13B", "pipe", ["7b", "13b"]),
        ("PipeSpec", "68M, 7B, 13B", "pipe", ["68m", "7b", "13b"])
    ]],
    ["HumanEval", "70b", "human_eval", [
        ("Autoregressive", "LLaMA3.1-70B", "base", ["70b"]),
        ("Speculative", "8B, 70B", "spec", ["8b", "70b"]),
        ("Speculative", "1B, 8B, 70B", "spec", ["1b", "8b", "70b"]),
        ("PipeSpec", "1B, 70B", "pipe", ["1b", "70b"]),
        ("PipeSpec", "8B, 70B", "pipe", ["8b", "70b"]),
        ("PipeSpec", "1B, 8B, 70B", "pipe", ["1b", "8b", "70b"])
    ]]
]

def bucket_averages(values, num_buckets=5):
    """
    Given a list of numeric values, sort them and split into num_buckets.
    Return the mean of each bucket as a list.
    """
    if not values:
        return [None] * num_buckets
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    bucket_size = math.ceil(n / num_buckets)
    bucket_means = []
    for i in range(num_buckets):
        start = i * bucket_size
        end = min(start + bucket_size, n)
        bucket = sorted_vals[start:end]
        avg = statistics.mean(bucket) if bucket else None
        bucket_means.append(avg)
    return bucket_means

def process_accepted_tokens(json_path):
    """
    Reads a JSON file and processes accepted tokens.
    For a 3-model async chain, there are 2 pairs:
      - Pair 1: model_configs[0] (draft) -> model_configs[1] (verify)
      - Pair 2: model_configs[1] (draft) -> model_configs[2] (verify)
    Returns a list of bucket average lists (one per model pair).
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    results = data.get("results", [])
    # There should be 2 pairs for 3 models.
    num_pairs = 2
    pair_tokens = [[] for _ in range(num_pairs)]
    for sample in results:
        metrics = sample.get("metrics", {})
        accepted_tokens_lists = metrics.get("accepted_tokens", [])
        if len(accepted_tokens_lists) != num_pairs:
            continue
        for i in range(num_pairs):
            # Each accepted_tokens[i] is a list; extend our aggregate list.
            pair_tokens[i].extend(accepted_tokens_lists[i])
    # Compute bucket averages for each pair.
    pair_bucket_avgs = [bucket_averages(tokens, num_buckets=5) for tokens in pair_tokens]
    return pair_bucket_avgs

def generate_markdown_table(results_dir):
    """
    For each experiment in each dataset, filters to async PipeSpec experiments
    that use 3 models. Processes the accepted tokens JSON file and prints a markdown
    table with one row per model pair.
    """
    # Table header:
    header = (
        "| Dataset | Experiment (Models) | Model Pair | 0-20% | 20-40% | 40-60% | 60-80% | 80-100% |\n"
        "|---------|---------------------|------------|-------|--------|--------|--------|---------|"
    )
    rows = [header]
    # Loop over experiments
    for dataset, base_model, folder, experiments in EXPERIMENTS:
        for exp in experiments:
            method, display_models, decoding, model_list, *rest = exp
            # We care only about async method ("PipeSpec") with decoding "pipe"
            # and with three models.
            if method != "PipeSpec" or decoding != "pipe" or (model_list is None) or (len(model_list) != 3):
                continue
            # Build the JSON file path.
            json_filename = make_path(decoding, model_list)
            json_path = Path(results_dir) / folder / json_filename
            try:
                pair_bucket_avgs = process_accepted_tokens(json_path)
            except Exception as e:
                print(f"Warning: Could not process {json_path}: {e}")
                continue

            # Build model pair descriptions using the full model names from MODELS.
            # For a 3-model async chain, pair 1 is model_list[0] -> model_list[1],
            # and pair 2 is model_list[1] -> model_list[2].
            pair_descs = [
                f"{MODELS[model_list[0]]} -> {MODELS[model_list[1]]}",
                f"{MODELS[model_list[1]]} -> {MODELS[model_list[2]]}"
            ]
            # For each model pair, add a row.
            for pair_idx, bucket_avgs in enumerate(pair_bucket_avgs):
                # Format bucket averages to 2 decimal places (or "N/A")
                buckets_fmt = " | ".join(f"{avg:.2f}" if avg is not None else "N/A" for avg in bucket_avgs)
                row = f"| {dataset} | {display_models} | {pair_descs[pair_idx]} | {buckets_fmt} |"
                rows.append(row)
    # Print the markdown table.
    print("\n".join(rows))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python print_accepted_tokens.py <results_dir>")
        sys.exit(1)
    results_directory = sys.argv[1]
    generate_markdown_table(results_directory)

