import json
from pathlib import Path

# Mapping for model names.
MODELS = {
    "68m": "llama-68m",
    "1b": "Llama-3.2-1B-Instruct",
    "7b": "Llama-2-7b-hf",
    "8b": "Llama-3.1-8B-Instruct",
    "13b": "Llama-2-13b-hf",
    "70b": "Meta-Llama-3.1-70B-Instruct"
}

def make_path(decoding, models):
    """
    Constructs a filename based on the decoding type and model list.
    
    For the autoregressive ("base") decoding, the filename is of the form:
      greedy_<modelname>.json
    """
    prefix = {"base": "greedy", "spec": "chain", "pipe": "async-chain"}[decoding]
    suffix = {"base": "", "spec": "_8", "pipe": "_0"}[decoding]
    return f"{prefix}_" + "-".join(MODELS[m] for m in models) + f"{suffix}.json"

def get_time_per_token(file_path):
    """
    Given the path to a result JSON file, returns the average time (in milliseconds) per token.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    total_time = sum(s["metrics"]["total_time"] for s in data["results"])
    total_tokens = sum(s["metrics"]["tokens_generated"] for s in data["results"])
    return (total_time / total_tokens * 1000) if total_tokens else 0

def print_tokens_per_second(results_dir, dataset_folder):
    """
    For a given dataset folder, iterates over all models and prints tokens per second (TPS)
    for each model's autoregressive results.
    """
    print(f"Tokens per second for dataset folder: {dataset_folder}\n")
    print(f"{'Model':<30} {'TPS':>10}")
    print("-" * 42)
    
    for model_key in MODELS:
        # Construct file path using the autoregressive (base) decoding.
        file_name = make_path("base", [model_key])
        file_path = Path(results_dir) / dataset_folder / file_name
        try:
            ms_per_token = get_time_per_token(file_path)
            tps = 1000 / ms_per_token if ms_per_token else 0
            print(f"{MODELS[model_key]:<30} {tps:10.2f}")
        except Exception as e:
            print(f"{MODELS[model_key]:<30} {'Error':>10} ({e})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <results_dir> <dataset_folder>")
        sys.exit(1)
    results_directory = sys.argv[1]
    dataset_folder = sys.argv[2]
    print_tokens_per_second(results_directory, dataset_folder)

