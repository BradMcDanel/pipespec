import json
import argparse
from typing import Dict, Any, List, Set
from statistics import mean
from dataclasses import dataclass
from pathlib import Path

# Model name mappings
MODEL_NAMES = {
    "1B": "Llama-3.2-1B-Instruct",
    "8B": "Llama-3.1-8B-Instruct",
    "70B": "Meta-Llama-3.1-70B-Instruct"
}

# Strategy name mappings with their run number suffixes
STRATEGIES = {
    "base": {"prefix": "greedy", "suffix": ".json"},          # no run number for baseline
    "spec": {"prefix": "chain", "suffix": "_8.json"},        # speculative uses _8
    "pipe": {"prefix": "async-chain", "suffix": "_0.json"}   # pipespec uses _0
}

@dataclass
class ResultGroup:
    dataset_name: str
    dataset_folder: str
    baseline_model: str
    experiments: List[Dict[str, str]]

def make_pattern(strategy: str, models: List[str]) -> str:
    """Create file pattern from strategy and models with appropriate suffix"""
    strat = STRATEGIES[strategy]
    return f"{strat['prefix']}_" + "-".join(MODEL_NAMES[m] for m in models) + strat['suffix']

# Configure the result groups we want to analyze
RESULT_GROUPS = [
    ResultGroup(
        dataset_name="HumanEval",
        dataset_folder="humaneval",
        baseline_model="70B",
        experiments=[
            # Baseline autoregressive
            {"pattern": make_pattern("base", ["70B"]), "method": "Autoregressive", "models": "L-70B"},
            # Speculative variants
            {"pattern": make_pattern("spec", ["1B", "70B"]), "method": "Speculative", "models": "L-1B,L-70B"},
            {"pattern": make_pattern("spec", ["8B", "70B"]), "method": "Speculative", "models": "L-8B,L-70B"},
            {"pattern": make_pattern("spec", ["1B", "8B", "70B"]), "method": "Speculative", "models": "L-1B,L-8B,L-70B"},
            # PipeSpec variants
            {"pattern": make_pattern("pipe", ["8B", "70B"]), "method": "PipeSpec", "models": "L-8B,L-70B"},
            {"pattern": make_pattern("pipe", ["1B", "70B"]), "method": "PipeSpec", "models": "L-1B,L-70B"},
            {"pattern": make_pattern("pipe", ["1B", "8B", "70B"]), "method": "PipeSpec", "models": "L-1B,L-8B,L-70B"},
        ]
    )
]

def get_active_gpus(metadata: Dict[str, Any]) -> Set[int]:
    """Extract active GPU indices from metadata"""
    active_gpus = set()
    for config in metadata["model_configs"]:
        for device in config["devices"]:
            if device.startswith("cuda:"):
                gpu_idx = int(device.split(":")[1])
                active_gpus.add(gpu_idx)
    return active_gpus

def find_matching_file(directory: Path, pattern: str) -> Path:
    """Find a file matching the pattern in the directory"""
    for file in directory.glob("*.json"):
        if pattern in file.name:
            return file
    raise FileNotFoundError(f"No file matching pattern '{pattern}' found in {directory}")

def analyze_single_file(file_path: str) -> Dict[str, float]:
    """Analyze a single results file and return key metrics"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    active_gpus = get_active_gpus(data["metadata"])
    
    # Initialize aggregation variables
    total_time = 0
    total_tokens = 0
    gpu_utils = {gpu_idx: [] for gpu_idx in active_gpus}
    gpu_powers = {gpu_idx: [] for gpu_idx in active_gpus}
    
    # Process each sample
    for sample in data["results"]:
        metrics = sample["metrics"]
        total_time += metrics['total_time']
        total_tokens += metrics['tokens_generated']
        
        # Process GPU stats
        for stat in sample["gpustats"]:
            for gpu_idx in active_gpus:
                gpu_utils[gpu_idx].append(stat["gpu_utilizations"][gpu_idx])
                gpu_powers[gpu_idx].append(stat["gpu_powers"][gpu_idx])
    
    # Calculate averages
    avg_time_per_token = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0
    
    # Calculate GPU averages
    avg_util = mean([mean(utils) for utils in gpu_utils.values()])
    avg_power = mean([mean(powers) for powers in gpu_powers.values()])
    
    # Calculate total energy and energy per token
    total_energy = sum([mean(powers) * total_time for powers in gpu_powers.values()])
    energy_per_token = total_energy / total_tokens if total_tokens > 0 else 0
    
    return {
        "avg_time": avg_time_per_token,
        "avg_util": avg_util,
        "avg_power": avg_power,
        "energy_per_token": energy_per_token
    }

def generate_results_table(results_dir: str):
    """Generate results table from configured experiments"""
    results_path = Path(results_dir)
    
    print("\nResults Table")
    print("=" * 120)
    print(f"{'Dataset':<15} {'Method':<15} {'Models':<20} {'Avg Time (ms)':<15} {'Speedup':<10} "
          f"{'Avg Util (%)':<12} {'Avg Power (W)':<15} {'Energy/tok (J)'}")
    print("-" * 120)
    
    for group in RESULT_GROUPS:
        baseline_metrics = None
        
        for exp in group.experiments:
            try:
                file_path = find_matching_file(results_path / group.dataset_folder, exp["pattern"])
                metrics = analyze_single_file(str(file_path))
                
                # If this is the baseline experiment, save its metrics
                if exp["method"] == "Autoregressive":
                    baseline_metrics = metrics
                
                # Calculate speedup relative to baseline
                speedup = baseline_metrics["avg_time"] / metrics["avg_time"] if baseline_metrics else 1.0
                
                print(f"{group.dataset_name:<15} {exp['method']:<15} {exp['models']:<20} "
                      f"{metrics['avg_time']:.2f}{'ms':<9} {speedup:.2f}{'×':<8} "
                      f"{metrics['avg_util']:.1f}{'%':<9} {metrics['avg_power']:.1f}{'W':<11} "
                      f"{metrics['energy_per_token']:.2f}")
                
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        print("-" * 120)

def main():
    parser = argparse.ArgumentParser(description="Generate results table from benchmark results.")
    parser.add_argument("directory", help="Path to the root directory containing benchmark results")
    args = parser.parse_args()
    
    generate_results_table(args.directory)

if __name__ == "__main__":
    main()
