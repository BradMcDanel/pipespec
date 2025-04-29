import json
import argparse
from typing import Dict, Any, List, Set
from statistics import mean

def get_active_gpus(metadata: Dict[str, Any]) -> Set[int]:
    """Extract active GPU indices from metadata"""
    active_gpus = set()
    for config in metadata["model_configs"]:
        for device in config["devices"]:
            if device.startswith("cuda:"):
                gpu_idx = int(device.split(":")[1])
                active_gpus.add(gpu_idx)
    return active_gpus

def analyze_results_file(file_path: str):
    """Analyze timing and GPU statistics from a single results JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get active GPUs from metadata
    active_gpus = get_active_gpus(data["metadata"])
    
    # Initialize aggregation variables
    total_time = 0
    total_tokens = 0
    times_per_token = []
    
    # Initialize GPU stat aggregation
    gpu_utils = {gpu_idx: [] for gpu_idx in active_gpus}
    gpu_powers = {gpu_idx: [] for gpu_idx in active_gpus}
    
    # Process each sample
    for sample in data["results"]:
        metrics = sample["metrics"]
        sample_time = metrics['total_time']
        sample_tokens = metrics['tokens_generated']
        
        total_time += sample_time
        total_tokens += sample_tokens
        
        # Calculate time per token for this sample (in milliseconds)
        if sample_tokens > 0:
            times_per_token.append((sample_time / sample_tokens) * 1000)
        
        # Process GPU stats for this sample
        for stat in sample["gpustats"]:
            for gpu_idx in active_gpus:
                gpu_utils[gpu_idx].append(stat["gpu_utilizations"][gpu_idx])
                gpu_powers[gpu_idx].append(stat["gpu_powers"][gpu_idx])
    
    # Calculate timing statistics
    avg_time_per_token = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0
    min_time_per_token = min(times_per_token) if times_per_token else 0
    max_time_per_token = max(times_per_token) if times_per_token else 0
    
    # Calculate GPU statistics
    gpu_stats = {}
    total_energy = 0
    for gpu_idx in active_gpus:
        avg_util = mean(gpu_utils[gpu_idx])
        avg_power = mean(gpu_powers[gpu_idx])
        
        # Calculate energy in joules (power in watts * time in seconds)
        energy = avg_power * total_time
        total_energy += energy
        
        gpu_stats[gpu_idx] = {
            'avg_util': avg_util,
            'avg_power': avg_power,
            'total_energy': energy
        }
    
    # Calculate energy per token
    energy_per_token = (total_energy / total_tokens) if total_tokens > 0 else 0
    
    # Print results
    print("\nResults Analysis")
    print("=" * 70)
    print(f"Total samples processed: {len(data['results'])}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    
    print("\nPer-token Statistics:")
    print(f"Average time per token: {avg_time_per_token:.2f} ms")
    print(f"Minimum time per token: {min_time_per_token:.2f} ms")
    print(f"Maximum time per token: {max_time_per_token:.2f} ms")
    print(f"Energy per token: {energy_per_token:.2f} joules")
    
    print("\nGPU Statistics:")
    print("-" * 70)
    print(f"{'GPU':<6} {'Avg Util':<10} {'Avg Power':<12} {'Total Energy'}")
    print("-" * 70)
    for gpu_idx in sorted(active_gpus):
        stats = gpu_stats[gpu_idx]
        print(f"GPU {gpu_idx:<2} {stats['avg_util']:.2f}% {stats['avg_power']:.2f}W {stats['total_energy']:.2f}J")
    
    print(f"\nTotal energy consumption: {total_energy:.2f}J")

def main():
    parser = argparse.ArgumentParser(description="Analyze timing and GPU statistics from a single results JSON file.")
    parser.add_argument("file", help="Path to the JSON results file")
    args = parser.parse_args()
    
    analyze_results_file(args.file)

if __name__ == "__main__":
    main()
