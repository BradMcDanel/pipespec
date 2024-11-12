import json
import argparse
from collections import defaultdict
import statistics
import os
from typing import List, Dict, Any
import numpy as np

def calculate_verification_stats(accepted_tokens: List[List[int]]) -> Dict[str, float]:
    """Calculate verification statistics from accepted tokens arrays"""
    stats = {}
    
    if not accepted_tokens:
        return stats
        
    # Flatten accepted tokens if needed
    all_tokens = [t for sublist in accepted_tokens for t in sublist]
    
    if all_tokens:
        stats["mean_tokens_accepted"] = statistics.mean(all_tokens)
        stats["median_tokens_accepted"] = statistics.median(all_tokens)
        stats["min_tokens_accepted"] = min(all_tokens)
        stats["max_tokens_accepted"] = max(all_tokens)
        stats["stdev_tokens_accepted"] = statistics.stdev(all_tokens) if len(all_tokens) > 1 else 0
        
        # Calculate acceptance rate distribution
        total_tokens = len(all_tokens)
        single_token_accepts = sum(1 for t in all_tokens if t == 1)
        stats["single_token_acceptance_rate"] = single_token_accepts / total_tokens
        stats["multi_token_acceptance_rate"] = 1 - stats["single_token_acceptance_rate"]
        
    return stats

def analyze_gpu_stats(gpustats: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze GPU utilization and memory usage"""
    stats = {}
    
    if not gpustats:
        return stats
        
    # Extract statistics for each GPU
    gpu_utils = defaultdict(list)
    gpu_mems = defaultdict(list)
    gpu_powers = defaultdict(list)
    
    for stat in gpustats:
        for i, (util, mem, power) in enumerate(zip(
            stat['gpu_utilizations'],
            stat['gpu_memories'],
            stat['gpu_powers']
        )):
            gpu_utils[i].append(util)
            gpu_mems[i].append(mem)
            gpu_powers[i].append(power)
    
    # Calculate statistics for each GPU
    for gpu_id in gpu_utils.keys():
        prefix = f"gpu{gpu_id}_"
        stats[prefix + "util_mean"] = statistics.mean(gpu_utils[gpu_id])
        stats[prefix + "util_peak"] = max(gpu_utils[gpu_id])
        stats[prefix + "mem_mean"] = statistics.mean(gpu_mems[gpu_id])
        stats[prefix + "mem_peak"] = max(gpu_mems[gpu_id])
        stats[prefix + "power_mean"] = statistics.mean(gpu_powers[gpu_id])
        stats[prefix + "power_peak"] = max(gpu_powers[gpu_id])
    
    return stats

def analyze_model_times(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Analyze model execution times"""
    stats = {}
    
    if 'model_times' not in metrics:
        return stats
        
    model_times = metrics['model_times']
    for i, times in enumerate(model_times):
        if times:
            prefix = f"model{i}_"
            stats[prefix + "mean_time"] = statistics.mean(times)
            stats[prefix + "median_time"] = statistics.median(times)
            stats[prefix + "min_time"] = min(times)
            stats[prefix + "max_time"] = max(times)
            stats[prefix + "stdev_time"] = statistics.stdev(times) if len(times) > 1 else 0
            
    return stats

def analyze_results(file_paths: List[str]):
    """Analyze results from multiple benchmark files"""
    all_results = {}
    reference_time = None
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata and results
        metadata = data["metadata"]
        results = data["results"]
        strategy = metadata["strategy"]
        
        # Initialize storage for this strategy
        strategy_stats = {
            "metadata": metadata,
            "aggregate_metrics": defaultdict(list),
            "verification_stats": [],
            "gpu_stats": [],
            "model_timing_stats": [],
            "samples_processed": len(results)
        }
        
        # Process each sample
        for sample in results:
            metrics = sample["metrics"]
            
            # Store basic metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    strategy_stats["aggregate_metrics"][key].append(value)
            
            # Calculate verification statistics if available
            if "accepted_tokens" in metrics:
                strategy_stats["verification_stats"].append(
                    calculate_verification_stats(metrics["accepted_tokens"])
                )
            
            # Calculate GPU statistics
            if "gpustats" in sample:
                strategy_stats["gpu_stats"].append(
                    analyze_gpu_stats(sample["gpustats"])
                )
            
            # Calculate model timing statistics
            strategy_stats["model_timing_stats"].append(
                analyze_model_times(metrics)
            )
        
        # Calculate final statistics
        metrics = strategy_stats["aggregate_metrics"]
        if 'total_time' in metrics and 'tokens_generated' in metrics:
            total_time = sum(metrics['total_time'])
            total_tokens = sum(metrics['tokens_generated'])
            avg_time_per_token = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0
            strategy_stats["time_per_token"] = avg_time_per_token
            
            if strategy == 'greedy':
                reference_time = avg_time_per_token
        
        all_results[strategy] = strategy_stats
    
    # Print analysis
    print("\nBenchmark Analysis Results")
    print("=========================\n")
    
    for strategy, stats in all_results.items():
        print(f"\nStrategy: {strategy}")
        print("=" * (len(strategy) + 10))
        print(f"Models used: {', '.join(cfg['path'] for cfg in stats['metadata']['model_configs'])}")
        print(f"Number of samples: {stats['samples_processed']}")
        print(f"Lookahead: {stats['metadata']['lookahead']}")
        print("\nPerformance Metrics:")
        print("-" * 20)
        
        # Print time per token and speedup
        print(f"Average time per token: {stats['time_per_token']:.3f} ms")
        if reference_time and strategy != 'greedy':
            speedup = reference_time / stats['time_per_token']
            print(f"Speedup vs greedy: {speedup:.2f}x")
        
        # Print verification statistics if available
        if stats['verification_stats']:
            print("\nVerification Statistics:")
            print("-" * 20)
            for metric, value in stats['verification_stats'][0].items():
                print(f"{metric}: {value:.3f}")
        
        # Print GPU utilization
        if stats['gpu_stats']:
            print("\nGPU Statistics:")
            print("-" * 20)
            for metric, value in stats['gpu_stats'][0].items():
                print(f"{metric}: {value:.2f}")
        
        print("\n" + "=" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and print statistics.")
    parser.add_argument("input_files", nargs='+', help="Paths to the JSON files containing benchmark results")
    args = parser.parse_args()
    
    analyze_results(args.input_files)

if __name__ == "__main__":
    main()
