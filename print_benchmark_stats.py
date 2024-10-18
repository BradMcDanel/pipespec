import json
import argparse
from collections import defaultdict
import statistics

def analyze_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    strategy_metrics = defaultdict(lambda: defaultdict(list))
    strategy_samples = defaultdict(int)
    greedy_time_per_token = None

    for sample in data:
        strategy = sample['strategy']
        metrics = sample['metrics']
        
        strategy_samples[strategy] += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                strategy_metrics[strategy][key].append(value)

    print("Benchmark Analysis Results:")
    print("===========================")
    for strategy, metrics in strategy_metrics.items():
        print(f"\nStrategy: {strategy}")
        print(f"Number of samples: {strategy_samples[strategy]}")
        print("-" * (len(strategy) + 10))
        for metric, values in metrics.items():
            if values and metric != 'tokens_generated':  # Exclude tokens_generated
                mean = statistics.mean(values)
                median = statistics.median(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                print(f"{metric}: Mean: {mean:.6f}, Median: {median:.6f}, StdDev: {stdev:.6f}")
        
        if 'total_time' in metrics and 'tokens_generated' in metrics:
            total_time = sum(metrics['total_time'])
            total_tokens = sum(metrics['tokens_generated'])
            avg_time_per_token = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0
            print(f"Average time per token: {avg_time_per_token:.6f} ms")

            if strategy == 'greedy':
                greedy_time_per_token = avg_time_per_token
            elif greedy_time_per_token:
                speedup = greedy_time_per_token / avg_time_per_token if avg_time_per_token > 0 else float('inf')
                print(f"Speedup relative to greedy: {speedup:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and print statistics.")
    parser.add_argument("input_file", help="Path to the input JSON file containing benchmark results")
    args = parser.parse_args()
    
    analyze_results(args.input_file)

if __name__ == "__main__":
    main()
