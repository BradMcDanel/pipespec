import json
import argparse
from collections import defaultdict
import statistics

def analyze_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    strategy_metrics = defaultdict(lambda: defaultdict(list))
    for sample in data:
        strategy = sample['strategy']
        metrics = sample['metrics']
        
        # Collect all metrics for each strategy
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                strategy_metrics[strategy][key].append(value)

    print("Benchmark Analysis Results:")
    print("===========================")
    for strategy, metrics in strategy_metrics.items():
        print(f"\nStrategy: {strategy}")
        print("-" * (len(strategy) + 10))
        for metric, values in metrics.items():
            if values:  # Check if we have any values for this metric
                mean = statistics.mean(values)
                median = statistics.median(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                print(f"{metric}:")
                print(f"  Mean:   {mean:.6f}")
                print(f"  Median: {median:.6f}")
                print(f"  StdDev: {stdev:.6f}")

        # Calculate and print average time per token in milliseconds
        if 'total_time' in metrics and 'tokens_generated' in metrics:
            total_time = sum(metrics['total_time'])
            total_tokens = sum(metrics['tokens_generated'])
            avg_time_per_token = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0
            print(f"Average time per token: {avg_time_per_token:.6f} ms")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and print statistics.")
    parser.add_argument("input_file", help="Path to the input JSON file containing benchmark results")
    args = parser.parse_args()
    
    analyze_results(args.input_file)

if __name__ == "__main__":
    main()
