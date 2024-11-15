import json
import argparse
import os
from typing import List, Dict, Any

def extract_model_sizes(model_str: str) -> List[str]:
    """Extract model sizes from a model string"""
    models = []
    for part in model_str.split('-'):
        if part.endswith('B'):
            models.append(part)
    return models

def get_model_size_value(size_str: str) -> int:
    """Convert model size string to numeric value (e.g., '70B' -> 70)"""
    return int(size_str.rstrip('B'))

def find_largest_model_size(file_paths: List[str]) -> str:
    """Find the largest model size from a list of file paths"""
    max_size = 0
    max_size_str = None
    
    for path in file_paths:
        filename = os.path.basename(path)
        model_str = filename.split('_')[1].replace('.json', '')
        sizes = extract_model_sizes(model_str)
        if sizes:
            size_value = get_model_size_value(sizes[-1])  # Get the last (largest) model size
            if size_value > max_size:
                max_size = size_value
                max_size_str = sizes[-1]
    
    return max_size_str

def analyze_dataset_results(folder_path: str):
    """Analyze results for a single dataset folder"""
    # Get all JSON files in the folder
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.endswith('.json')]
    
    # Find the largest model size in this dataset
    largest_model = find_largest_model_size(file_paths)
    if not largest_model:
        print(f"Error: Could not determine largest model size in {folder_path}")
        return
    
    # Find baseline (largest model with greedy decoding)
    baseline_time = None
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename.startswith('greedy_') and largest_model in filename:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            total_time = 0
            total_tokens = 0
            for sample in data["results"]:
                metrics = sample["metrics"]
                total_time += metrics['total_time']
                total_tokens += metrics['tokens_generated']
            
            baseline_time = (total_time / total_tokens) * 1000
            break
    
    # Collect all results for largest model
    results = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if not largest_model in filename:
            continue
            
        strategy = filename.split('_')[0]
        model_str = filename.split('_')[1].replace('.json', '')
        model_sizes = extract_model_sizes(model_str)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Calculate average time per token
        total_time = 0
        total_tokens = 0
        for sample in data["results"]:
            metrics = sample["metrics"]
            total_time += metrics['total_time']
            total_tokens += metrics['tokens_generated']
        
        avg_time = (total_time / total_tokens) * 1000  # convert to ms
        
        # Format the configuration name with new naming convention
        if 'async-chain' in strategy:
            method = 'PipeSpec'
        elif 'chain' in strategy:
            method = 'Speculative'
        elif 'greedy' in strategy:
            method = 'Autoregressive'
            
        results.append({
            'method': method,
            'time': avg_time,
            'base_model': model_sizes[-1],
            'helper_models': model_sizes[:-1] if len(model_sizes) > 1 else [],
            'full_path': file_path
        })
    
    # Print results
    dataset_name = os.path.basename(folder_path)
    print(f"\nResults for dataset: {dataset_name}")
    print("=" * 80)
    
    # Print header
    if baseline_time:
        print(f"{'Base Model':<12} {'Method':<15} {'Helper Models':<20} {'Mean Token Time':<20} {'Speedup':<10}")
    else:
        print(f"{'Base Model':<12} {'Method':<15} {'Helper Models':<20} {'Mean Token Time':<20}")
    print("-" * 80)
    
    # Sort by method order
    method_order = {'Autoregressive': 1, 'Speculative': 2, 'PipeSpec': 3}
    results.sort(key=lambda x: method_order[x['method']])
    
    # Print results
    for result in results:
        helper_str = ','.join(result['helper_models']) if result['helper_models'] else '-'
        if baseline_time:
            speedup = baseline_time / result['time'] if result['method'] != 'Autoregressive' else 1.00
            print(f"{result['base_model']:<12} {result['method']:<15} {helper_str:<20} {result['time']:.2f} ms/token {' '*5} {speedup:.2f}×")
        else:
            print(f"{result['base_model']:<12} {result['method']:<15} {helper_str:<20} {result['time']:.2f} ms/token")

def analyze_results(root_folder: str):
    """Analyze results for all dataset folders"""
    # Process each dataset folder
    for dataset_folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, dataset_folder)
        if os.path.isdir(folder_path):
            analyze_dataset_results(folder_path)

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and print statistics.")
    parser.add_argument("folder", help="Path to the root folder containing dataset results")
    args = parser.parse_args()
    
    analyze_results(args.folder)

if __name__ == "__main__":
    main()
