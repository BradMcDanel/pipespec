import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    set_plotting_style,
    style_axis,
    add_legend,
    COLORS,
    MODEL_CONFIGS
)

def get_bin_index(token: int) -> int:
    """Get the bin index for a given token count."""
    if token <= 9:
        return token - 1  # Bins 0-8 for tokens 1-9
    elif token <= 14:
        return 9  # Bin 9 for tokens 10-14
    elif token <= 19:
        return 10  # Bin 10 for tokens 15-19
    elif token <= 24:
        return 11  # Bin 11 for tokens 20-24
    elif token <= 29:
        return 12  # Bin 12 for tokens 25-29
    else:
        return 13  # Bin 13 for tokens 30+

def analyze_frequencies(file_path: str) -> Tuple[List[float], int]:
    """Analyze token frequencies from a results file."""
    token_frequencies = [0] * 14
    total_count = 0

    with open(file_path, 'r') as file:
        data = json.load(file)

    for result in data.get('results', []):
        metrics = result.get('metrics', {})
        acc_tokens = metrics.get('accepted_tokens', [[]])[-1]
        
        for token in acc_tokens:
            bin_idx = get_bin_index(token)
            token_frequencies[bin_idx] += 1
            total_count += 1

    normalized_frequencies = [
        round(freq / total_count * 100, 2) if total_count > 0 else 0.00
        for freq in token_frequencies
    ]

    return normalized_frequencies, total_count

def analyze_token_times(file_path: str) -> Tuple[List[float], List[int]]:
    """Analyze normalized time (time per token) for different acceptance counts."""
    time_sums = [0.0] * 14
    time_counts = [0] * 14
    
    with open(file_path, 'r') as file:
        data = json.load(file)

    for result in data.get('results', []):
        metrics = result.get('metrics', {})
        acc_tokens = metrics.get('accepted_tokens', [[]])[-1]
        model_times = metrics.get('model_times', [[]])[-1]
        
        if not acc_tokens or not model_times:
            continue
            
        for time, tokens in zip(model_times, acc_tokens):
            time_per_token = time / tokens if tokens > 0 else 0
            bin_idx = get_bin_index(tokens)
            time_sums[bin_idx] += time_per_token
            time_counts[bin_idx] += 1

    avg_times = [
        time_sum / count if count > 0 else 0
        for time_sum, count in zip(time_sums, time_counts)
    ]
    
    return avg_times, time_counts

def plot_stacked_analysis(folder_path: str):
    """Create stacked plots showing frequency and time analysis."""
    set_plotting_style()
    
    # Create figure with two subplots without sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[1.2, 1])

    # Create x-axis labels
    x_labels = [str(i) for i in range(1, 10)]  # 1-9
    x_labels.extend(['10-14', '15-19', '20-24', '25-29', '30+'])
    x = np.arange(len(x_labels))
    
    # Filter out baseline model
    spec_models = {k: v for k, v in MODEL_CONFIGS.items() 
                  if not v.startswith('BL')}
    
    n_models = len(spec_models)
    bar_width = 0.8 / n_models

    # Plot both frequency and time data
    for i, (filename, model_name) in enumerate(spec_models.items()):
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {filename} - file not found")
            continue
            
        print(f"Processing {model_name}")
        
        # Process frequency data
        frequencies, count = analyze_frequencies(file_path)
        offset = bar_width * (i - n_models / 2 + 0.5)
        ax1.bar(x + offset, 
               frequencies, 
               bar_width,
               label=model_name,
               color=COLORS[model_name],
               alpha=0.8)
        
        # Process time data
        avg_times, counts = analyze_token_times(file_path)
        ax2.bar(x + offset, 
               avg_times, 
               bar_width,
               label=model_name,
               color=COLORS[model_name],
               alpha=0.8)

    # Style the top plot (frequencies)
    style_axis(ax1,
              title='Token Generation: Distribution and Timing Analysis',
              ylabel='Frequency (%)')
    
    # Style the bottom plot (times)
    style_axis(ax2,
              xlabel='Number of Accepted Tokens per Verify Model Step',
              ylabel='Time per Token (seconds)')

    # Set x-axis ticks and labels for both plots
    for ax in [ax1, ax2]:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', rotation=0)  # Ensure labels are horizontal
    
    # Add single legend at the top
    legend = add_legend(ax2,
                       loc='upper right',
                       bbox_to_anchor=(0.98, 0.98),
                       ncol=2)
    
    # Reduce spacing between subplots
    plt.subplots_adjust(hspace=0.15)  # Slightly increased to accommodate both x-labels
    
    # Save figures
    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', 'token-analysis')
    
    # Save as PDF
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight', pad_inches=0.1)
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate token analysis plots.")
    parser.add_argument("folder", help="Path to the folder containing results")
    
    args = parser.parse_args()
    plot_stacked_analysis(args.folder)

if __name__ == "__main__":
    main()
