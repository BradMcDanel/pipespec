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
    get_figure_axes,
    style_axis,
    add_legend,
    COLORS,
    MODEL_CONFIGS
)

def get_bin_index(token: int) -> int:
    """
    Get the bin index for a given token count.
    
    Args:
        token: Number of tokens
        
    Returns:
        Index of the bin this token count belongs to
    """
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

def analyze_token_frequencies(file_path: str) -> Tuple[List[float], int]:
    """
    Analyze token frequencies from a results file with custom binning.
    
    Args:
        file_path: Path to the JSON results file
        
    Returns:
        Tuple of normalized frequencies and total count
    """
    # 14 bins total: 1-9 individual, then 10-14, 15-19, 20-24, 25-29, 30+
    token_frequencies = [0] * 14
    total_count = 0

    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate through all results
    for result in data.get('results', []):
        metrics = result.get('metrics', {})
        # Get the last accepted tokens array for speculative models
        acc_tokens = metrics.get('accepted_tokens', [[]])[-1]
        
        for token in acc_tokens:
            bin_idx = get_bin_index(token)
            token_frequencies[bin_idx] += 1
            total_count += 1

    # Normalize frequencies to percentages
    normalized_frequencies = [
        round(freq / total_count * 100, 2) if total_count > 0 else 0.00
        for freq in token_frequencies
    ]

    return normalized_frequencies, total_count

def plot_token_frequencies(folder_path: str):
    """
    Create token frequency plot for all models with custom binning.
    
    Args:
        folder_path: Path to the folder containing result files
    """
    set_plotting_style()
    fig, ax = get_figure_axes('double_column')

    # Create x-axis labels
    x_labels = [str(i) for i in range(1, 10)]  # 1-9
    x_labels.extend(['10-14', '15-19', '20-24', '25-29', '30+'])
    x = np.arange(len(x_labels))
    
    # Filter out baseline model and count remaining models
    spec_models = {k: v for k, v in MODEL_CONFIGS.items() 
                  if not v.startswith('BL')}  # Skip baseline model
    n_models = len(spec_models)
    bar_width = 0.8 / n_models  # Leave 20% space between groups
    
    # Track processed models for legend
    processed_models = []

    for i, (filename, model_name) in enumerate(spec_models.items()):
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {filename} - file not found")
            continue
            
        print(f"Processing {model_name}")
        frequencies, total_count = analyze_token_frequencies(file_path)
        
        if total_count > 0:
            # Calculate offset for bar position
            offset = bar_width * (i - n_models / 2 + 0.5)
            
            # Plot bars
            ax.bar(x + offset, 
                  frequencies, 
                  bar_width,
                  label=model_name,
                  color=COLORS[model_name],
                  alpha=0.8)
            
            processed_models.append(model_name)

    # Style the plot
    style_axis(ax,
              title='Token Frequency Distribution',
              xlabel='Number of Accepted Tokens',
              ylabel='Frequency (%)')

    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    # Add legend with processed models only
    add_legend(ax,
              loc='upper right',
              bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    
    # Save figures
    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', 'token-frequency-distribution')
    
    # Save as PDF
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight', pad_inches=0.1)
    
    # Save as PNG with high DPI
    plt.savefig(f'{output_path}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate token frequency plots for benchmark results.")
    parser.add_argument("folder", help="Path to the folder containing results")
    
    args = parser.parse_args()
    plot_token_frequencies(args.folder)

if __name__ == "__main__":
    main()
