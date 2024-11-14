import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz.utils import (
    set_plotting_style, 
    get_figure_axes, 
    style_axis, 
    add_legend,
    COLORS, 
    MODEL_CONFIGS
)


def get_cumulative_tokens_times(metrics: Dict, is_greedy: bool) -> Tuple[List[float], List[int]]:
    """Calculate cumulative tokens and times for a single sample"""
    cumulative_times = []
    tokens = []
    
    if is_greedy:
        if 'token_times' not in metrics:
            return [], []
            
        times = metrics['token_times']
        cumulative_time = 0
        for i, time in enumerate(times):
            cumulative_time += time
            cumulative_times.append(cumulative_time)
            tokens.append(i + 1)
    else:
        if 'model_times' not in metrics:
            return [], []
            
        times = metrics['model_times'][-1]
        tokens_emitted = 0
        cumulative_time = 0
        
        accepted_tokens = metrics.get('accepted_tokens', [])[-1] if metrics.get('accepted_tokens') else [1] * len(times)
        
        for i, (time, accepted) in enumerate(zip(times, accepted_tokens)):
            cumulative_time += time
            tokens_emitted += accepted
            cumulative_times.append(cumulative_time)
            tokens.append(tokens_emitted)
    
    return cumulative_times, tokens

def aggregate_samples(samples: List[Dict], is_greedy: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate data across all samples and compute statistics"""
    all_times = []
    all_tokens = []
    
    for sample in samples:
        times, tokens = get_cumulative_tokens_times(sample['metrics'], is_greedy)
        if times and tokens:
            all_times.append(times)
            all_tokens.append(tokens)
    
    if not all_times:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    min_tokens = min(max(tokens) for tokens in all_tokens)
    token_positions = np.arange(1, min_tokens + 1)
    
    aligned_times = []
    for times, tokens in zip(all_times, all_tokens):
        times = np.array(times)
        tokens = np.array(tokens)
        
        mask = tokens <= min_tokens
        if not any(mask):
            continue
            
        times = times[mask]
        tokens = tokens[mask]
        
        interpolated_times = np.interp(token_positions, tokens, times)
        aligned_times.append(interpolated_times)
    
    aligned_times = np.array(aligned_times)
    
    mean_times = np.mean(aligned_times, axis=0)
    std_times = np.std(aligned_times, axis=0)
    
    return token_positions, mean_times, mean_times - std_times, mean_times + std_times

def plot_token_generation(folder_path: str):
    """Create token generation plot with averaged samples"""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for filename, label in MODEL_CONFIGS.items():
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {filename} - file not found")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data.get("results"):
            continue
        
        is_greedy = 'greedy' in filename
        print(f"Processing {label} with {len(data['results'])} samples")
        
        tokens, mean_times, lower_times, upper_times = aggregate_samples(data["results"], is_greedy)
        if len(tokens) > 0:
            color = COLORS[label]
            ax.plot(mean_times, tokens, color=color, label=label, linewidth=1.5)
            ax.fill_betweenx(tokens, lower_times, upper_times, color=color, alpha=0.1)

    style_axis(ax,
              xlabel='Time (seconds)',
              ylabel='Number of Verified Tokens',
              title='Token Generation Comparison')
    
    # Move legend inside the plot area
    legend = ax.legend(
        loc='center right',    # Position on right side inside plot
        bbox_to_anchor=(0.98, 0.5),  # Fine-tune position
        ncol=1,               # Single column for better readability
        frameon=True,         # Keep the frame
        framealpha=0.9,       # Slightly transparent background
        edgecolor='lightgray',# Subtle edge
        handlelength=1.5,     # Keep shorter lines
        borderpad=0.5,        # Padding inside the box
    )
    
    # Set legend background color to white with some transparency
    legend.get_frame().set_facecolor('white')
    
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', f'token-generation-comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate token generation plots for benchmark results.")
    parser.add_argument("folder", help="Path to the folder containing results")
    args = parser.parse_args()
    
    plot_token_generation(args.folder)

if __name__ == "__main__":
    main()
