import json
import matplotlib.pyplot as plt
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

def plot_token_generation(folder_path: str):
    """Create token generation plot for a single sample"""
    set_plotting_style()
    
    # Create wider, shorter figure for double-column format
    fig, ax = plt.subplots(figsize=(7, 4))  # Adjusted to be wider and shorter
    
    for filename, label in MODEL_CONFIGS.items():
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {filename} - file not found")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data.get("results") or not data["results"]:
            continue
        
        sample = data["results"][1]
        is_greedy = 'greedy' in filename
        print(f"Processing {label} single sample")
        
        times, tokens = get_cumulative_tokens_times(sample['metrics'], is_greedy)
        if times and tokens:
            color = COLORS[label]
            ax.plot(times, tokens, color=color, label=label, linewidth=2.0)  # Increased line thickness
    
    style_axis(ax,
              xlabel='Time (seconds)',
              ylabel='Number of Verified Tokens',
              title='Token Generation Comparison')
    
    # Adjusted legend position for shorter plot
    legend = add_legend(ax,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),
        borderaxespad=0,
        borderpad=0.1,           # Reduced padding
        handlelength=1.0,
        handletextpad=0.3,       # Reduced spacing
        ncol=2                   # Use two columns to save vertical space
    )
    
    # Tighter layout with smaller margins
    plt.tight_layout(pad=0.2)
    
    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', f'token-generation-comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.02)  # Reduced padding
    plt.close()
    
    # Reset to default style after saving
    set_plotting_style()

def main():
    parser = argparse.ArgumentParser(description="Generate token generation plots for benchmark results.")
    parser.add_argument("folder", help="Path to the folder containing results")
    args = parser.parse_args()
    
    plot_token_generation(args.folder)

if __name__ == "__main__":
    main()
