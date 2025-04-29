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
    COLORS
)

def get_completion_time(file_path: str) -> float:
    """Calculate average completion time from results file in milliseconds."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    total_time = 0
    total_tokens = 0
    
    for result in data.get('results', []):
        metrics = result.get('metrics', {})
        if 'model_times' in metrics:
            total_time += metrics["total_time"]
            total_tokens += metrics["tokens_generated"]
    
    return (total_time / total_tokens) * 1000  # Convert to milliseconds

def analyze_lookahead_times(folder_path: str):
    """Create plot comparing lookahead values for chain and async-chain models."""
    set_plotting_style()
    
    # Create wider, shorter figure for double-column format
    fig, ax = plt.subplots(figsize=(7, 2.5))  # Adjusted to be wider and shorter
    
    # Fixed base names
    chain_base = 'chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct'
    async_chain_base = 'async-chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct'
    
    # Initialize data structures
    model_data = {
        chain_base: {},
        async_chain_base: {}
    }
    
    # Collect data for lookahead values 0-19
    for lookahead in range(20):
        for base in model_data:
            file_path = os.path.join(folder_path, f"{base}_{lookahead}.json")
            if os.path.exists(file_path):
                model_data[base][lookahead] = get_completion_time(file_path)
    
    # Plot data with updated styling
    labels = {
        chain_base: 'SD {1B,8B}',
        async_chain_base: 'PS {1B,8B}'
    }
    markers = ['o', 's']
    colors = [COLORS['SD {1B,8B}'], COLORS['PS {1B,8B}']]
    
    for idx, (base_name, lookahead_times) in enumerate(model_data.items()):
        lookaheads = sorted(lookahead_times.keys())
        times = [lookahead_times[k] for k in lookaheads]
        
        ax.plot(lookaheads, times,
                marker=markers[idx],
                label=labels[base_name],
                color=colors[idx],
                linewidth=2.0,
                markersize=6,
                markeredgewidth=1.5)
    
    style_axis(ax,
              title='Token Generation Time vs. Draft Lookahead Size',
              xlabel='Lookahead Size',
              ylabel='Avg. Token Time (ms)')
    
    # Adjusted legend positioning for shorter plot
    legend = add_legend(ax,
        loc='upper right',
        borderaxespad=0.1,
        borderpad=0.2,
        handlelength=1.0,
        handletextpad=0.3
    )
    
    # Tighter layout with smaller margins
    plt.tight_layout(pad=0.2)
    ax.set_xticks([0, 5, 10, 15, 20])
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/lookahead-analysis.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    # Reset to default style
    set_plotting_style()

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot lookahead times.")
    parser.add_argument("folder", help="Path to the folder containing results")
    args = parser.parse_args()
    
    analyze_lookahead_times(args.folder)

if __name__ == "__main__":
    main()
