import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse
from typing import Dict, List, Tuple, Set
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    set_plotting_style,
    style_axis,
    COLORS,
    MODEL_CONFIGS
)

def smooth_data(data: np.ndarray, window: int = 51, poly: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth the data."""
    return savgol_filter(data, window, poly, axis=0)

def get_active_gpus(metadata: Dict) -> Set[int]:
    """Extract active GPU indices from metadata."""
    active_gpus = set()
    for config in metadata.get('model_configs', []):
        for device in config.get('devices', []):
            if device.startswith('cuda:'):
                gpu_idx = int(device.split(':')[1])
                active_gpus.add(gpu_idx)
    return active_gpus

def process_model_data(data: Dict, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process data for a single model configuration."""
    active_gpus = get_active_gpus(data['metadata'])
    
    if not data.get('results', []):
        return None, None, None
        
    # Process first sample for demonstration
    sample = data['results'][0]
    if not sample.get('gpustats', []):
        return None, None, None
    
    # Convert timestamps to relative time
    timestamps = np.array([stat['timestamp'] for stat in sample['gpustats']])
    timestamps -= timestamps[0]
    
    n_samples = len(sample['gpustats'])
    util_data = np.zeros((n_samples, len(active_gpus)))
    power_data = np.zeros_like(util_data)
    
    # Fill arrays
    for i, stat in enumerate(sample['gpustats']):
        for j, gpu_idx in enumerate(sorted(active_gpus)):
            util_data[i, j] = stat['gpu_utilizations'][gpu_idx]
            power_data[i, j] = stat['gpu_powers'][gpu_idx]
    
    # Apply smoothing - adapt window size based on data length
    window = min(51, len(timestamps) - (len(timestamps) % 2) - 1)
    if window > 3:  # Need at least 4 points for smoothing
        util_data = smooth_data(util_data, window=window)
        power_data = smooth_data(power_data, window=window)
    
    return timestamps, util_data, power_data

def plot_gpu_metrics(folder_path: str):
    """Create separate plots for each method showing GPU utilization and power usage."""
    set_plotting_style()
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    max_time = 60
    
    # Select only baseline and 3-tier models
    methods = {
        'Baseline': ['greedy_Meta-Llama-3.1-70B-Instruct.json'],
        'Speculative': ['chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct_5.json'],
        'PipeSpec': ['async-chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct_0.json']
    }
    
    # Define line styles for different roles
    role_styles = {
        0: {'linestyle': '-', 'label': 'Small Draft'},    # 1B
        1: {'linestyle': '--', 'label': 'Medium Draft'},  # 8B
        2: {'linestyle': ':', 'label': 'Target'},         # 70B
        3: {'linestyle': '-.', 'label': 'Target (2)'}     # 70B split
    }
    
    for row, (method, filenames) in enumerate(methods.items()):
        ax_util = fig.add_subplot(gs[row, 0])
        ax_power = fig.add_subplot(gs[row, 1])
        
        ax_util.set_ylim(0, 100)
        ax_power.set_ylim(0, 300)
        
        for filename in filenames:
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                continue
                
            model_name = MODEL_CONFIGS[filename]
            print(f"Processing {model_name}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            times, utils, powers = process_model_data(data)
            if times is None:
                continue
                
            # Limit time range
            mask = times <= max_time
            times = times[mask]
            utils = utils[mask]
            powers = powers[mask]
            
            color = COLORS[model_name]
            active_gpus = get_active_gpus(data['metadata'])
            
            for gpu_idx, gpu_id in enumerate(sorted(active_gpus)):
                style = role_styles[gpu_idx]
                label = f"{model_name} ({style['label']})"
                
                ax_util.plot(times, utils[:, gpu_idx],
                           label=label, color=color,
                           linestyle=style['linestyle'],
                           linewidth=2)
                           
                ax_power.plot(times, powers[:, gpu_idx],
                            label=label, color=color,
                            linestyle=style['linestyle'],
                            linewidth=2)
        
        # Style plots
        title_suffix = f" - {method}"
        style_axis(ax_util, 
                  title='GPU Utilization' + title_suffix if row == 0 else '',
                  ylabel='Utilization (%)')
        style_axis(ax_power,
                  title='GPU Power Usage' + title_suffix if row == 0 else '',
                  ylabel='Power (W)')
        
        # Set x-axis limits and labels
        ax_util.set_xlim(0, max_time)
        ax_power.set_xlim(0, max_time)
        
        if row == 2:
            ax_util.set_xlabel('Time (seconds)')
            ax_power.set_xlabel('Time (seconds)')
        
        # Add legends inside plots
        ax_util.legend(loc='upper right', 
                      ncol=1, 
                      fontsize=8, 
                      framealpha=0.9)
        ax_power.legend(loc='upper right', 
                       ncol=1, 
                       fontsize=8, 
                       framealpha=0.9)
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', 'gpu-usage-analysis')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate GPU usage analysis plots.")
    parser.add_argument("folder", help="Path to the folder containing results")
    
    args = parser.parse_args()
    plot_gpu_metrics(args.folder)

if __name__ == "__main__":
    main()
