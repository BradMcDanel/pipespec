import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse
from typing import Dict, List, Tuple, Set
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    set_plotting_style,
    style_axis,
    COLORS,
    MODEL_CONFIGS
)

def smooth_data(data: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Apply gentle Savitzky-Golay filter to smooth the data."""
    if len(data) < window:
        return data
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

def process_model_data(data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process data for a single model configuration."""
    active_gpus = get_active_gpus(data['metadata'])
    
    if not data.get('results', []):
        return None, None, None
    
    sample = data['results'][0]
    if not sample.get('gpustats', []):
        return None, None, None
    
    timestamps = np.array([stat['timestamp'] for stat in sample['gpustats']])
    timestamps -= timestamps[0]
    
    n_samples = len(sample['gpustats'])
    util_data = np.zeros((n_samples, len(active_gpus)))
    power_data = np.zeros_like(util_data)
    
    for i, stat in enumerate(sample['gpustats']):
        for j, gpu_idx in enumerate(sorted(active_gpus)):
            util_data[i, j] = stat['gpu_utilizations'][gpu_idx]
            power_data[i, j] = stat['gpu_powers'][gpu_idx]
    
    for j in range(util_data.shape[1]):
        util_data[:, j] = smooth_data(util_data[:, j])
        power_data[:, j] = smooth_data(power_data[:, j])
    
    return timestamps, util_data, power_data

def calculate_aggregate_metrics(utils: np.ndarray, powers: np.ndarray, times: np.ndarray, sample: Dict) -> Dict[str, float]:
    """Calculate aggregate metrics across all GPUs."""
    metrics = {}
    
    metrics['avg_util'] = np.mean(utils)
    metrics['avg_power'] = np.mean(powers)
    
    # Get tokens generated using the same logic as token_generation_comparison.py
    is_greedy = 'greedy' in sample.get('config', {}).get('name', '')
    if is_greedy:
        total_tokens = len(sample['metrics'].get('token_times', []))
    else:
        accepted_tokens = sample['metrics'].get('accepted_tokens', [])[-1] if sample['metrics'].get('accepted_tokens') else []
        total_tokens = sum(accepted_tokens)
    
    # Calculate total energy then divide by number of tokens
    total_energy = np.sum(np.trapz(powers, times, axis=0))
    metrics['avg_energy_per_token'] = total_energy / max(total_tokens, 1)  # Avoid division by zero
    
    return metrics

def format_aggregate_metrics_text(metrics: Dict[str, float]) -> str:
    """Format aggregate metrics into a readable string."""
    return (f"Avg Util: {metrics['avg_util']:.1f}%\t"
            f"Avg Power: {metrics['avg_power']:.1f}W\t"
            f"Avg Energy/Token: {metrics['avg_energy_per_token']:.1f}J")

def plot_gpu_metrics(folder_path: str):
    """Create three-row plot showing GPU utilization with metrics."""
    set_plotting_style()
    
    # Create figure with wider aspect ratio and additional space at bottom for legend
    fig = plt.figure(figsize=(7, 5.5))  # Slightly taller to accommodate legend
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    roles = {
        'Autoregressive': {
            0: 'L-3.1-70B (CUDA:2)',
            1: 'L-3.1-70B (CUDA:3)'
        },
        'Speculative Decoding': {
            0: 'L-3.2-1B (CUDA:0)',
            1: 'L-3.1-8B (CUDA:1)',
            2: 'L-3.1-70B (CUDA:2)',
            3: 'L-3.1-70B (CUDA:3)'
        },
        'PipeSpec': {
            0: 'L-3.2-1B (CUDA:0)',
            1: 'L-3.1-8B (CUDA:1)',
            2: 'L-3.1-70B (CUDA:2)',
            3: 'L-3.1-70B (CUDA:3)'
        }
    }
    methods = {
        'Autoregressive': ['*greedy*70B*.json'],
        'Speculative Decoding': ['*chain*1B*8B*70B*8.json'],
        'PipeSpec': ['*async-chain*1B*8B*70B*0.json']
    }
    
    gpu_colors = {
        0: '#FA8072',
        1: '#4F94CD',
        2: '#8B7BB5',
        3: '#DAA520',
    }
    
    color_mapping = {
        'Autoregressive': {
            0: gpu_colors[2],
            1: gpu_colors[3]
        }
    }
    
    max_time = 45
    axes = []
    all_handles = []
    all_labels = []
    

    for idx, (method, patterns) in enumerate(methods.items()):
        ax = fig.add_subplot(gs[idx, 0])
        axes.append(ax)
        
        if idx < len(methods) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        
        for pattern in patterns:
            matching_files = glob.glob(os.path.join(folder_path, pattern))
            
            for file_path in matching_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Get the sample data
                if not data.get("results"):
                    continue
                sample = data["results"][1]  # Use second sample like in token_generation_comparison

                times, utils, powers = process_model_data(data)
                if times is None:
                    continue
                
                mask = times <= max_time
                times = times[mask]
                utils = utils[mask]
                powers = powers[mask]
                
                active_gpus = get_active_gpus(data['metadata'])
                
                for gpu_idx, gpu_id in enumerate(sorted(active_gpus)):
                    if gpu_idx not in roles[method]:
                        continue
                    
                    label = roles[method][gpu_idx]
                    
                    if method in color_mapping and gpu_idx in color_mapping[method]:
                        color = color_mapping[method][gpu_idx]
                    else:
                        color = gpu_colors[gpu_idx]
                    
                    line = ax.plot(times, utils[:, gpu_idx],
                                 label=label,
                                 color=color,
                                 linestyle='-',
                                 linewidth=1.5)[0]
                    
                    if label not in all_labels:
                        all_handles.append(line)
                        all_labels.append(label)
                
                metrics = calculate_aggregate_metrics(utils, powers, times, sample)
                stats_text = f"Avg Util: {metrics['avg_util']:.1f}% | Avg Power: {metrics['avg_power']:.1f}W | Avg Energy/Token: {metrics['avg_energy_per_token']:.1f}J"
                
                ax.text(0.02, 0.95, stats_text,
                       transform=ax.transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       horizontalalignment='left')
        
        ax.set_ylim(-5, 120)
        ax.set_xlim(0, max_time)
        
        if idx == len(methods) - 1:
            ax.set_xlabel('Time (seconds)', fontsize=11)
        
        ax.set_ylabel('Utilization (%)', fontsize=11)
        ax.set_title(method, fontsize=12, pad=3)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        style_axis(ax)
    
    # Share x-axis between all plots
    for ax in axes[1:]:
        ax.sharex(axes[0])
    
    # Add legend below the plots
    fig.legend(all_handles, all_labels,
              loc='center',
              ncol=4,
              fontsize=9,
              frameon=True,
              framealpha=0.9,
              bbox_to_anchor=(0.5, 0.02),  # Position at bottom
              columnspacing=1.0,
              handlelength=1.0,
              handletextpad=0.3)

    plt.tight_layout()
    # Adjust subplot spacing to make room for legend
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    
    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', 'gpu-usage-analysis')
    plt.savefig(f'{output_path}.pdf', 
                bbox_inches='tight', 
                pad_inches=0.02,
                dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate GPU usage analysis plots.")
    parser.add_argument("folder", help="Path to the folder containing results")
    args = parser.parse_args()
    plot_gpu_metrics(args.folder)

if __name__ == "__main__":
    main()
