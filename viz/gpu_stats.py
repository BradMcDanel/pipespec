import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple, Set
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    set_plotting_style,
    style_axis,
    add_legend,
    COLORS,
    MODEL_CONFIGS
)

def get_active_gpus(metadata: Dict) -> Tuple[Set[int], Dict[int, str]]:
    """
    Extract active GPU indices and their roles from metadata.
    Returns both the active GPUs and their roles (draft/target).
    """
    active_gpus = set()
    gpu_roles = {}
    
    for idx, config in enumerate(metadata.get('model_configs', [])):
        model_name = config.get('model_name', '')
        for device in config.get('devices', []):
            if device.startswith('cuda:'):
                gpu_idx = int(device.split(':')[1])
                active_gpus.add(gpu_idx)
                
                # Assign roles based on model position and size
                if 'Llama-3.2-1B' in model_name:
                    role = 'Small Draft'
                elif 'Llama-3.1-8B' in model_name:
                    role = 'Medium Draft'
                elif 'Llama-3.1-70B' in model_name:
                    role = 'Target'
                else:
                    role = f'GPU {gpu_idx}'
                gpu_roles[gpu_idx] = role
                
    return active_gpus, gpu_roles

def analyze_gpu_metrics(file_path: str) -> Dict:
    """Analyze GPU metrics with enhanced statistics."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    active_gpus, gpu_roles = get_active_gpus(data['metadata'])
    
    metrics = {
        'power': {gpu_idx: [] for gpu_idx in active_gpus},
        'utilization': {gpu_idx: [] for gpu_idx in active_gpus},
        'energy': {gpu_idx: [] for gpu_idx in active_gpus},
        'throughput': []  # tokens/second
    }
    
    for result in data.get('results', []):
        metrics_data = result.get('metrics', {})
        total_time = metrics_data.get('total_time', 0)
        tokens_generated = metrics_data.get('tokens_generated', 0)
        gpustats = metrics_data.get('gpustats', [])
        
        # Calculate throughput
        if total_time > 0:
            metrics['throughput'].append(tokens_generated / total_time)
        
        for stat in gpustats:
            for gpu_idx in active_gpus:
                metrics['power'][gpu_idx].append(stat['gpu_powers'][gpu_idx])
                metrics['utilization'][gpu_idx].append(stat['gpu_utilizations'][gpu_idx])
                metrics['energy'][gpu_idx].append(stat['gpu_powers'][gpu_idx] * total_time)
    
    # Calculate comprehensive statistics
    stats = {
        'roles': gpu_roles,
        'avg_throughput': np.mean(metrics['throughput']) if metrics['throughput'] else 0,
        'total_energy': sum(np.mean(metrics['energy'][gpu]) for gpu in active_gpus),
        'per_gpu': {}
    }
    
    for gpu_idx in active_gpus:
        stats['per_gpu'][gpu_idx] = {
            'avg_power': np.mean(metrics['power'][gpu_idx]),
            'peak_power': np.max(metrics['power'][gpu_idx]),
            'avg_util': np.mean(metrics['utilization'][gpu_idx]),
            'energy': np.mean(metrics['energy'][gpu_idx])
        }
    
    return stats

def plot_gpu_analysis(folder_path: str):
    """Create an enhanced visualization focusing on efficiency metrics."""
    set_plotting_style()
    
    # Create figure with two rows: top for power/utilization, bottom for efficiency metrics
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
    
    ax_power = fig.add_subplot(gs[0, 0])
    ax_util = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 0])
    ax_throughput = fig.add_subplot(gs[1, 1])
    
    methods = {
        'Baseline': 'greedy_Meta-Llama-3.1-70B-Instruct.json',
        'Speculative': 'chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct_5.json',
        'PipeSpec': 'async-chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct_0.json'
    }
    
    # Color scheme for different GPU roles
    role_colors = {
        'Small Draft': '#2F4F4F',
        'Medium Draft': '#8B0000',
        'Target': '#4682B4'
    }
    
    x = np.arange(len(methods))
    width = 0.25
    
    method_stats = {}
    for method_name, filename in methods.items():
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            method_stats[method_name] = analyze_gpu_metrics(file_path)
    
    # Plot all metrics
    for idx, method_name in enumerate(methods):
        if method_name not in method_stats:
            continue
            
        stats = method_stats[method_name]
        
        # Power and Utilization plots (stacked by role)
        for gpu_idx, role in stats['roles'].items():
            gpu_data = stats['per_gpu'][gpu_idx]
            color = role_colors.get(role, '#808080')
            
            ax_power.bar(idx, gpu_data['avg_power'], width,
                        bottom=sum(method_stats[method_name]['per_gpu'][g]['avg_power'] 
                                 for g in stats['roles'] if g < gpu_idx),
                        color=color, label=role if idx == 0 else "")
                        
            ax_util.bar(idx, gpu_data['avg_util'], width,
                       bottom=sum(method_stats[method_name]['per_gpu'][g]['avg_util']
                                for g in stats['roles'] if g < gpu_idx),
                       color=color, label=role if idx == 0 else "")
        
        # Energy plot (total)
        ax_energy.bar(idx, stats['total_energy'], width, color=COLORS.get(method_name, '#808080'))
        
        # Throughput plot
        ax_throughput.bar(idx, stats['avg_throughput'], width, color=COLORS.get(method_name, '#808080'))
    
    # Style plots
    for ax, (title, ylabel) in zip(
        [ax_power, ax_util, ax_energy, ax_throughput],
        [('GPU Power Usage', 'Average Power (W)'),
         ('GPU Utilization', 'Utilization (%)'),
         ('Total Energy Consumption', 'Energy (J)'),
         ('Generation Throughput', 'Tokens/second')]
    ):
        style_axis(ax, title=title, ylabel=ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(methods.keys(), rotation=45)
    
    # Add legends
    add_legend(ax_power, loc='upper right')
    add_legend(ax_util, loc='upper right')
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/gpu-analysis.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced GPU analysis plots.")
    parser.add_argument("folder", help="Path to the folder containing results")
    args = parser.parse_args()
    plot_gpu_analysis(args.folder)

if __name__ == "__main__":
    main()
