import json
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import os

def process_gpu_data(data):
    strategies = ['Autoregressive', 'Speculative', 'AMUSD (ours)']
    results = {strategy: {'power': [], 'utilization': [], 'energy': [], 'time': []} for strategy in strategies}
    
    for item in data:
        strategy = item['strategy']
        if strategy == 'greedy':
            strategy = 'Autoregressive'
        elif strategy == 'sd':
            strategy = 'Speculative'
        elif strategy == 'amusd':
            strategy = 'AMUSD (ours)'
        
        gpustats = item['metrics']['gpustats']
        total_time = item['metrics']['total_time']  # Get the total time
        
        for stat in gpustats:
            results[strategy]['power'].append((stat['gpu_powers'][0], stat['gpu_powers'][1]))
            results[strategy]['utilization'].append((stat['gpu_utilizations'][0], stat['gpu_utilizations'][1]))
            # Calculate energy for each GPU (Power × Time)
            energy_gpu0 = stat['gpu_powers'][0] * total_time
            energy_gpu1 = stat['gpu_powers'][1] * total_time
            results[strategy]['energy'].append((energy_gpu0, energy_gpu1))
            results[strategy]['time'].append(total_time)
    
    # Calculate averages
    for strategy in strategies:
        results[strategy]['avg_power'] = (mean([p[0] for p in results[strategy]['power']]),
                                        mean([p[1] for p in results[strategy]['power']]))
        results[strategy]['avg_utilization'] = (mean([u[0] for u in results[strategy]['utilization']]),
                                              mean([u[1] for u in results[strategy]['utilization']]))
        results[strategy]['avg_energy'] = (mean([e[0] for e in results[strategy]['energy']]),
                                         mean([e[1] for e in results[strategy]['energy']]))
        results[strategy]['avg_time'] = mean(results[strategy]['time'])
    
    return results

def create_gpu_visualization(results):
    strategies = list(results.keys())
    n_groups = len(strategies)

    # Prepare data for plotting
    power_data = []
    utilization_data = []
    energy_data = []
    for strategy in strategies:
        power_data.extend(results[strategy]['avg_power'])
        utilization_data.extend(results[strategy]['avg_utilization'])
        energy_data.extend(results[strategy]['avg_energy'])

    plt.rcParams.update({
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.grid': True,
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'font.size': 12,
        'legend.fontsize': 10,
        'figure.figsize': [10, 7.5],  # Increased height for third subplot
        'lines.linewidth': 2,
        'hatch.linewidth': 1.5,
        'axes.prop_cycle': plt.cycler('color', ['#2F4F4F', '#8B0000', '#4682B4']),
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'savefig.dpi': 300,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.5))  # Added third subplot

    # Bar width and positions
    bar_width = 0.35
    r1 = np.arange(n_groups)
    r2 = [x + bar_width for x in r1]

    # Plot power data
    ax1.bar(r1, power_data[::2], color='#2F4F4F', edgecolor='black', width=bar_width, label='GPU0')
    ax1.bar(r2, power_data[1::2], color='#8B0000', edgecolor='black', width=bar_width, hatch='//', label='GPU1')

    # Plot utilization data
    ax2.bar(r1, utilization_data[::2], color='#2F4F4F', edgecolor='black', width=bar_width, label='GPU0')
    ax2.bar(r2, utilization_data[1::2], color='#8B0000', edgecolor='black', width=bar_width, hatch='//', label='GPU1')

    # Plot energy data
    ax3.bar(r1, energy_data[::2], color='#2F4F4F', edgecolor='black', width=bar_width, label='GPU0')
    ax3.bar(r2, energy_data[1::2], color='#8B0000', edgecolor='black', width=bar_width, hatch='//', label='GPU1')

    # Label settings
    ax1.set_ylabel('Power (W)', fontweight='bold')
    ax1.set_title('Average GPU Power per Strategy', fontweight='bold')
    ax2.set_ylabel('Utilization (%)', fontweight='bold')
    ax2.set_title('Average GPU Utilization per Strategy', fontweight='bold')
    ax3.set_ylabel('Energy (J)', fontweight='bold')
    ax3.set_title('Total Energy Consumption per Strategy', fontweight='bold')

    # Xticks and labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([r + bar_width / 2 for r in range(n_groups)])
        ax.set_xticklabels(strategies, fontweight='bold')
        ax.legend(loc='upper right', fontsize='medium')

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)

    # Save figure with high resolution
    plt.savefig('figures/gpu-comparison.pdf', dpi=400, bbox_inches='tight')
    plt.close()

# Load the data from the JSON file
with open('results/refactorchat_sample.json', 'r') as file:
    data = json.load(file)

# Process the data
results = process_gpu_data(data)

# Create and save the visualization
create_gpu_visualization(results)

print("Analysis complete. Visualization saved as 'figures/gpu-comparison.pdf'")

# Print results for verification
for strategy, stats in results.items():
    print(f"\n{strategy}:")
    print(f"  Avg Power: GPU0 = {stats['avg_power'][0]:.2f}W, GPU1 = {stats['avg_power'][1]:.2f}W")
    print(f"  Avg Utilization: GPU0 = {stats['avg_utilization'][0]:.2f}%, GPU1 = {stats['avg_utilization'][1]:.2f}%")
    print(f"  Avg Energy: GPU0 = {stats['avg_energy'][0]:.2f}J, GPU1 = {stats['avg_energy'][1]:.2f}J")
    print(f"  Avg Time: {stats['avg_time']:.2f}s")
