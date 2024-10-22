import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the JSON data
with open('results/refactorchat_sample.json', 'r') as f:
    data = json.load(f)

# Extract data for each approach
greedy_data = next(item for item in data if item['strategy'] == 'greedy')
sd_data = next(item for item in data if item['strategy'] == 'sd')
amusd_data = next(item for item in data if item['strategy'] == 'amusd')

# Get the total number of tokens from the greedy strategy
total_tokens = len(greedy_data['metrics'].get('token_times', []))

# Prepare cumulative data
def get_cumulative_tokens_times(strategy_data, strategy_name, total_tokens):
    cumulative_times = []
    tokens = []
    
    if strategy_name == 'greedy':
        token_times = strategy_data['metrics'].get('token_times', [])
        cumulative_time = 0
        for i, time in enumerate(token_times):
            cumulative_time += time
            cumulative_times.append(cumulative_time)
            tokens.append(i + 1)
        return cumulative_times, tokens
    
    elif strategy_name == 'sd':
        draft_times = strategy_data['metrics'].get('draft_times', [])
        verify_times = strategy_data['metrics'].get('verify_times', [])
        cumulative_time = 0
        tokens_emitted = 0
        cumulative_times = []
        tokens = []
        
        # Assume equal batch sizes
        num_verifications = len(verify_times)
        batch_size = total_tokens // num_verifications
        extra_tokens = total_tokens % num_verifications
        batch_sizes = [batch_size + (1 if i < extra_tokens else 0) for i in range(num_verifications)]
        
        idx = 0  # Index for draft times
        for i in range(len(verify_times)):
            # Add draft times for the batch
            batch_draft_times = draft_times[idx:idx+batch_sizes[i]]
            cumulative_time += sum(batch_draft_times)
            idx += batch_sizes[i]
            # Add verification time
            cumulative_time += verify_times[i]
            tokens_emitted += batch_sizes[i]
            cumulative_times.append(cumulative_time)
            tokens.append(tokens_emitted)
        return cumulative_times, tokens
    
    elif strategy_name == 'amusd':
        verify_times = strategy_data['metrics'].get('verify_times', [])
        cumulative_time = 0
        tokens_emitted = 0
        cumulative_times = []
        tokens = []
        
        num_verifications = len(verify_times)
        batch_size = total_tokens // num_verifications
        extra_tokens = total_tokens % num_verifications
        batch_sizes = [batch_size + (1 if i < extra_tokens else 0) for i in range(num_verifications)]
        
        for i in range(len(verify_times)):
            cumulative_time += verify_times[i]
            tokens_emitted += batch_sizes[i]
            cumulative_times.append(cumulative_time)
            tokens.append(tokens_emitted)
        return cumulative_times, tokens

# Get cumulative data for each strategy
greedy_times, greedy_tokens = get_cumulative_tokens_times(greedy_data, 'greedy', total_tokens)
sd_times, sd_tokens = get_cumulative_tokens_times(sd_data, 'sd', total_tokens)
amusd_times, amusd_tokens = get_cumulative_tokens_times(amusd_data, 'amusd', total_tokens)

# Update total_tokens to be the maximum tokens emitted among all strategies
max_tokens = max(greedy_tokens[-1], sd_tokens[-1], amusd_tokens[-1])

# Custom "old-school" styling with new colors
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
    'figure.figsize': [8, 4],
    'lines.linewidth': 2,
    'axes.prop_cycle': plt.cycler('color', ['#4682B4', '#32CD32', '#FF8C00']),  # Steel blue, lime green, dark orange
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'savefig.dpi': 400,
})

# Plot cumulative tokens over time using step plots
fig, ax = plt.subplots(figsize=(8, 4))

# Main lines
ax.step(greedy_times, greedy_tokens, where='post', linewidth=2, color='#4682B4')
ax.step(sd_times, sd_tokens, where='post', linewidth=2, color='#32CD32')
ax.step(amusd_times, amusd_tokens, where='post', linewidth=2, color='#FF8C00')

# Adding direct labels to each line with rotation
strategies = [
    {'times': greedy_times, 'tokens': greedy_tokens, 'label': 'Autoregressive', 'color': '#4682B4'},
    {'times': sd_times, 'tokens': sd_tokens, 'label': 'Speculative Decoding', 'color': '#32CD32'},
    {'times': amusd_times, 'tokens': amusd_tokens, 'label': 'AMUSD (ours)', 'color': '#FF8C00'},
]

# Adjust y-offsets and positions for labels
strategy_text_offsets = {
    "Autoregressive": (12.5, 580, 24),
    "Speculative Decoding": (7, 420, 28.5),
    "AMUSD (ours)": (3, 205, 37.5),
}

for strat in strategies:
    times = strat['times']
    tokens = strat['tokens']
    label = strat['label']
    color = strat['color']

    x, y, rotation = strategy_text_offsets[label]
    
    # Place the label slightly offset from the line
    ax.text(
        x, y, label,
        fontsize=10, fontweight='bold', color=color,
        rotation=rotation, rotation_mode='anchor',
        ha='left', va='bottom',
        backgroundcolor='white',
        bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.7)
    )

# Labels and title
ax.set_xlabel('Time (seconds)', fontweight='bold')
ax.set_ylabel('# of Verified Tokens', fontweight='bold')
ax.set_title('Verified Tokens Generated Over Time by Strategy', fontweight='bold')

# Grid
ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

# Set plot limits based on data
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Ensure labels are within plot area
plt.tight_layout()
# plt.subplots_adjust(right=0.95)  # Adjust if labels are cut off

# Save the plot
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/token-generation-comparison.pdf', dpi=400, bbox_inches='tight')
plt.show()


def get_average_accepted(strategy_data):
    accepted_tokens = strategy_data['metrics'].get('accepted_tokens', [])
    if accepted_tokens:
        return sum(accepted_tokens) / len(accepted_tokens)
    return 0

print("\nAverage accepted tokens per batch:")
print(f"Speculative Decoding: {get_average_accepted(sd_data):.2f}")
print(f"AMUSD: {get_average_accepted(amusd_data):.2f}")
