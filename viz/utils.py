import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

COLOR_SCHEMES = {
    'base_models': {
        'BL(70B)': '#1f77b4',  # Classic blue for baseline
    },
    'speculative': {
        'SD(1B,70B)':'#99d8c9',  # Light teal - distinct from medium 
        'SD(8B,70B)': '#66c2a5',     # Teal green - completely different tone
        'SD(1B,8B,70B)': '#2ca02c',     # Deep green
        'SD(1B,8B)': '#2ca02c',     # Deep green
    },
    'pipespec': {
        'PS(1B,70B)': '#ffd700',  # Gold yellow - made brighter and more distinct
        'PS(8B,70B)': '#ff7f0e',     # Vibrant orange
        'PS(1B,8B,70B)': '#d62728',     # Deep red
        'PS(1B,8B)': '#d62728',     # Deep red
    },
}

COLORS = {k: v for d in COLOR_SCHEMES.values() for k, v in d.items()}

FIGURE_SIZES = {
    'single_column': (5, 3.5),    # Standard single column
    'double_column': (7, 4),      # Standard double column
    'large': (10, 6),            # Large format
}

def set_plotting_style(style: str = 'default') -> None:
    """Set up the matplotlib style for academic/research plots"""
    plt.style.use('seaborn-v0_8-paper')
    
    # Base style settings with updated font settings
    base_settings = {
        # Font settings - using standard LaTeX-like fonts
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Palatino', 'Times'],
        'mathtext.fontset': 'stix',  # Use STIX fonts for math
        
        # Significantly increased font sizes
        'font.size': 14,              
        'axes.labelsize': 14,         
        'axes.titlesize': 16,         
        'xtick.labelsize': 13,        
        'ytick.labelsize': 13,        
        'legend.fontsize': 13,        
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # Grid settings
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Line settings
        'lines.linewidth': 2.0,
        
        # Spine settings
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Axes settings
        'axes.linewidth': 1.2,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        
        # Tick settings
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    }
    
    plt.rcParams.update(base_settings)
    
    if style == 'dark':
        dark_settings = {
            'axes.facecolor': '#2d3436',
            'figure.facecolor': '#2d3436',
            'grid.color': '#636e72',
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
        }
        plt.rcParams.update(dark_settings)


def get_figure_axes(size: str = 'single_column', **kwargs) -> tuple:
    """Create figure and axes with standard sizes"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZES[size], **kwargs)
    return fig, ax

def style_axis(ax: plt.Axes, title: str = None, xlabel: str = None, ylabel: str = None) -> None:
    """Apply consistent styling to axis"""
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

def add_legend(ax: plt.Axes, **kwargs) -> None:
    """Add styled legend to axis"""
    legend = ax.legend(frameon=True, 
                      fancybox=False, 
                      framealpha=0.9, 
                      edgecolor='black',
                      **kwargs)
    return legend

# Model configurations for easy reference
MODEL_CONFIGS = {
    'greedy_Meta-Llama-3.1-70B-Instruct.json': 'BL(70B)',
    'chain_Llama-3.2-1B-Instruct-Meta-Llama-3.1-70B-Instruct_8.json': 'SD(1B,70B)',
    'chain_Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct_8.json': 'SD(8B,70B)',
    'chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct_8.json': 'SD(1B,8B,70B)',
    'async-chain_Llama-3.2-1B-Instruct-Meta-Llama-3.1-70B-Instruct_0.json': 'PS(1B,70B)',
    'async-chain_Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct_0.json': 'PS(8B,70B)',
    'async-chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct_0.json': 'PS(1B,8B,70B)'
}
