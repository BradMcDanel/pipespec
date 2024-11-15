import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    set_plotting_style,
    get_figure_axes,
    style_axis,
    add_legend,
    COLORS,
    MODEL_CONFIGS
)


def analyze_token_frequencies(dir_path, file_path, x_max, tri_model=False, tri_model_acc_list=1):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, '..', dir_path, file_path)

    with open(target_path, 'r') as file:
        json_data = file.read()

    # Initialize list of length x_max + 2 (indices 0 to x_max + 1)
    token_frequencies = [0] * (x_max + 1)
    total_count = 0

    # Parse the JSON data
    data = json.loads(json_data)

    # Iterate through all results
    for result in data['results']:
        # Access the accepted_tokens arrays
        acc_tokens = result['metrics']['accepted_tokens'][tri_model_acc_list] if tri_model else \
        result['metrics']['accepted_tokens'][0]

        for token in acc_tokens:
            # If token count would exceed x_max, cap it at x_max + 1
            if token <= x_max:
                token_frequencies[token-1] = token_frequencies[token-1] + 1
            else:
                token_frequencies[x_max] = token_frequencies[x_max] + 1
            total_count += 1

    normalized_frequencies = [round(freq / total_count, 2) if total_count > 0 else 0.00
                            for freq in token_frequencies]

    return normalized_frequencies


def make_figure(pic, dir_path="benchmark_results/humaneval"):
    set_plotting_style()
    fig, ax = get_figure_axes('double_column')

    pic_name = pic[0]["pic_name"]
    x_max = pic[0]["x_max"]
    tri_list = pic[0]["tri_model_acc_list"]

    tests = []
    legends = []

    for i in range(1, len(pic)):
        filename = pic[i]["filename"]
        test_data = analyze_token_frequencies(
            dir_path=dir_path,
            file_path=filename,
            x_max=x_max,
            tri_model=(i >= len(pic) - 2),
            tri_model_acc_list=tri_list
        )
        tests.append(test_data)
        legend_name = MODEL_CONFIGS[filename]
        legends.append(legend_name)

    x_labels = [str(i + 1) for i in range(x_max)]
    x_labels.append(f'{x_max}+')
    x = np.arange(len(x_labels))

    bar_width = 0.2
    for i, test_data in enumerate(tests):
        offset = bar_width * (i - len(tests) / 2 + 0.5)
        ax.bar(x + offset, test_data, bar_width,
               label=legends[i],
               color=COLORS[legends[i]],
               alpha=0.8)

    style_axis(ax,
               title=pic_name,
               xlabel='Number of Accepted Tokens',
               ylabel='Frequency (%)')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    # Simplified legend call
    add_legend(ax,
               loc='upper right',
               bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    return fig, ax


def save_figure(fig, name):
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Save as PDF
    pdf_path = os.path.join('figures', f'{name}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1)

    # Save as PNG with high DPI
    png_path = os.path.join('figures', f'{name}.png')
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.1, dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate token frequency visualization')
    parser.add_argument('--save', action='store_true',
                      help='Save the figure to PDF and PNG files')
    parser.add_argument('--show', action='store_true',
                      help='Display the figure using plt.show()')
    return parser.parse_args()


pic1 = [{"pic_name": "Accepted tokens Frequency(8B-70B)", "tri_model_acc_list": 1, "x_max": 10},
        {"filename": "chain_Llama-3.2-1B-Instruct-Meta-Llama-3.1-70B-Instruct.json"},
        {"filename": "async-chain_Llama-3.2-1B-Instruct-Meta-Llama-3.1-70B-Instruct.json"},
        {"filename": "chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct.json"},
        {"filename": "async-chain_Llama-3.2-1B-Instruct-Llama-3.1-8B-Instruct-Meta-Llama-3.1-70B-Instruct.json"}
        ]

if __name__ == "__main__":
    args = parse_args()
    fig, ax = make_figure(pic1)

    if args.save:
        save_figure(fig, args.output_name)

    if args.show:
        plt.show()


