import json
from pathlib import Path

MODELS = {
    "68m": "llama-68m",
    "1b": "Llama-3.2-1B-Instruct",
    "7b": "Llama-2-7b-hf",
    "8b": "Llama-3.1-8B-Instruct",
    "13b": "Llama-2-13b-hf",
    "70b": "Meta-Llama-3.1-70B-Instruct"
}

def make_path(decoding, models):
    prefix = {"base": "greedy", "spec": "chain", "pipe": "async-chain"}[decoding]
    suffix = {"base": "", "spec": "_8", "pipe": "_0"}[decoding]
    return f"{prefix}_" + "-".join(MODELS[m] for m in models) + f"{suffix}.json"

# [dataset, base_model, folder_path, [(method, display_models, decoding, model_list[, speedup, citation])]]
EXPERIMENTS = [
    ["CNN/DM", "7b", "cnn_dailymail", [
        ("Autoregressive", "LLaMA2-7B", "base", ["7b"]),
        ("Speculative", "68M, 7B", "spec", ["68m", "7b"]),
        ("LayerSkip", "LLaMA2-7B", None, None, 1.86, "elhoushi2024layer"),
        ("PipeSpec", "68M, 7B", "pipe", ["68m", "7b"])
    ]],
    ["CNN/DM", "13b", "cnn_dailymail", [
        ("Autoregressive", "LLaMA2-13B", "base", ["13b"]),
        # ("Speculative", "68M, 13B", "spec", ["68m", "13b"]),
        ("Speculative", "68M, 7B, 13B", "spec", ["68m", "7b", "13b"]),
        ("Draft\\&Verify", "LLaMA2-13B", None, None, 1.56, "zhang2023draft"),
        ("LayerSkip", "LLaMA2-13B", None, None, 1.81, "elhoushi2024layer"),
        ("PipeSpec", "68M, 13B", "pipe", ["68m", "13b"]),
        ("PipeSpec", "7B, 13B", "pipe", ["7b", "13b"]),
        ("PipeSpec", "68M, 7B,13B", "pipe", ["68m", "7b", "13b"])
    ]],
    ["XSum", "7b", "xsum", [
        ("Autoregressive", "LLaMA2-7B", "base", ["7b"]),
        ("Speculative", "68M, 7B", "spec", ["68m", "7b"]),
        ("LayerSkip", "LLaMA2-7B", None, None, 1.54, "elhoushi2024layer"),
        ("PipeSpec", "68M, 7B", "pipe", ["68m", "7b"])
    ]],
    ["XSum", "13b", "xsum", [
        ("Autoregressive", "LLaMA2-13B", "base", ["13b"]),
        ("Speculative", "68M, 13B", "spec", ["68m", "13b"]),
        ("Speculative", "68M, 7B, 13B", "spec", ["68m", "7b", "13b"]),
        ("Draft\\&Verify", "LLaMA2-13B", None, None, 1.43, "zhang2023draft"),
        ("LayerSkip", "LLaMA2-13B", None, None, 1.48, "elhoushi2024layer"),
        ("PipeSpec", "68M, 13B", "pipe", ["68m", "13b"]),
        ("PipeSpec", "7B, 13B", "pipe", ["7b", "13b"]),
        ("PipeSpec", "68M, 7B, 13B", "pipe", ["68m", "7b", "13b"])
    ]],
    ["HumanEval", "13b", "human_eval", [
        ("Autoregressive", "LLaMA2-13B", "base", ["13b"]),
        ("Speculative", "68M, 13B", "spec", ["68m", "13b"]),
        ("Speculative", "68M, 7B, 13B", "spec", ["68m", "7b", "13b"]),
        ("Draft\\&Verify", "CodeLLaMA2-13B", None, None, 1.46, "zhang2023draft"),
        ("LayerSkip", "LLaMA2-13B", None, None, 1.66, "elhoushi2024layer"),
        ("PipeSpec", "68M, 13B", "pipe", ["68m", "13b"]),
        ("PipeSpec", "7B, 13B", "pipe", ["7b", "13b"]),
        ("PipeSpec", "68M, 7B, 13B", "pipe", ["68m", "7b", "13b"])
    ]],
    ["HumanEval", "70b", "human_eval", [
        ("Autoregressive", "LLaMA3.1-70B", "base", ["70b"]),
        ("Speculative", "8B, 70B", "spec", ["8b", "70b"]),
        ("Speculative", "1B, 8B, 70B", "spec", ["1b", "8b", "70b"]),
        ("PipeSpec", "1B, 70B", "pipe", ["1b", "70b"]),
        ("PipeSpec", "8B, 70B", "pipe", ["8b", "70b"]),
        ("PipeSpec", "1B, 8B, 70B", "pipe", ["1b", "8b", "70b"])
    ]],
    ["GSM8K", "13b", "gsm8k", [
        ("Autoregressive", "LLaMA2-13B", "base", ["13b"]),
        ("PipeSpec", "68M, 7B, 13B", "pipe", ["68m", "7b", "13b"])
    ]],
    ["MMLU", "13b", "mmlu", [
        ("Autoregressive", "LLaMA2-13B", "base", ["13b"]),
        ("PipeSpec", "68M, 7B, 13B", "pipe", ["68m", "7b", "13b"])
    ]]
]

def get_time_per_token(file_path):
    data = json.load(open(file_path))
    total_time = sum(s["metrics"]["total_time"] for s in data["results"])
    total_tokens = sum(s["metrics"]["tokens_generated"] for s in data["results"])
    return (total_time / total_tokens * 1000) if total_tokens else 0

def generate_tables(results_dir):
    # Plain text table
    print("\nResults Table")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Method':<15} {'Models':<20} {'Time(ms)':<10} {'Speedup'}")
    print("-" * 80)

    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Performance and efficiency across decoding strategies.}",
        "\\label{tab:results}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{threeparttable}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{c|l|l rr}",
        "\\toprule",
        "Dataset & Method & Models & \\makecell{Time\\\\(ms/tok)} & Speedup \\\\"
    ]

    # Calculate row counts for each dataset
    dataset_rows = {}
    current_dataset = None
    
    for dataset, _, _, experiments in EXPERIMENTS:
        if dataset not in dataset_rows:
            dataset_rows[dataset] = len(experiments)
        else:
            dataset_rows[dataset] += len(experiments)

    prev_dataset = None
    first_row_of_dataset = True
    first_group = True
    
    for dataset, model, folder, experiments in EXPERIMENTS:
        results = []
        baseline_time = None

        for exp in experiments:
            method, display_models = exp[0], exp[1]
            if method == "Speculative" or method == "PipeSpec":
                display_models = "\\{" + display_models + "\\}"
            speedup = exp[4] if len(exp) > 4 else None
            citation = exp[5] if len(exp) > 5 else None
            
            if len(exp) == 4:  # Computed result
                try:
                    json_path = make_path(exp[2], exp[3])
                    full_path = Path(results_dir) / folder / json_path
                    time = get_time_per_token(full_path)
                    if method == "Autoregressive":
                        baseline_time = time
                    speedup = baseline_time / time if baseline_time else 1.0
                    results.append((method, display_models, time, speedup, citation))
                except Exception as e:
                    print(f"Warning: Could not process {full_path}: {e}")
                    continue
            else:  # Reference result
                results.append((method, display_models, None, speedup, citation))

        # Find best speedup first
        best_speedup = max((r[3] for r in results if r[3] is not None), default=None)

        # Add appropriate separator
        if not first_group:
            if dataset != prev_dataset:
                latex.append("\\midrule[\\heavyrulewidth]")
                first_row_of_dataset = True
            else:
                latex.append("\\cmidrule{2-5}")
                first_row_of_dataset = False
        else:
            latex.append("\\midrule")
            first_group = False

        # Print results
        for i, (method, models, time, speedup, citation) in enumerate(results):
            # Plain text row
            time_str = f"{time:.2f}" if time is not None else "-"
            speedup_str = f"{speedup:.2f}" if speedup is not None else "-"
            print(f"{dataset:<12} {method:<15} {models:<20} {time_str:<10} {speedup_str}x")

            # LaTeX row formatting
            time_str = f"{time:.2f}" if time is not None else "--"
            speedup_str = f"{speedup:.2f}" if speedup is not None else "--"
            
            is_best = abs(speedup - best_speedup) < 0.01 if speedup and best_speedup else False
            if is_best:
                time_str = f"\\best{{{time_str}}}"
                speedup_str = f"\\best{{{speedup_str}}}"

            method_str = f"{method} (ours)" if method == "PipeSpec" else method
            if citation:
                method_str += f"\\cite{{{citation}}}"
            
            # Add dataset name only for first row of the entire dataset section
            if first_row_of_dataset and i == 0:
                dataset_str = f"\\multirow{{{dataset_rows[dataset]}}}{{*}}{{\\centering {dataset}}}"
                first_row_of_dataset = False
            else:
                dataset_str = ""

            # For non-first rows of a section, add leading space in place of dataset
            prefix = " " if not dataset_str else ""
            
            row = f"{prefix}{dataset_str} & {method_str} & {models} & {time_str} & {speedup_str}$\\times$ \\\\"
            latex.append(row)
        
        prev_dataset = dataset

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{threeparttable}",
        "\\end{table}"
    ])

    print("\nLaTeX Table:")
    print("\n".join(latex))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <results_dir>")
        sys.exit(1)
    generate_tables(sys.argv[1])
