# PipeSpec: Breaking Stage Dependencies in Hierarchical LLM Decoding

This repository contains an implementation of a novel hierarchical decoding strategy for large language models (LLMs) called **PipeSpec**. PipeSpec breaks the stage dependencies inherent in traditional sequential decoding, enabling parallelism and improved throughput. The codebase includes modules for various decoding strategies, GPU monitoring, benchmarking, result analysis, and visualization.

---

## Repository Structure

- **`benchmark.py`**  
  Benchmarking script for evaluating decoding strategies.
  
- **`decoding.py`**  
  Implements different decoding strategies:
  - `GreedyDecoder`
  - `ChainSpeculativeDecoder`
  - `AsyncChainSpeculativeDecoder`

- **`gpu_monitor.py`**  
  Utility to monitor GPU statistics (utilization, memory, power) during generation.

- **`print_benchmark_stats.py`**  
  Analyzes benchmark results and prints a summary table.

- **`run_example.py`**  
  Runs a sample generation using a chosen decoding strategy and dataset.

- **`viz/`**  
  Contains various visualization scripts:
  - `frequency_accepted_tokens.py`
  - `gpu_stats.py`
  - `gpu_usage_analysis.py`
  - `token_analysis.py`
  - `token_generation_comparison.py`
  - `utils.py` (plotting utilities and configurations)

---

## Installation

Clone the repository and navigate into the directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install the necessary Python packages (e.g., PyTorch, Transformers, Datasets, tqdm, pynvml, Matplotlib, NumPy, SciPy).

---

## Usage

### 1. Benchmarking

The main benchmarking script (`benchmark.py`) evaluates the different decoding strategies over a dataset.

**Command-line Arguments:**

- `--strategy`: Decoding strategy to benchmark (`greedy`, `chain`, or `async-chain`).
- `--dataset`: Name or path of the dataset to use.
- `--models-config-path`: Path to a JSON file containing model configuration details.
- `--num-samples`: (Optional) Number of samples to process.
- `--lookahead`: (Optional) Number of tokens to look ahead for speculative decoding (default is 5).
- `--output-dir`: (Optional) Directory where the benchmark results will be saved.
- `--force`: (Optional) Overwrite the output file if it already exists.

**Example:**

```bash
python benchmark.py \
  --strategy chain \
  --dataset MyDataset \
  --models-config-path configs/models.json \
  --num-samples 100 \
  --lookahead 5
```

### 2. Running a Generation Example

Use `run_example.py` to run a sample generation using a selected decoding strategy.

**Command-line Arguments:**

- `--strategy`: Decoding strategy (`greedy`, `chain`, or `async-chain`).
- `--dataset`: Name or path of the dataset.
- `--sample-index`: Index of the sample to use from the dataset.
- `--models-config-path`: Path to the JSON file with model configurations.
- `--lookahead`: (Optional) Lookahead value for speculative decoding.
- `--prefill`: (Optional) Number of tokens to prefill (for custom prompt datasets).
- `--use-basic-template`: (Optional) Use a simple USER/ASSISTANT template instead of the model’s default.

**Example:**

```bash
python run_example.py \
  --strategy chain \
  --dataset MyDataset \
  --sample-index 0 \
  --models-config-path configs/models.json \
  --lookahead 5
```

### 3. Analyzing Benchmark Statistics

After running benchmarks, you can analyze and print the results using:

```bash
python print_benchmark_stats.py <RESULTS_DIRECTORY>
```

Replace `<RESULTS_DIRECTORY>` with the path to the folder containing your benchmark result JSON files.

### 4. Visualizing Results

The `viz/` folder contains scripts to generate various plots and analyses from the benchmark results. Below are a few examples:

- **Frequency of Accepted Tokens:**

  ```bash
  python viz/frequency_accepted_tokens.py <RESULTS_FOLDER>
  ```

- **GPU Statistics Analysis:**

  ```bash
  python viz/gpu_stats.py <RESULTS_FOLDER>
  ```

- **GPU Usage Analysis:**

  ```bash
  python viz/gpu_usage_analysis.py <RESULTS_FOLDER>
  ```

- **Token Analysis:**

  ```bash
  python viz/token_analysis.py <RESULTS_FOLDER>
  ```

- **Token Generation Comparison:**

  ```bash
  python viz/token_generation_comparison.py <RESULTS_FOLDER>
  ```

Replace `<RESULTS_FOLDER>` with the directory where your benchmark JSON files are stored.

---

## Model Configuration File

The model configuration JSON should be an array of configurations. Each configuration must include:

- **`path`**: Path or identifier for the model checkpoint.
- **`devices`**: Device or list of devices (e.g., `"cuda:0"` or `["cuda:0", "cuda:1"]`).
- **`dtype`**: Data type (e.g., `"float16"` or `"bfloat16"`).
- **`quantize`** (Optional): Quantization type (e.g., `"4bit"` or `"8bit"`).

**Example `configs/models.json`:**

```json
[
  {
    "path": "path/to/model1",
    "devices": "cuda:0",
    "dtype": "float16"
  },
  {
    "path": "path/to/model2",
    "devices": ["cuda:1", "cuda:2"],
    "dtype": "bfloat16",
    "quantize": "4bit"
  }
]
```

---

## GPU Monitoring

The `GPUMonitor` class (in `gpu_monitor.py`) continuously monitors GPU utilization, memory usage, and power consumption during generation. This information is automatically captured and attached to the benchmark results.

---

## Custom Prompt Datasets

For datasets that require a custom prompt format (such as certain book or narrative datasets), the code provides specialized handling. Supported dataset names include `cnn_dailymail`, `pg19`, `narrativeqa`, and `one-shot`. When using these datasets, the appropriate prompt template is applied in `run_example.py`.

---

## Notes

- The code is designed for multi-GPU setups. Ensure that your CUDA environment is properly configured.
- For multi-GPU inference, consider using libraries such as `accelerate` if needed.
- The asynchronous decoding strategy (`async-chain`) uses Python’s multiprocessing. If you encounter issues, you may need to adjust the multiprocessing start method.
- Visualization scripts require benchmark result JSON files. Run the benchmark scripts first to generate these files.

