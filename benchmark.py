import argparse
import json
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional
import os
mp.set_start_method('spawn', force=True)
import decoding
from gpu_monitor import GPUMonitor
import gc

MAX_NEW_TOKENS = 4096
STRATEGIES = ['greedy', 'chain', 'async-chain']

def create_output_filename(strategy: str, models_config: List[dict], output_dir: str) -> str:
    """Create a descriptive filename based on strategy and model configs"""
    # Extract model names from paths
    model_names = [os.path.basename(cfg['path'].rstrip('/')) for cfg in models_config]
    # Create a shortened string of model names
    if len(model_names) > 1:
        model_str = '-'.join([name for name in model_names])
    else:
        model_str = model_names[0]

    # Create filename
    filename = f"{strategy}_{model_str}.json"
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

def run_benchmark(args, strategy: str, model_configs: List[decoding.ModelConfig], monitor: GPUMonitor):
    # Initialize decoder based on strategy
    if strategy == 'greedy':
        decoder = decoding.GreedyDecoder(model_configs[-1], max_new_tokens=MAX_NEW_TOKENS)
    elif strategy == 'chain':
        if len(model_configs) < 2:
            raise ValueError("chain strategy requires at least 2 models")
        decoder = decoding.ChainSpeculativeDecoder(model_configs, draft_lookahead=args.lookahead, 
                                                 max_new_tokens=MAX_NEW_TOKENS)
    elif strategy == 'async-chain':
        if len(model_configs) < 2:
            raise ValueError("async-chain strategy requires at least 2 models")
        decoder = decoding.AsyncChainSpeculativeDecoder(model_configs, max_new_tokens=MAX_NEW_TOKENS)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Load tokenizer from the largest model
    tok = AutoTokenizer.from_pretrained(model_configs[-1].model_path)
    
    try:
        dataset = load_dataset(args.dataset)
        dataset = dataset["train"]
    except:
        dataset = load_from_disk(args.dataset)
    
    results = []
    num_samples = len(dataset) if args.num_samples is None else min(args.num_samples, len(dataset))
    
    for i in tqdm(range(num_samples), desc=f"Running {strategy}"):
        conversation = dataset[i]['conversation']
        last_assistant_turn = next((turn for turn in reversed(conversation) if turn['role'] == 'assistant'), None)
        if last_assistant_turn is None:
            last_turn_index = len(conversation)
        else:
            last_turn_index = conversation.index(last_assistant_turn)
        
        conversation = conversation[:last_turn_index]
        input_ids = tok.apply_chat_template(conversation, return_tensors="pt")
        
        # Generate and monitor
        monitor.start()
        output_ids, metrics = decoder.generate(input_ids, return_metrics=True)
        monitor.stop()
        
        # Add results
        result = {
            "sample_index": i,
            "strategy": strategy,
            "metrics": metrics,
            "output_text": tok.decode(output_ids, skip_special_tokens=True),
            "input_text": tok.decode(input_ids[0], skip_special_tokens=True),
            "gpustats": monitor.get_results()
        }
        results.append(result)
        monitor.clear()
    
    if hasattr(decoder, 'close'):
        decoder.close()
    del decoder
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True,
                      choices=STRATEGIES,
                      help="Decoding strategy to benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path or name of the dataset to use")
    parser.add_argument("--num-samples", type=int,
                      help="Number of samples to process (optional, processes all samples if not specified)")
    parser.add_argument("--lookahead", type=int, default=4,
                      help="Number of tokens to look ahead in speculative decoding")
    parser.add_argument("--models-config-path", type=str, required=True,
                      help="Path to the model configs JSON file")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                      help="Output directory for results")
    args = parser.parse_args()
    
    # Load model configurations
    with open(args.models_config_path, 'r') as f:
        models_json = json.load(f)
        
    model_configs = []
    for cfg in models_json:
        dtype = torch.bfloat16 if cfg.get('dtype', 'float16') == 'bfloat16' else torch.float16
        quantization_config = None
        if cfg.get('quantize') == '4bit':
            quantization_config = decoding.create_4bit_config(compute_dtype=dtype)
        elif cfg.get('quantize') == '8bit':
            quantization_config = decoding.create_8bit_config()
            
        model_config = decoding.ModelConfig(
            model_path=cfg['path'],
            devices=cfg['devices'],
            dtype=dtype,
            quantization_config=quantization_config
        )
        model_configs.append(model_config)
    
    # Initialize monitor
    monitor = GPUMonitor()
    
    # Run benchmark for the specified strategy
    results = run_benchmark(args, args.strategy, model_configs, monitor)
    
    # Add metadata to results
    metadata = {
        "strategy": args.strategy,
        "model_configs": models_json,
        "lookahead": args.lookahead,
        "dataset": args.dataset,
        "num_samples": args.num_samples
    }
    
    final_output = {
        "metadata": metadata,
        "results": results
    }
    
    # Save results
    output_file = create_output_filename(args.strategy, models_json, args.output_dir)
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
