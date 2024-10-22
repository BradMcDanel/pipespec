import argparse
import json
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import amusd
from gpu_monitor import GPUMonitor
import gc

MAX_NEW_TOKENS = 4096
STRATEGIES = ['greedy', 'sd', 'amusd']

def run_benchmark(args, strategy, monitor):
    if strategy == 'greedy':
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float16)
        decoder = amusd.GreedyDecoder(verify, max_new_tokens=MAX_NEW_TOKENS)
    elif strategy == 'sd':
        draft = amusd.ModelConfig(args.draft_model_path, "cuda:0", torch.float16)
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float16)
        decoder = amusd.SyncSpeculativeDecoder(draft, verify, max_new_tokens=MAX_NEW_TOKENS)
    elif strategy == 'amusd':
        draft = amusd.ModelConfig(args.draft_model_path, "cuda:1", torch.float16)
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float16)
        decoder = amusd.AsyncMultiGPUSpeculativeDecoder(draft, verify, max_new_tokens=MAX_NEW_TOKENS)
    
    tok = AutoTokenizer.from_pretrained(args.verify_model_path)
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
        monitor.start()
        _, metrics = decoder.generate(input_ids, return_metrics=True)
        monitor.stop()
        metrics["gpustats"] = monitor.get_results()
        monitor.clear()
        results.append({
            "sample_index": i,
            "strategy": strategy,
            "metrics": metrics
        })
    
    if strategy == 'amusd':
        decoder.close()

    del decoder
    torch.cuda.empty_cache()
    gc.collect()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model-path", type=str, required=True, help="Path to the draft model")
    parser.add_argument("--verify-model-path", type=str, required=True, help="Path to the verify model")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--num-samples", type=int, help="Number of samples to process (optional, processes all samples if not specified)")
    parser.add_argument("--output-file", type=str, default="benchmark_results.json", help="Output JSON file name")
    parser.add_argument("--strategies", nargs='+', choices=STRATEGIES, default=STRATEGIES,
                        help="Strategies to benchmark (default: all strategies)")
    args = parser.parse_args()
    
    monitor = GPUMonitor()
    
    all_results = []
    
    for strategy in args.strategies:
        results = run_benchmark(args, strategy, monitor)
        all_results.extend(results)
    
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
