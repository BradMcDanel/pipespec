import argparse
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk, load_dataset
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional
import json
from dataset_prompts import get_dataset, build_book_prompt  # Import our dataset handling functions

mp.set_start_method('spawn', force=True)
import decoding
from gpu_monitor import GPUMonitor

MAX_NEW_TOKENS = 1024

# List of datasets that use our custom prompt format
CUSTOM_PROMPT_DATASETS = ['cnn_dailymail', 'pg19', 'narrativeqa', 'one-shot']

def apply_basic_chat_template(conversation, tokenizer):
    """
    Applies a simple chat template for non-instruction tuned models.
    Format: USER:\n{user_message}\nASSISTANT:\n{assistant_message}
    """
    formatted_text = ""
    for turn in conversation:
        role = turn['role'].upper()
        content = turn['content']
        formatted_text += f"{role}:\n{content}\n"
    
    if not formatted_text.rstrip().endswith("ASSISTANT:"):
        formatted_text += "ASSISTANT:\n"
    
    return tokenizer.encode(formatted_text, return_tensors="pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="three-model", 
                      choices=["greedy", "chain", "async-chain"],
                      help="Decoding strategy (three-model=sync, amusd-three=async)")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path or name of the dataset to use")
    parser.add_argument("--sample-index", type=int, default=0,
                      help="Index of the sample to use from the dataset")
    parser.add_argument("--lookahead", type=int, default=4,
                      help="Number of tokens to look ahead in speculative decoding")
    parser.add_argument("--models-config-path", type=str, required=True, 
                      help="Path to the model configs")
    parser.add_argument("--prefill", type=int, default=4096,
                      help="Number of tokens to use from book text")
    parser.add_argument("--use-basic-template", action="store_true",
                      help="Use basic USER/ASSISTANT template instead of model's chat template")
    args = parser.parse_args()

    # Parse model configurations
    model_configs = []
    with open(args.models_config_path, 'r') as f:
        models_json = json.load(f)
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

    monitor = GPUMonitor()
    # Initialize decoder based on strategy
    if args.strategy == 'greedy':
        decoder = decoding.GreedyDecoder(model_configs[-1], max_new_tokens=MAX_NEW_TOKENS)
    elif args.strategy == 'chain':
        if len(model_configs) < 2:
            raise ValueError("chain strategy requires at least 2 models")
        decoder = decoding.ChainSpeculativeDecoder(model_configs, draft_lookahead=args.lookahead,
                                                   max_new_tokens=MAX_NEW_TOKENS)
    elif args.strategy == 'async-chain':
        if len(model_configs) < 2:
            raise ValueError("async-chain strategy requires at least 2 models")
        decoder = decoding.AsyncChainSpeculativeDecoder(model_configs, draft_lookahead=args.lookahead,
                                                        max_new_tokens=MAX_NEW_TOKENS)

    # Load tokenizer from the largest model
    tok = AutoTokenizer.from_pretrained(model_configs[-1].model_path)

    # Process dataset based on type
    dataset_name = args.dataset.split('/')[-1] if '/' in args.dataset else args.dataset
    
    if dataset_name in CUSTOM_PROMPT_DATASETS:
        # Use our custom prompt handling for book datasets
        prompts = get_dataset(dataset_name, tok, args.prefill)
        input_ids = prompts[args.sample_index]
    else:
        # Use standard chat dataset handling
        try:
            dataset = load_dataset(args.dataset)
            dataset = dataset["train"]
        except:
            dataset = load_from_disk(args.dataset)
        
        conversation = dataset[args.sample_index]['conversation']
        last_assistant_turn = next((turn for turn in reversed(conversation) 
                                  if turn['role'] == 'assistant'), None)
        if last_assistant_turn is None:
            last_turn_index = len(conversation)
        else:
            last_turn_index = conversation.index(last_assistant_turn)
        
        conversation = conversation[:last_turn_index]
        
        if args.use_basic_template:
            input_ids = apply_basic_chat_template(conversation, tok)
        else:
            input_ids = tok.apply_chat_template(conversation, return_tensors="pt")

    # Generate and monitor
    monitor.start()
    output_ids, metrics = decoder.generate(input_ids, return_metrics=True)
    monitor.stop()
    metrics["gpustats"] = monitor.get_results()
    
    print(tok.decode(output_ids, skip_special_tokens=True))
    print(metrics["time_per_token"])
    if hasattr(decoder, 'close'):
        decoder.close()
