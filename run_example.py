import argparse
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk, load_dataset
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional
import json
mp.set_start_method('spawn', force=True)
import decoding
from gpu_monitor import GPUMonitor

MAX_NEW_TOKENS = 4096

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="three-model", 
                      choices=["greedy", "sd", "amusd", "three-model", "amusd-three", "chain"],
                      help="Decoding strategy (three-model=sync, amusd-three=async)")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path or name of the dataset to use")
    parser.add_argument("--sample-index", type=int, default=0,
                      help="Index of the sample to use from the dataset")
    parser.add_argument("--lookahead", type=int, default=4,
                      help="Number of tokens to look ahead in speculative decoding")
    parser.add_argument("--models-config-path", type=str, required=True, help="Path to the model configs")
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
    elif args.strategy == 'sd':
        if len(model_configs) < 2:
            raise ValueError("sd strategy requires at least 2 models")
        decoder = decoding.SyncSpeculativeDecoder(
            model_configs[0], 
            model_configs[-1], 
            max_new_tokens=MAX_NEW_TOKENS
        )
    elif args.strategy == 'amusd':
        if len(model_configs) < 2:
            raise ValueError("amusd strategy requires at least 2 models")
        decoder = decoding.AsyncMultiGPUSpeculativeDecoder(
            model_configs[0],
            model_configs[-1],
            max_new_tokens=MAX_NEW_TOKENS
        )
    elif args.strategy == 'three-model':
        if len(model_configs) < 3:
            raise ValueError("three-model strategy requires at least 3 models")
        decoder = decoding.ThreeStageSpeculativeDecoder(
            model_configs[0],  # small
            model_configs[1],  # medium
            model_configs[2],  # large
            lookahead=args.lookahead,
            max_new_tokens=MAX_NEW_TOKENS
        )
    elif args.strategy == 'chain':
        if len(model_configs) < 2:
            raise ValueError("chain strategy requires at least 2 models")
        decoder = decoding.ChainSpeculativeDecoder(model_configs, max_new_tokens=MAX_NEW_TOKENS)
    elif args.strategy == 'amusd-three':
        if len(model_configs) < 3:
            raise ValueError("amusd-three strategy requires at least 3 models")
        decoder = decoding.ThreeStageAsyncSpeculativeDecoder(
            model_configs[0],  # small
            model_configs[1],  # medium
            model_configs[2],  # large
            max_new_tokens=MAX_NEW_TOKENS
        )

    # Load tokenizer from the largest model
    tok = AutoTokenizer.from_pretrained(model_configs[-1].model_path)

    try:
        dataset = load_dataset(args.dataset)
        dataset = dataset["train"]
    except:
        dataset = load_from_disk(args.dataset)

    conversation = dataset[args.sample_index]['conversation']
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
    metrics["gpustats"] = monitor.get_results()
    
    # print(tok.decode(output_ids, skip_special_tokens=True))
    print(metrics["time_per_token"])

    if hasattr(decoder, 'close'):
        decoder.close()
