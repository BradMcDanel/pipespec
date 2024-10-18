import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import amusd
from gpu_monitor import GPUMonitor

MAX_NEW_TOKENS = 4096

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Path to the draft model")
    parser.add_argument("--verify-model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path to the verify model (for sd and amusd)")
    parser.add_argument("--strategy", type=str, default="edit-based", choices=["greedy", "sd", "amusd"], help="Decoding strategy")
    parser.add_argument("--dataset", type=str, default="lmsys-chat-100k-clustered-labeled", help="Name of the dataset to use")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the sample to use from the dataset")
    args = parser.parse_args()

    monitor = GPUMonitor()

    if args.strategy == 'greedy':
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float16)
        decoder = amusd.GreedyDecoder(verify, max_new_tokens=MAX_NEW_TOKENS)
    elif args.strategy == 'sd':
        draft = amusd.ModelConfig(args.draft_model_path, "cuda:0", torch.float16)
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float16)
        decoder = amusd.SyncSpeculativeDecoder(draft, verify, max_new_tokens=MAX_NEW_TOKENS)
    elif args.strategy == 'amusd':
        draft = amusd.ModelConfig(args.draft_model_path, "cuda:1", torch.float16)
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float16)
        decoder = amusd.AsyncMultiGPUSpeculativeDecoder(draft, verify, max_new_tokens=MAX_NEW_TOKENS)

    tok = AutoTokenizer.from_pretrained(verify.model_path)

    dataset = load_from_disk(args.dataset)
    conversation = dataset[args.sample_index]['conversation']

    last_assistant_turn = next((turn for turn in reversed(conversation) if turn['role'] == 'assistant'), None)
    if last_assistant_turn is None:
        last_turn_index = len(conversation)
    else:
        last_turn_index = conversation.index(last_assistant_turn)

    conversation = conversation[:last_turn_index]

    input_ids = tok.apply_chat_template(conversation, return_tensors="pt")

    monitor.start()
    output_ids, metrics = decoder.generate(input_ids, return_metrics=True)
    monitor.stop()
    metrics["gpustats"] = monitor.get_results()
    
    print(tok.decode(output_ids, skip_special_tokens=True))
    print(metrics)

    if args.strategy == 'amusd':
        decoder.close()
