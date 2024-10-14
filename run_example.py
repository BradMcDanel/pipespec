import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import amusd
import torch.multiprocessing as mp

MAX_NEW_TOKENS = 4096

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Path to the draft model")
    parser.add_argument("--verify-model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path to the verify model (for sd and amusd)")
    parser.add_argument("--strategy", type=str, default="edit-based", choices=["greedy", "sd", "amusd"], help="Decoding strategy")
    parser.add_argument("--dataset", type=str, default="lmsys-chat-100k-clustered-labeled", help="Name of the dataset to use")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the sample to use from the dataset")
    args = parser.parse_args()

    if args.strategy == 'greedy':
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float32)
        decoder = amusd.GreedyDecoder(verify, max_new_tokens=MAX_NEW_TOKENS)
    elif args.strategy == 'sd':
        draft = amusd.ModelConfig(args.draft_model_path, "cuda:0", torch.float32)
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float32)
        decoder = amusd.SyncSpeculativeDecoder(draft, verify, max_new_tokens=MAX_NEW_TOKENS)
    elif args.strategy == 'amusd':
        mp.set_start_method('spawn')
        draft = amusd.ModelConfig(args.draft_model_path, "cuda:1", torch.float32)
        verify = amusd.ModelConfig(args.verify_model_path, "cuda:0", torch.float32)
        decoder = amusd.AsyncMultiGPUSpeculativeDecoder(draft, verify, max_new_tokens=MAX_NEW_TOKENS)

    tok = AutoTokenizer.from_pretrained(verify.model_path)

    dataset = load_from_disk(args.dataset)

    for i in range(args.sample_index, args.sample_index + 1000):
        if len(dataset[i]['conversation']) > 2:
            args.sample_index = i
            break

    conversation = dataset[args.sample_index]['conversation']

    # Process only the last turn
    # print number of turns
    print(f"Number of turns: {len(conversation)}")
    last_assistant_turn = next((turn for turn in reversed(conversation) if turn['role'] == 'assistant'), None)
    if last_assistant_turn is None:
        raise ValueError("No assistant turn found in the conversation")

    last_turn_index = conversation.index(last_assistant_turn)
    input_ids = tok.apply_chat_template(conversation[:last_turn_index], return_tensors="pt")

    output_ids, metrics = decoder.generate(input_ids, return_metrics=True)
    print(output_ids)
    print(tok.decode(output_ids, skip_special_tokens=False))
    print(metrics)

    if args.strategy == 'amusd':
        decoder.close()
