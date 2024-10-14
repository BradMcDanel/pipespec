import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import amusd
from datasets import load_from_disk


if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    # verify = amusd.ModelConfig("meta-llama/Llama-3.1-8B-Instruct", "cuda:0", torch.float32)
    # decoder = amusd.GreedyDecoder(verify)

    # draft = amusd.ModelConfig("meta-llama/Llama-3.2-1B-Instruct", "cuda:0", torch.float32)
    # verify = amusd.ModelConfig("meta-llama/Llama-3.1-8B-Instruct", "cuda:0", torch.float32)
    # decoder = amusd.SyncSpeculativeDecoder(draft, verify)

    draft = amusd.ModelConfig("meta-llama/Llama-3.2-1B-Instruct", "cuda:1", torch.float32)
    verify = amusd.ModelConfig("meta-llama/Llama-3.1-8B-Instruct", "cuda:0", torch.float32)
    decoder = amusd.AsyncMultiGPUSpeculativeDecoder(draft, verify)
    
    tok = AutoTokenizer.from_pretrained(verify.model_path)
    
    conversation = [{"role": "user", "content": "Who wrote the Illiad and where?"}]
    input_ids = tok.apply_chat_template(conversation, return_tensors="pt")
    
    output_ids, metrics = decoder.generate(input_ids, return_metrics=True)
    print(output_ids)
    print(tok.decode(output_ids, skip_special_tokens=False))
    print(metrics)

    # decoder.close()

