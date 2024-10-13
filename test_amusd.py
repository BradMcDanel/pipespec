import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import amusd
from datasets import load_from_disk


if __name__ == "__main__":
    mp.set_start_method('spawn')

    mp1 = "meta-llama/Llama-3.2-1B-Instruct"
    mp2 = "meta-llama/Llama-3.1-8B-Instruct"
    
    decoder = amusd.AsyncMultiGPUSpeculativeDecoder(mp1, mp2)
    
    tok = AutoTokenizer.from_pretrained(mp2)
    
    conversation = [{"role": "user", "content": "Who wrote the Illiad and where?"}]
    input_ids = tok.apply_chat_template(conversation, return_tensors="pt")
    
    output_ids, metrics = decoder.generate(input_ids, return_metrics=True)
    print(tok.decode(output_ids, skip_special_tokens=False))
    print(metrics)

    # You can generate more outputs with different inputs
    conversation = [{"role": "user", "content": "What is the capital of France?"}]
    input_ids = tok.apply_chat_template(conversation, return_tensors="pt")
    
    output_ids = decoder.generate(input_ids)
    print(tok.decode(output_ids, skip_special_tokens=False))

    decoder.close()
