import time

import torch.multiprocessing as mp
from transformers import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def find_divergence_point(tensor1, tensor2):
    # Get the minimum length between the two tensors
    min_length = min(tensor1.size(0), tensor2.size(0))
    
    # Compare only the common part
    common_part1 = tensor1[:min_length]
    common_part2 = tensor2[:min_length]
    
    # Find where the common parts differ
    diff_mask = common_part1 != common_part2
    
    if not diff_mask.any():
        # If common parts are identical, check if sizes are different
        if tensor1.size(0) != tensor2.size(0):
            return min_length
        else:
            return None  # Tensors are identical
    
    # Find the first index where they differ
    divergence_index = diff_mask.nonzero(as_tuple=True)[0][0].item()
    
    return divergence_index

def init_model(model_path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

def step(model, input_ids, output_ids=None, past_key_values=None, num_input_tokens=-1):
    input_ids = input_ids.to(model.device)

    if past_key_values is None:
        past_key_values = DynamicCache()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

    next_token_logits = outputs.logits[:, -1, :]
    next_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

    if output_ids is None:
        output_ids = next_ids
    else:
        output_ids = torch.cat([output_ids, next_ids], dim=1)

    if num_input_tokens == -1:
        num_input_tokens = input_ids.size(1)

    return {
        "input_ids": next_ids,
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "past_key_values": past_key_values
    }

def build_verify_inputs(draft_output_ids, input_ids, output_ids, past_key_values, num_input_tokens):
    draft_ids = draft_output_ids.to(output_ids.device)

    # Get lengths of both tensors
    verify_len = output_ids.size(1)
    draft_len = draft_ids.size(1)

    # Case 1: Valid state is longer or tokens are different (we don't do anything)
    if verify_len > draft_len or not torch.equal(output_ids[:, :verify_len], draft_ids[:, :verify_len]):
        pass
    else:
        input_ids = draft_ids[:, verify_len-1:]

    return {
        "output_ids": output_ids,
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "num_input_tokens": num_input_tokens
    }

def build_draft_inputs(verify_output_ids, input_ids, output_ids, past_key_values, num_input_tokens):
    verify_ids = verify_output_ids.to(input_ids.device)
    divergence_index = find_divergence_point(output_ids[0], verify_ids[0])

    if divergence_index is not None:
        past_key_values.crop(num_input_tokens + divergence_index)
        input_ids = verify_ids[:, divergence_index-1:]
        output_ids = verify_ids

    return {
        "output_ids": output_ids,
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "num_input_tokens": num_input_tokens
    }


def compute_verified_ids(input_ids, verify_output_ids):
    # get where each tensor is 
    a = input_ids[:, 1:] != verify_output_ids[:, :-1]
    b = a.cumsum(-1) == 0
    idx =  b.sum(-1) + 1

    verify_ids = verify_output_ids[:, :idx[0]]
    return verify_ids

def perform_verification(input_ids, verify_outputs, output_ids, past_key_values, num_input_tokens):
    # Get the predicted output ids
    verify_output_ids = torch.argmax(verify_outputs.logits, dim=-1)
    
    # Compute verified_ids without converting tensors to lists
    verified_ids = compute_verified_ids(input_ids, verify_output_ids)

    # Update output_ids and input_ids
    output_ids = torch.cat([output_ids, verified_ids], dim=1)
    input_ids = verified_ids[:, -1:]
    
    # Crop past_key_values if necessary (assuming crop is properly defined)
    past_key_values.crop(num_input_tokens + output_ids.size(1) - 1)
    
    return output_ids, input_ids, past_key_values


def verify_step(model, input_ids, output_ids, past_key_values, num_input_tokens):
    print(input_ids.size())
    # (1) Model forward pass
    with torch.no_grad():
        verify_outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    
    # (2) Verification and update process
    if input_ids.size(1) > 1:
        output_ids, input_ids, past_key_values = perform_verification(
            input_ids, verify_outputs, output_ids, past_key_values, num_input_tokens
        )
    else:
        next_token_logits = verify_outputs.logits[:, -1, :]
        input_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        output_ids = torch.cat([output_ids, input_ids], dim=1)

    return {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "past_key_values": past_key_values,
        "num_input_tokens": num_input_tokens
    }

def standard_decoding(model, tokenizer, input_ids, max_new_tokens):
    num_input_tokens = input_ids.size(1)
    
    # Initialize DynamicCache
    cache = DynamicCache()
    
    # Initialize output_ids with input_ids
    output_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        input_ids = next_token.unsqueeze(0)  # Only use the last token as input
        output_ids = torch.cat([output_ids, input_ids], dim=1)  # Append the new token to output_ids
        
        # Update cache
        cache = outputs.past_key_values
        
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Use output_ids for the final output
    generated_ids = output_ids[0, num_input_tokens:]

    return generated_ids

class AsyncMultiGPUSpeculativeDecoder:
    def __init__(self, draft_model_path, verify_model_path, draft_device="cuda:0", verify_device="cuda:1", draft_dtype=torch.float32, verify_dtype=torch.float32):
        self.draft_model_path = draft_model_path
        self.verify_model_path = verify_model_path
        self.draft_device = draft_device
        self.verify_device = verify_device
        self.draft_dtype = draft_dtype
        self.verify_dtype = verify_dtype
        
        self.manager = mp.Manager()
        self.draft_ready_event = self.manager.Event()
        self.verify_ready_event = self.manager.Event()
        self.start_event = self.manager.Event()
        self.completion_event = self.manager.Event()
        self.close_event = self.manager.Event()
        
        self.max_length = 128000
        self.draft_tensor = torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_()
        self.verify_tensor = torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_()

        self.max_log_entries = 100000
        self.draft_times = torch.zeros(self.max_log_entries, dtype=torch.float32).share_memory_()
        self.verify_times = torch.zeros(self.max_log_entries, dtype=torch.float32).share_memory_()
        self.accepted_tokens = torch.zeros(self.max_log_entries, dtype=torch.int32).share_memory_()
        self.token_buffer_sizes = torch.zeros(self.max_log_entries, dtype=torch.float32).share_memory_()
        self.draft_index = torch.tensor([0], dtype=torch.int32).share_memory_()
        self.verify_index = torch.tensor([0], dtype=torch.int32).share_memory_()
        
        self.pool = mp.Pool(processes=2)
        self._start_processes()

    def _start_processes(self):
        self.verify_process = self.pool.apply_async(self._verify_process, (
            self.verify_model_path, self.verify_device, self.verify_dtype,
            self.draft_tensor, self.verify_tensor,
            self.verify_ready_event, self.start_event, self.completion_event, self.close_event,
            self.verify_times, self.accepted_tokens, self.token_buffer_sizes,
            self.verify_index, self.max_log_entries
        ))
        self.draft_process = self.pool.apply_async(self._draft_process, (
            self.draft_model_path, self.draft_device, self.draft_dtype,
            self.draft_tensor, self.verify_tensor,
            self.draft_ready_event, self.start_event, self.completion_event, self.close_event,
            self.draft_times, self.draft_index, self.max_log_entries
        ))
        
        self.draft_ready_event.wait()
        self.verify_ready_event.wait()

    @staticmethod
    def _draft_process(model_path, device, dtype, draft_tensor, verify_tensor, ready_event, start_event, completion_event, close_event, draft_times, draft_index, max_log_entries):
        try:
            draft_model, _ = init_model(model_path, device, dtype)
            ready_event.set()
            
            while True:
                start_event.wait()
                if close_event.is_set():
                    break
                input_ids = draft_tensor[1:draft_tensor[0].item() + 1].unsqueeze(0).to(device)
                draft_state = step(draft_model, input_ids)
                draft_tensor[1:1+draft_state["output_ids"].size(1)] = draft_state["output_ids"]
                draft_tensor[0] = draft_state["output_ids"].size(1)
                
                last_verify_length = 0
                local_index = 0
                while not completion_event.is_set():
                    start_time = time.time()

                    draft_state = step(draft_model, **draft_state)

                    # write to shared memory
                    output_ids = draft_state["output_ids"]
                    draft_tensor[1:1+output_ids.size(1)] = output_ids
                    draft_tensor[0] = output_ids.size(1)

                    verify_length = verify_tensor[0].item()
                    if verify_length > last_verify_length:
                        verify_output_ids = verify_tensor[1:verify_length+1].unsqueeze(0)
                        draft_state = build_draft_inputs(verify_output_ids, **draft_state)
                        last_verify_length = verify_length

                    end_time = time.time()

                    # Log draft time
                    idx = local_index % max_log_entries
                    draft_times[idx] = end_time - start_time
                    local_index += 1
                    draft_index[0] = local_index
                
                draft_index[0] = local_index
        except Exception as e:
            print(f"Exception in draft process: {e}")

    @staticmethod
    def _verify_process(model_path, device, dtype, draft_tensor, verify_tensor, ready_event, start_event, completion_event, close_event, verify_times, accepted_tokens, token_buffer_sizes, verify_index, max_log_entries):
        try:
            verify_model, tok = init_model(model_path, device, dtype)
            ready_event.set()
            
            while True:
                start_event.wait()
                if close_event.is_set():
                    break

                max_new_tokens = 4096
                token_buffer_size = 1.0
                growth_exponent = 2
                decay_exponent = 2

                input_ids = verify_tensor[1:verify_tensor[0].item() + 1].unsqueeze(0).to(device)
                verify_state = step(verify_model, input_ids)
                
                local_index = 0
                while verify_state["output_ids"].size(1) < max_new_tokens:
                    start_time = time.time()

                    # update shared memory
                    output_ids = verify_state["output_ids"]
                    verify_tensor[1:1+output_ids.size(1)] = output_ids
                    verify_tensor[0] = output_ids.size(1)

                    while draft_tensor[0].item() < verify_tensor[0].item() + int(round(token_buffer_size)):
                        pass

                    draft_length = draft_tensor[0].item()
                    draft_output_ids = draft_tensor[1:draft_length+1].unsqueeze(0)
                    num_old_verified_tokens = verify_state["output_ids"].size(1)

                    verify_state = build_verify_inputs(draft_output_ids, **verify_state)
                    verify_state = verify_step(verify_model, **verify_state)
                    
                    num_new_verified_tokens = verify_state["output_ids"].size(1) - num_old_verified_tokens
                    print(num_new_verified_tokens)
                    
                    # Dynamically adjust token_buffer_size
                    if num_new_verified_tokens > token_buffer_size:
                        # Growth
                        token_buffer_size *= (num_new_verified_tokens / token_buffer_size) ** growth_exponent
                    else:
                        # Decay
                        token_buffer_size *= (num_new_verified_tokens / token_buffer_size) ** decay_exponent
                    
                    token_buffer_size = max(1.0, token_buffer_size)

                    end_time = time.time()

                    # Log metrics
                    idx = local_index % max_log_entries
                    verify_times[idx] = end_time - start_time
                    accepted_tokens[idx] = num_new_verified_tokens
                    token_buffer_sizes[idx] = token_buffer_size
                    local_index += 1
                    
                    # Check only newly generated tokens for eos_token_id
                    if tok.eos_token_id in verify_state["output_ids"][0, -num_new_verified_tokens:]:
                        break
                
                verify_tensor[0] = verify_state["output_ids"].size(1)
                verify_tensor[1:1+verify_tensor[0].item()] = verify_state["output_ids"]
                verify_index[0] = local_index
                start_event.clear()
                completion_event.set()
        except Exception as e:
            print(f"Exception in verify process: {e}")

    def generate(self, input_ids, return_metrics=False):
        # Send input_ids to shared memory
        input_length = input_ids.size(1)
        self.draft_tensor[0] = input_length
        self.draft_tensor[1:1+input_length] = input_ids[0]
        self.verify_tensor[0] = input_length
        self.verify_tensor[1:1+input_length] = input_ids[0]

        # Start the processes
        self.completion_event.clear()
        self.start_event.set()

        # Running the processes in parallel
        start_time = time.time()
        self.completion_event.wait()
        total_time = time.time() - start_time
        
        final_output_ids = self.verify_tensor[1:self.verify_tensor[0].item()].clone()
        tokens_generated = len(final_output_ids)
        time_per_token = total_time / tokens_generated

        # Process logged metrics
        num_draft_logs = min(self.draft_index[0].item(), self.max_log_entries)
        num_verify_logs = min(self.verify_index[0].item(), self.max_log_entries)
        draft_times = self.draft_times[:num_draft_logs].tolist()
        verify_times = self.verify_times[:num_verify_logs].tolist()
        accepted_tokens = self.accepted_tokens[:num_verify_logs].tolist()
        token_buffer_sizes = self.token_buffer_sizes[:num_verify_logs].tolist()

        metrics = {
            "draft_times": draft_times,
            "verify_times": verify_times,
            "accepted_tokens": accepted_tokens,
            "token_buffer_sizes": token_buffer_sizes,
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "time_per_token": time_per_token
        }

        # Reset shared memory and indices
        self.draft_tensor.fill_(0)
        self.verify_tensor.fill_(0)
        self.draft_times.fill_(0)
        self.verify_times.fill_(0)
        self.accepted_tokens.fill_(0)
        self.token_buffer_sizes.fill_(0)
        self.draft_index.fill_(0)
        self.verify_index.fill_(0)

        if return_metrics:
            return final_output_ids, metrics

        return final_output_ids

    def close(self):
        self.start_event.set()
        self.close_event.set()
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
