import time
from dataclasses import dataclass

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

@dataclass
class ModelConfig:
    model_path: str
    device: str
    dtype: torch.dtype

@dataclass
class SharedMemory:
    input_tensor: torch.Tensor
    draft_tensor: torch.Tensor
    verify_tensor: torch.Tensor
    rollback_requests: torch.Tensor
    rollback_responses: torch.Tensor
    draft_times: torch.Tensor
    verify_times: torch.Tensor
    accepted_tokens: torch.Tensor
    draft_index: torch.Tensor
    verify_index: torch.Tensor

class GreedyDecoder:
    def __init__(self, model_config, max_new_tokens=4096):
        self.model_config = model_config
        self.max_new_tokens = max_new_tokens
        self._init_model()

    def _init_model(self):
        self.model, self.tokenizer = init_model(
            self.model_config.model_path,
            self.model_config.device,
            self.model_config.dtype
        )

    def generate(self, input_ids, return_metrics=False):
        start_time = time.time()

        state = step(self.model, input_ids)

        metrics = {
            "token_times": []
        }
        
        for _ in range(self.max_new_tokens):
            token_start_time = time.time()
            
            state = step(self.model, **state)
            
            token_time = time.time() - token_start_time
            metrics["token_times"].append(token_time)
            
            if state["input_ids"].item() == self.tokenizer.eos_token_id:
                break

        total_time = time.time() - start_time
        tokens_generated = state["output_ids"].size(1)
        time_per_token = total_time / tokens_generated

        metrics.update({
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "time_per_token": time_per_token
        })

        # Use output_ids for the final output
        generated_ids = state["output_ids"][0]

        if return_metrics:
            return generated_ids, metrics

        return generated_ids

class SyncSpeculativeDecoder:
    def __init__(self, draft_config: ModelConfig, verify_config: ModelConfig, 
                 draft_lookahead: int = 5,
                 max_new_tokens: int = 4096):
        self.draft_config = draft_config
        self.verify_config = verify_config
        self.draft_lookahead = draft_lookahead
        self.max_new_tokens = max_new_tokens

        self._init_models()

    def _init_models(self):
        self.draft_model, _ = init_model(self.draft_config.model_path, self.draft_config.device, self.draft_config.dtype)
        self.verify_model, self.tokenizer = init_model(self.verify_config.model_path, self.verify_config.device, self.verify_config.dtype)

    def generate(self, input_ids, return_metrics=False):
        draft_state = step(self.draft_model, input_ids)
        verify_state = step(self.verify_model, input_ids)

        start_time = time.time()
        metrics = {
            "draft_times": [],
            "verify_times": [],
            "accepted_tokens": [],
        }

        while verify_state["output_ids"].size(1) < self.max_new_tokens:
            # Draft step
            while draft_state["output_ids"].size(1) < verify_state["output_ids"].size(1) + self.draft_lookahead:
                draft_start_time = time.time()
                draft_state = step(self.draft_model, **draft_state)
                draft_time = time.time() - draft_start_time
                metrics["draft_times"].append(draft_time)

            # Verify step
            verify_start_time = time.time()
            num_old_verified_tokens = verify_state["output_ids"].size(1)
            verify_state = build_verify_inputs(draft_state["output_ids"], **verify_state)
            verify_state = verify_step(self.verify_model, **verify_state)
            num_new_verified_tokens = verify_state["output_ids"].size(1) - num_old_verified_tokens
            verify_time = time.time() - verify_start_time

            # Update metrics
            metrics["verify_times"].append(verify_time)
            metrics["accepted_tokens"].append(num_new_verified_tokens)

            # Check for EOS token
            if self.tokenizer.eos_token_id in verify_state["output_ids"][0, -num_new_verified_tokens:]:
                eos_index = (verify_state["output_ids"][0] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][-1].item()
                verify_state["output_ids"] = verify_state["output_ids"][:, :eos_index+1]
                break

            # Update draft state
            draft_state = build_draft_inputs(verify_state["output_ids"], **draft_state)

        total_time = time.time() - start_time
        tokens_generated = verify_state["output_ids"].size(1)
        time_per_token = total_time / tokens_generated

        metrics.update({
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "time_per_token": time_per_token
        })

        if return_metrics:
            return verify_state["output_ids"][0], metrics

        return verify_state["output_ids"][0]

class AsyncMultiGPUSpeculativeDecoder:
    def __init__(self, draft_config: ModelConfig, verify_config: ModelConfig, 
                 max_new_tokens: int = 4096, max_length: int = 8096, 
                 max_log_entries: int = 10000):
        self.draft_config = draft_config
        self.verify_config = verify_config
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.max_log_entries = max_log_entries

        self._init_shared_memory()
        self._init_multiprocessing()

    def _init_shared_memory(self):
        self.shared_memory = SharedMemory(
            input_tensor=torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_(),
            draft_tensor=torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_(),
            rollback_requests=torch.zeros(1, dtype=torch.int32).share_memory_(),
            rollback_responses=torch.zeros(1, dtype=torch.int32).share_memory_(),
            verify_tensor=torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_(),
            draft_times=torch.zeros(self.max_log_entries, dtype=torch.float32).share_memory_(),
            verify_times=torch.zeros(self.max_log_entries, dtype=torch.float32).share_memory_(),
            accepted_tokens=torch.zeros(self.max_log_entries, dtype=torch.int32).share_memory_(),
            draft_index=torch.tensor([0], dtype=torch.int32).share_memory_(),
            verify_index=torch.tensor([0], dtype=torch.int32).share_memory_()
        )

    def _init_multiprocessing(self):
        self.manager = mp.Manager()
        self.draft_ready_event = self.manager.Event()
        self.draft_reset_event = self.manager.Event()
        self.verify_ready_event = self.manager.Event()
        self.start_event = self.manager.Event()
        self.completion_event = self.manager.Event()
        self.close_event = self.manager.Event()

        self.pool = mp.Pool(processes=2)
        self._start_processes()

    def _start_processes(self):
        self.verify_process = self.pool.apply_async(self._verify_process, (
            self.verify_config, self.shared_memory,
            self.verify_ready_event, self.start_event, self.completion_event, 
            self.close_event, self.max_log_entries, self.max_new_tokens
        ))
        self.draft_process = self.pool.apply_async(self._draft_process, (
            self.draft_config, self.shared_memory,
            self.draft_ready_event, self.draft_reset_event, self.start_event, self.completion_event, 
            self.close_event, self.max_log_entries
        ))
        
        self.draft_ready_event.wait()
        self.verify_ready_event.wait()

    @staticmethod
    def _draft_process(config: ModelConfig, shared_memory: SharedMemory, 
                       ready_event, reset_event, start_event, completion_event, close_event, 
                       max_log_entries):
        try:
            draft_model, _ = init_model(config.model_path, config.device, config.dtype)
            ready_event.set()

            while True:
                start_event.wait()
                if close_event.is_set():
                    break
                input_ids = shared_memory.input_tensor[1:shared_memory.input_tensor[0].item() + 1].unsqueeze(0).to(config.device)
                # wait for initial verify token
                while shared_memory.verify_tensor[0].item() == 0:
                    time.sleep(0.0001)

                draft_state = step(draft_model, input_ids)

                verify_output_ids = shared_memory.verify_tensor[1:shared_memory.verify_tensor[0].item()+1].unsqueeze(0)
                last_verify_length = verify_output_ids.size(1)
                draft_state = build_draft_inputs(verify_output_ids, **draft_state)

                draft_state = step(draft_model, input_ids)

                shared_memory.draft_tensor[1:1+draft_state["output_ids"].size(1)] = draft_state["output_ids"]
                shared_memory.draft_tensor[0] = draft_state["output_ids"].size(1)
            
                local_index = 0
                while not completion_event.is_set():
                    start_time = time.time()

                    draft_state = step(draft_model, **draft_state)

                    # write to shared memory
                    output_ids = draft_state["output_ids"]
                    shared_memory.draft_tensor[1:1+output_ids.size(1)] = output_ids
                    shared_memory.draft_tensor[0] = output_ids.size(1)
                
                    verify_length = shared_memory.verify_tensor[0].item()

                    # Verify update occured
                    if verify_length > last_verify_length:
                        last_draft_length = draft_state["output_ids"].size(1)
                        verify_output_ids = shared_memory.verify_tensor[1:verify_length+1].unsqueeze(0)
                        draft_state = build_draft_inputs(verify_output_ids, **draft_state)
                        draft_length = draft_state["output_ids"].size(1)
                        last_verify_length = verify_length
                        
                        # Rollback occured
                        if draft_length < last_draft_length:
                            shared_memory.rollback_responses[0] += 1

                    end_time = time.time()

                    # Log draft time
                    idx = local_index % max_log_entries
                    shared_memory.draft_times[idx] = end_time - start_time
                    local_index += 1
                    shared_memory.draft_index[0] = local_index
                
                shared_memory.draft_index[0] = local_index
                reset_event.set()
        except Exception as e:
            print(f"Exception in draft process: {e}")

    @staticmethod
    def _verify_process(config: ModelConfig, shared_memory: SharedMemory, 
                        ready_event, start_event, completion_event, close_event, 
                        max_log_entries, max_new_tokens):
        try:
            verify_model, tok = init_model(config.model_path, config.device, config.dtype)
            ready_event.set()
            
            while True:
                start_event.wait()
                if close_event.is_set():
                    break


                input_ids = shared_memory.input_tensor[1:shared_memory.input_tensor[0].item() + 1].unsqueeze(0).to(config.device)
                verify_state = step(verify_model, input_ids)
                
                local_index = 0
                while verify_state["output_ids"].size(1) < max_new_tokens:
                    start_time = time.time()

                    # inform draft process of new verify token(s)
                    output_ids = verify_state["output_ids"]
                    shared_memory.verify_tensor[1:1+output_ids.size(1)] = output_ids
                    shared_memory.verify_tensor[0] = output_ids.size(1)

                    # rollback check if necessary
                    while shared_memory.rollback_requests[0] > shared_memory.rollback_responses[0]:
                        time.sleep(0.0001)

                    while shared_memory.draft_tensor[0].item() <= shared_memory.verify_tensor[0].item() + 1:
                        time.sleep(0.0001)

                    draft_length = shared_memory.draft_tensor[0].item()
                    draft_output_ids = shared_memory.draft_tensor[1:draft_length+1].unsqueeze(0)
                    num_old_verified_tokens = verify_state["output_ids"].size(1)

                    verify_state = build_verify_inputs(draft_output_ids, **verify_state)

                    num_draft_tokens = verify_state["input_ids"].size(1)

                    verify_state = verify_step(verify_model, **verify_state)
                    
                    # Not all draft tokens were accepted, so draft process should rollback to the last verified token
                    num_new_verified_tokens = verify_state["output_ids"].size(1) - num_old_verified_tokens
                    if num_new_verified_tokens != num_draft_tokens:
                        shared_memory.rollback_requests[0] += 1

                    end_time = time.time()

                    # Log metrics
                    idx = local_index % max_log_entries
                    shared_memory.verify_times[idx] = end_time - start_time
                    shared_memory.accepted_tokens[idx] = num_new_verified_tokens
                    local_index += 1
                    
                    # Check only newly generated tokens for eos_token_id
                    if tok.eos_token_id in verify_state["output_ids"][0, -num_new_verified_tokens:]:
                        # find eos_token_id
                        eos_index = (verify_state["output_ids"][0] == tok.eos_token_id).nonzero(as_tuple=True)[0][-1].item()
                        # trim output_ids
                        verify_state["output_ids"] = verify_state["output_ids"][:, :eos_index+1]
                        break

                
                shared_memory.verify_tensor[0] = verify_state["output_ids"].size(1)
                shared_memory.verify_tensor[1:1+shared_memory.verify_tensor[0].item()] = verify_state["output_ids"]
                shared_memory.verify_index[0] = local_index
                start_event.clear()
                completion_event.set()
        except Exception as e:
            print(f"Exception in verify process: {e}")

    def generate(self, input_ids, return_metrics=False):
        # Send input_ids to shared memory
        input_length = input_ids.size(1)
        self.shared_memory.input_tensor[0] = input_length
        self.shared_memory.input_tensor[1:1+input_length] = input_ids[0]
        
        # Clear draft and verify tensors
        self.shared_memory.draft_tensor.fill_(0)
        self.shared_memory.verify_tensor.fill_(0)

        # Start the processes
        self.completion_event.clear()
        self.start_event.set()

        # Running the processes in parallel
        start_time = time.time()
        self.completion_event.wait()
        total_time = time.time() - start_time

        # wait for draft reset event
        self.draft_reset_event.wait()
        self.draft_reset_event.clear()

        final_output_ids = self.shared_memory.verify_tensor[1:self.shared_memory.verify_tensor[0].item()+1].clone()
        tokens_generated = len(final_output_ids)
        time_per_token = total_time / tokens_generated

        # Process logged metrics
        num_draft_logs = min(self.shared_memory.draft_index[0].item(), self.max_log_entries)
        num_verify_logs = min(self.shared_memory.verify_index[0].item(), self.max_log_entries)
        draft_times = self.shared_memory.draft_times[:num_draft_logs].tolist()
        verify_times = self.shared_memory.verify_times[:num_verify_logs].tolist()
        accepted_tokens = self.shared_memory.accepted_tokens[:num_verify_logs].tolist()

        metrics = {
            "draft_times": draft_times,
            "verify_times": verify_times,
            "accepted_tokens": accepted_tokens,
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "time_per_token": time_per_token
        }

        # Reset shared memory and indices
        self._reset_shared_memory()

        if return_metrics:
            return final_output_ids, metrics

        return final_output_ids

    def _reset_shared_memory(self):
        for tensor in vars(self.shared_memory).values():
            tensor.fill_(0)

    def close(self):
        self.close_event.set()
        self.start_event.set()
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
