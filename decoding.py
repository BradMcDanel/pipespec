import time
import traceback
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List

import torch.multiprocessing as mp
from transformers import BitsAndBytesConfig, DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()


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

def step(model, input_ids, output_ids=None, past_key_values=None, num_input_tokens=-1, **kwargs):
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

def build_verify_inputs(draft_output_ids, input_ids, output_ids, past_key_values, num_input_tokens, **kwargs):
    draft_ids = draft_output_ids.to(output_ids.device)

    # Get lengths of both tensors
    verify_len = output_ids.size(1)
    draft_len = draft_ids.size(1)

    # Case 1: Valid state is longer or tokens are different (we don't take the draft tokens)
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

def build_draft_inputs(verify_output_ids, input_ids, output_ids, past_key_values, num_input_tokens, **kwargs):
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
    
    return output_ids, input_ids, past_key_values, verified_ids.size(1)


def verify_step(model, input_ids, output_ids, past_key_values, num_input_tokens, **kwargs):
    num_input_ids = input_ids.size(1)

    # (1) Model forward pass
    with torch.no_grad():
        verify_outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    
    # (2) Verification and update process
    if input_ids.size(1) > 1:
        output_ids, input_ids, past_key_values, num_verified_tokens = perform_verification(
            input_ids, verify_outputs, output_ids, past_key_values, num_input_tokens
        )
    else:
        next_token_logits = verify_outputs.logits[:, -1, :]
        input_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        output_ids = torch.cat([output_ids, input_ids], dim=1)
        num_verified_tokens = 1

    return {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "past_key_values": past_key_values,
        "num_input_tokens": num_input_tokens,
        "all_tokens_were_verified": num_verified_tokens == num_input_ids,
        "num_verified_tokens": num_verified_tokens
    }

def put_sm_tensor(src_tensor, sm_tensor, lock):
    with lock:
        src_tensor = src_tensor.squeeze(0)
        sm_tensor[0] = src_tensor.size(0)
        sm_tensor[1:src_tensor.size(0)+1] = src_tensor

def get_sm_tensor(sm_tensor, lock):
    with lock:
        return sm_tensor[1:sm_tensor[0].item()+1].unsqueeze(0)

def update_log(times_tensor, index_tensor, idx, elapsed_time):
    if idx >= times_tensor.size(0):
        raise ValueError("Exceeded maximum log entries")
    times_tensor[idx] = elapsed_time
    idx += 1
    index_tensor[0] = idx

@dataclass
class ModelConfig:
    model_path: str
    devices: Union[str, List[str]]  # Can be "auto", "cpu", "cuda:0" or ["cuda:0", "cuda:1", ...]
    dtype: torch.dtype
    quantization_config: Optional[BitsAndBytesConfig] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
            
        # Handle device mapping
        if isinstance(self.devices, str):
            if self.devices == "auto":
                self.model_kwargs["device_map"] = "auto"
            elif self.devices == "cpu":
                self.model_kwargs["device_map"] = "cpu"
            else:
                # Single CUDA device
                gpu_id = int(self.devices.split(':')[1])
                self.model_kwargs["device_map"] = gpu_id
        else:
            # Multiple devices specified
            if len(self.devices) > 1:
                self.model_kwargs["device_map"] = "auto"
                # Convert cuda:X to integer indices for GPUs
                max_memory = {}
                for device in self.devices:
                    if device.startswith('cuda:'):
                        gpu_id = int(device.split(':')[1])
                        max_memory[gpu_id] = "40GiB"
                    elif device == "cpu":
                        max_memory["cpu"] = "30GiB"
                
                # Always include some CPU memory
                if "cpu" not in max_memory:
                    max_memory["cpu"] = "30GiB"
                
                self.model_kwargs["max_memory"] = max_memory
            else:
                # Single device in list format
                device = self.devices[0]
                if device.startswith('cuda:'):
                    gpu_id = int(device.split(':')[1])
                    self.model_kwargs["device_map"] = gpu_id
                else:
                    self.model_kwargs["device_map"] = device
            
        # Set up dtype
        self.model_kwargs["torch_dtype"] = self.dtype
        
        # Add quantization config if provided
        if self.quantization_config is not None:
            self.model_kwargs["quantization_config"] = self.quantization_config

    @property
    def device(self) -> str:
        """Return the primary device for tensor operations.
        For multi-GPU setups, returns the first device."""
        if isinstance(self.devices, str):
            if self.devices == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            return self.devices
        else:
            return self.devices[0]

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get all kwargs for model initialization"""
        return self.model_kwargs.copy()

def init_model(model_config: ModelConfig):
    """Initialize model and tokenizer with the given configuration"""
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        **model_config.get_model_kwargs()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
    
    return model, tokenizer

def create_4bit_config(
    compute_dtype: torch.dtype = torch.float16,
    quant_type: str = "nf4",
    double_quant: bool = True
) -> BitsAndBytesConfig:
    """Helper function to create 4-bit quantization config"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant
    )

def create_8bit_config() -> BitsAndBytesConfig:
    """Helper function to create 8-bit quantization config"""
    return BitsAndBytesConfig(
        load_in_8bit=True
    )

class GreedyDecoder:
    def __init__(self, model_config, max_new_tokens=4096):
        self.model_config = model_config
        self.max_new_tokens = max_new_tokens
        self._init_model()

    def _init_model(self):
        self.model, self.tokenizer = init_model(self.model_config)

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

class ChainSpeculativeDecoder:
    def __init__(self, model_configs: List[ModelConfig], 
                 draft_lookahead: int = 5,
                 max_new_tokens: int = 4096):
        if len(model_configs) < 2:
            raise ValueError("Need at least 2 models in chain")
            
        self.model_configs = model_configs
        self.draft_lookahead = draft_lookahead
        self.max_new_tokens = max_new_tokens
        self.num_positions = len(model_configs)
        
        self._init_models()
        
    def _init_models(self):
        self.models = []
        self.tokenizer = None
        
        for config in self.model_configs:
            model, tokenizer = init_model(config)
            self.models.append(model)
            if self.tokenizer is None:
                self.tokenizer = tokenizer

    @staticmethod
    def update_states(states, output_ids):
        for i in range(len(states)-1):
            states[i] = build_draft_inputs(output_ids, **states[i])

        return states
    
    def generate(self, input_ids, return_metrics=False):
        metrics = {
            "model_times": [[] for _ in range(self.num_positions)],
            "accepted_tokens": [[] for _ in range(self.num_positions - 1)],
            "total_time": 0,
            "tokens_generated": 0
        }
        
        start_time = time.time()
        
        # Initialize states
        states = []
        for i, model in enumerate(self.models):
            outputs = step(model, input_ids)
            states.append({
                "input_ids": outputs["input_ids"],
                "output_ids": outputs["output_ids"],
                "past_key_values": outputs["past_key_values"],
                "num_input_tokens": outputs["num_input_tokens"]
            })

        states = self.update_states(states, states[-1]["output_ids"])
        
        while states[-1]["output_ids"].size(1) < self.max_new_tokens:
            # Draft phase for first model
            draft_token_start = time.time()
            while states[0]["output_ids"].size(1) - states[1]["output_ids"].size(1) < self.draft_lookahead - 1:
                states[0] = step(self.models[0], **states[0])
                metrics["model_times"][0].append(time.time() - draft_token_start)
                draft_token_start = time.time()
            
            # Verify and draft through middle models
            for i in range(1, self.num_positions - 1):
                draft_start = time.time()
                
                # Verify tokens from previous model
                states[i] = build_verify_inputs(states[i-1]["output_ids"], **states[i])
                states[i] = verify_step(self.models[i], **states[i])
                
                metrics["accepted_tokens"][i-1].append(states[i]["num_verified_tokens"])
            
                # Generate new draft tokens
                while states[i]["output_ids"].size(1) - states[i+1]["output_ids"].size(1) < self.draft_lookahead - 1:
                    states[i] = step(self.models[i], **states[i])

                draft_time = time.time() - draft_start
                metrics["model_times"][i].append(draft_time)

            
            # Final verification
            verify_start = time.time()
            states[-1] = build_verify_inputs(states[-2]["output_ids"], **states[-1])
            states[-1] = verify_step(self.models[-1], **states[-1])

            states = self.update_states(states, states[-1]["output_ids"])
            verify_time = time.time() - verify_start

            metrics["model_times"][-1].append(verify_time)
            metrics["accepted_tokens"][-1].append(states[-1]["num_verified_tokens"])

        
            # Check for EOS
            if self.tokenizer.eos_token_id in states[-1]["output_ids"]:
                eos_index = (states[-1]["output_ids"][0] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][-1].item()
                states[-1]["output_ids"] = states[-1]["output_ids"][:, :eos_index+1]
                break
        
        if return_metrics:
            total_time = time.time() - start_time
            tokens_generated = states[-1]["output_ids"].size(1)
            metrics.update({
                "total_time": total_time,
                "tokens_generated": tokens_generated,
                "time_per_token": total_time / tokens_generated
            })
            return states[-1]["output_ids"][0], metrics
            
        return states[-1]["output_ids"][0]

@dataclass
class ChainEvents:
    """Event system for n-model chain"""
    def __init__(self, manager: mp.Manager, num_positions: int):
        self.start = manager.Event()
        self.completion = manager.Event()
        self.close = manager.Event()
        self.ready = [manager.Event() for _ in range(num_positions)]
        self.lock = manager.Lock()
        
    def wait_all_ready(self):
        for event in self.ready:
            event.wait()
            
@dataclass
class SharedChainMemory:
    """Shared memory for n-model chain"""
    input_tensor: torch.Tensor
    chain_tensors: List[torch.Tensor]  # One tensor per model in chain
    sync_requests: torch.Tensor    # Size (n-1) for n models, tracks requests between pairs  
    sync_responses: torch.Tensor   # Size (n-1) for n models, tracks responses between pairs
    execution_times: torch.Tensor      # Track execution time per model
    accepted_tokens: torch.Tensor      # Track accepted tokens per verification
    position_indices: torch.Tensor     # Track indices per position

class AsyncChainSpeculativeDecoder:
    """N-model chain-based speculative decoder"""
    def __init__(self, model_configs: List[ModelConfig],
                 draft_lookahead: int = 5,
                 max_new_tokens: int = 4096, 
                 max_length: int = 8096,
                 max_log_entries: int = 100000):
        self.model_configs = model_configs
        self.draft_lookahead = draft_lookahead
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.max_log_entries = max_log_entries
        self.num_positions = len(self.model_configs)
        
        if self.num_positions < 2:
            raise ValueError("Need at least 2 models in chain")
            
        self._init_shared_memory()
        self._init_multiprocessing()
    
    def _init_shared_memory(self):
        self.shared_memory = SharedChainMemory(
            input_tensor=torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_(),
            chain_tensors=[
                torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_() 
                for _ in range(self.num_positions)
            ],
            sync_requests=torch.zeros(self.num_positions - 1, dtype=torch.int32).share_memory_(),
            sync_responses=torch.zeros(self.num_positions - 1, dtype=torch.int32).share_memory_(),
            execution_times=torch.zeros((self.num_positions, self.max_log_entries), 
                                     dtype=torch.float32).share_memory_(),
            accepted_tokens=torch.zeros((self.num_positions - 1, self.max_log_entries), 
                                     dtype=torch.int32).share_memory_(),
            position_indices=torch.zeros(self.num_positions, dtype=torch.int32).share_memory_()
        )

    def _init_multiprocessing(self):
        """Initialize multiprocessing components"""
        self.manager = mp.Manager()
        self.events = ChainEvents(self.manager, self.num_positions)
        self.pool = mp.Pool(processes=self.num_positions)
        self._start_processes()

    def _start_processes(self):
        """Start all chain processes"""
        self.processes = []
        for i, config in enumerate(self.model_configs):
            process = self.pool.apply_async(
                self._chain_process, 
                (i, config, self.shared_memory, self.events, 
                 self.max_new_tokens, self.num_positions, self.draft_lookahead)
            )
            self.processes.append(process)
        
        self.events.wait_all_ready()

    @staticmethod
    def _chain_process(position: int, model_config: ModelConfig,
                      sm: SharedChainMemory, events: ChainEvents, 
                      max_new_tokens: int, num_positions: int,
                      draft_lookahead: int):
        """Chain process worker with graceful shutdown handling"""
        try:
            model, tokenizer = init_model(model_config)
            events.ready[position].set()
            
            while True:
                if events.close.wait(timeout=0.01):
                    break

                if not events.start.wait(timeout=0.01):
                    continue
                    
                local_index = 0
                
                # Initialize state with input
                input_ids = get_sm_tensor(sm.input_tensor, events.lock).to(model_config.device)
                state = step(model, input_ids)
                put_sm_tensor(state["output_ids"], sm.chain_tensors[position], events.lock)
                
                if position == 0:  # First model (pure draft)
                    while not events.completion.is_set() and not events.close.is_set():
                        start_time = time.time()

                        # Handle sync from next model
                        if sm.sync_requests[0] > sm.sync_responses[0]:
                            verify_output_ids = get_sm_tensor(sm.chain_tensors[position + 1], events.lock)
                            state = build_draft_inputs(
                                verify_output_ids,
                                state["output_ids"][:, -1:],
                                state["output_ids"],
                                state["past_key_values"],
                                state["num_input_tokens"]
                            )
                            put_sm_tensor(state["output_ids"], sm.chain_tensors[position], events.lock)
                            sm.sync_responses[0] += 1

                        state = step(model, 
                                   state["output_ids"][:, -1:],
                                   state["output_ids"], 
                                   state["past_key_values"],
                                   state["num_input_tokens"])
                        put_sm_tensor(state["output_ids"], sm.chain_tensors[position], events.lock)

                        # Log execution time
                        elapsed_time = time.time() - start_time
                        update_log(sm.execution_times[position],
                                 sm.position_indices[position:position+1],
                                 local_index, elapsed_time)
                        local_index += 1
                        
                elif position < num_positions - 1:  # Middle models (verify and draft)
                    prev_pair = position - 1  # Index for previous model pair
                    next_pair = position      # Index for next model pair
                    
                    while not events.completion.is_set() and not events.close.is_set():
                        start_time = time.time()
                        ran_step = False
                        
                        # Wait for previous model's sync completion if needed
                        while sm.sync_requests[prev_pair] > sm.sync_responses[prev_pair]:
                            if events.close.is_set():
                                return
                            time.sleep(0.0001)

                        # Get current lengths
                        curr_length = sm.chain_tensors[position][0]
                        prev_length = sm.chain_tensors[position - 1][0]
                            
                        # If we got a sync request from next model
                        if sm.sync_requests[next_pair] > sm.sync_responses[next_pair]:
                            verify_output_ids = get_sm_tensor(sm.chain_tensors[position + 1], events.lock)
                            state = build_draft_inputs(
                                verify_output_ids,
                                state["output_ids"][:, -1:],
                                state["output_ids"],
                                state["past_key_values"],
                                state["num_input_tokens"]
                            )
                            put_sm_tensor(state["output_ids"], sm.chain_tensors[position], events.lock)
                            sm.sync_responses[next_pair] += 1

                            # send sync request to previous model
                            sm.sync_requests[prev_pair] += 1
                            while sm.sync_requests[prev_pair] > sm.sync_responses[prev_pair]:
                                time.sleep(0.0001)
                        
                        if curr_length <= prev_length - draft_lookahead + 1:
                            ran_step = True
                            draft_output_ids = get_sm_tensor(sm.chain_tensors[position - 1], events.lock)
                            verify_inputs = build_verify_inputs(
                                draft_output_ids,
                                state["output_ids"][:, -1:],
                                state["output_ids"],
                                state["past_key_values"],
                                state["num_input_tokens"]
                            )
                            state = verify_step(model, **verify_inputs)
                            put_sm_tensor(state["output_ids"], sm.chain_tensors[position], events.lock)
                            
                            # Request sync from previous model if verification failed
                            sm.sync_requests[prev_pair] += 1
                                
                            # Log verification metrics
                            sm.accepted_tokens[position-1, local_index] = state["num_verified_tokens"]
                        
                        # Log execution time
                        if ran_step:
                            elapsed_time = time.time() - start_time
                            update_log(sm.execution_times[position],
                                     sm.position_indices[position:position+1],
                                     local_index, elapsed_time)
                            local_index += 1
                else:  # Last model (pure verify)
                    prev_pair = position - 1
                    while state["output_ids"].size(1) < max_new_tokens:
                        start_time = time.time()
                        ran_step = False
                        
                        # Wait for previous model's sync completion with timeout
                        while sm.sync_requests[prev_pair] > sm.sync_responses[prev_pair]:
                            if events.close.is_set():
                                return
                            time.sleep(0.0001)
                                
                        # Get current lengths
                        curr_length = sm.chain_tensors[position][0]
                        prev_length = sm.chain_tensors[prev_pair][0]
                            
                        # Only verify if we have new draft tokens
                        if curr_length <= prev_length - draft_lookahead + 1:
                            # Verify tokens from previous model
                            draft_output_ids = get_sm_tensor(sm.chain_tensors[position - 1], events.lock)
                            verify_inputs = build_verify_inputs(
                                draft_output_ids,
                                state["output_ids"][:, -1:],
                                state["output_ids"],
                                state["past_key_values"],
                                state["num_input_tokens"]
                            )
                            ran_step = True
                            state = verify_step(model, **verify_inputs)
                            put_sm_tensor(state["output_ids"], sm.chain_tensors[position], events.lock)

                            # Request sync
                            sm.sync_requests[prev_pair] += 1
                                
                            # Log verification metrics
                            sm.accepted_tokens[position-1, local_index] = state["num_verified_tokens"]
                            
                            # Check for EOS token in newly verified tokens
                            if tokenizer.eos_token_id in state["output_ids"][0, -state["num_verified_tokens"]:]:
                                eos_index = (state["output_ids"][0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][-1].item()
                                state["output_ids"] = state["output_ids"][:, :eos_index+1]
                                put_sm_tensor(state["output_ids"], sm.chain_tensors[position], events.lock)
                                break
                        
                        # Log execution time
                        if ran_step:
                            elapsed_time = time.time() - start_time
                            update_log(sm.execution_times[position],
                                     sm.position_indices[position:position+1],
                                     local_index, elapsed_time)
                            local_index += 1

                # Send completion signal
                if position == num_positions - 1:
                    events.completion.set()
                    
        except Exception as e:
            print(f"Exception in chain process at position {position}: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            try:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def generate(self, input_ids, return_metrics=False):
        self._reset_shared_memory()
        
        # Setup input
        input_length = input_ids.size(1)
        self.shared_memory.input_tensor[0] = input_length
        self.shared_memory.input_tensor[1:1+input_length] = input_ids[0]
        
        # Start generation
        self.events.completion.clear()
        self.events.start.set()
        
        # Wait for completion and measure time
        start_time = time.time()
        self.events.completion.wait()
        total_time = time.time() - start_time
        self.events.start.clear()
        
        # Get final output from last model
        final_output = get_sm_tensor(self.shared_memory.chain_tensors[-1], self.events.lock)[0]
        tokens_generated = len(final_output)
        time_per_token = total_time / tokens_generated if tokens_generated > 0 else 0
        
        # Format metrics to match baseline expectations
        metrics = {
            "model_times": [
                self.shared_memory.execution_times[i][:self.shared_memory.position_indices[i]].tolist()
                for i in range(self.num_positions)
            ],
            "accepted_tokens": [
                self.shared_memory.accepted_tokens[i][:self.shared_memory.position_indices[i+1]].tolist()
                for i in range(self.num_positions - 1)
            ],
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "time_per_token": time_per_token
        }
        
        if return_metrics:
            return final_output, metrics
            
        return final_output

    def _reset_shared_memory(self):
        """Reset all shared memory tensors"""
        self.shared_memory.input_tensor.fill_(0)
        for tensor in self.shared_memory.chain_tensors:
            tensor.fill_(777)
        self.shared_memory.sync_requests.fill_(0)
        self.shared_memory.sync_responses.fill_(0)
        self.shared_memory.execution_times.fill_(0)
        self.shared_memory.accepted_tokens.fill_(0)
        self.shared_memory.position_indices.fill_(0)
    
    def close(self):
        """Clean up multiprocessing resources"""
        self.events.close.set()
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
