import time
import traceback
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List

import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
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
    
    return output_ids, input_ids, past_key_values, verified_ids.size(1)


def verify_step(model, input_ids, output_ids, past_key_values, num_input_tokens, **kwargs):
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
        "all_tokens_were_verified": num_verified_tokens == input_ids.size(1),
        "num_verified_tokens": num_verified_tokens
    }

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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
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
        self.draft_model, _ = init_model(self.draft_config)
        self.verify_model, self.tokenizer = init_model(self.verify_config)

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

@dataclass
class ModelState:
    """Tracks the state of each model in the chain"""
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    past_key_values: Optional[DynamicCache] = None
    num_input_tokens: int = -1
    
    @property
    def last_token(self):
        return self.output_ids[:, -1:]
    
    @property
    def num_tokens(self):
        return self.output_ids.size(1)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ModelState':
        return cls(**d)

class ThreeStageSpeculativeDecoder:
    def __init__(self, small_config: ModelConfig, 
                 medium_config: ModelConfig,
                 large_config: ModelConfig,
                 lookahead: int = 4,
                 max_new_tokens: int = 4096):
        self.lookahead = lookahead
        self.max_new_tokens = max_new_tokens
        
        # Initialize models largest to smallest to share tokenizer
        self.large_model, self.tokenizer = init_model(large_config)
        self.medium_model, _ = init_model(medium_config)
        self.small_model, _ = init_model(small_config)
        
    def _small_draft_step(self, state: ModelState) -> ModelState:
        """Generate next token with small model"""
        outputs = step(self.small_model, state.input_ids, state.output_ids,
                      state.past_key_values, state.num_input_tokens)
        return ModelState.from_dict(outputs)
    
    def _medium_verify_and_draft(self, state: ModelState, draft_ids: torch.Tensor) -> tuple[ModelState, int]:
        """Verify small model tokens and generate medium model drafts"""
        # First verify the draft tokens
        verify_state_dict = build_verify_inputs(draft_ids, state.input_ids, 
                                              state.output_ids, state.past_key_values,
                                              state.num_input_tokens)
        verify_state = ModelState.from_dict(verify_state_dict)
        
        num_draft_tokens = verify_state.input_ids.size(1)
        old_verified_len = verify_state.output_ids.size(1)
        
        verify_outputs = verify_step(self.medium_model, 
                                   verify_state.input_ids,
                                   verify_state.output_ids,
                                   verify_state.past_key_values,
                                   verify_state.num_input_tokens)
        verify_state = ModelState.from_dict(verify_outputs)
        
        num_accepted = verify_state.output_ids.size(1) - old_verified_len
        
        # Draft additional tokens if we verified successfully
        if num_accepted > 0:
            outputs = step(self.medium_model, 
                         verify_state.input_ids,
                         verify_state.output_ids,
                         verify_state.past_key_values,
                         verify_state.num_input_tokens)
            verify_state = ModelState.from_dict(outputs)
            
        return verify_state, num_accepted
        
    def _large_verify(self, state: ModelState, draft_ids: torch.Tensor) -> tuple[ModelState, int]:
        """Verify tokens from medium model"""
        verify_state_dict = build_verify_inputs(draft_ids, state.input_ids,
                                              state.output_ids, state.past_key_values,
                                              state.num_input_tokens)
        verify_state = ModelState.from_dict(verify_state_dict)
        
        old_verified_len = verify_state.output_ids.size(1)
        
        verify_outputs = verify_step(self.large_model, 
                                   verify_state.input_ids,
                                   verify_state.output_ids,
                                   verify_state.past_key_values,
                                   verify_state.num_input_tokens)
        verify_state = ModelState.from_dict(verify_outputs)
        
        return verify_state, verify_state.output_ids.size(1) - old_verified_len

    def generate(self, input_ids: torch.Tensor, return_metrics: bool = False):
        # Initialize states
        small_state = ModelState.from_dict(step(self.small_model, input_ids))
        medium_state = ModelState.from_dict(step(self.medium_model, input_ids))
        large_state = ModelState.from_dict(step(self.large_model, input_ids))
        
        start_time = time.time()
        metrics = {
            "small_times": [],
            "medium_times": [],
            "large_times": [],
            "medium_accepted_tokens": [],
            "large_accepted_tokens": []
        }
        
        while large_state.num_tokens < self.max_new_tokens:
            # Small model generates lookahead tokens
            small_start = time.time()
            while small_state.num_tokens < medium_state.num_tokens + self.lookahead:
                small_state = self._small_draft_step(small_state)
            metrics["small_times"].append(time.time() - small_start)
            
            # Medium model verifies and drafts
            medium_start = time.time()
            medium_state, num_medium_accepted = self._medium_verify_and_draft(
                medium_state, small_state.output_ids
            )
            metrics["medium_times"].append(time.time() - medium_start)
            metrics["medium_accepted_tokens"].append(num_medium_accepted)
            
            # If medium didn't accept enough tokens, generate more from small
            while num_medium_accepted < self.lookahead:
                small_start = time.time()
                small_state = self._small_draft_step(small_state)
                metrics["small_times"].append(time.time() - small_start)
                
                medium_start = time.time()
                medium_state, additional_accepted = self._medium_verify_and_draft(
                    medium_state, small_state.output_ids
                )
                metrics["medium_times"].append(time.time() - medium_start)
                metrics["medium_accepted_tokens"].append(additional_accepted)
                num_medium_accepted += additional_accepted
            
            # Large model verifies medium's tokens
            large_start = time.time()
            large_state, num_large_accepted = self._large_verify(
                large_state, medium_state.output_ids
            )
            metrics["large_times"].append(time.time() - large_start)
            metrics["large_accepted_tokens"].append(num_large_accepted)
            
            # Check for EOS token in newly verified tokens
            if self.tokenizer.eos_token_id in large_state.output_ids[0, -num_large_accepted:]:
                eos_idx = (large_state.output_ids[0] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][-1].item()
                large_state.output_ids = large_state.output_ids[:, :eos_idx + 1]
                break
            
            # Update small and medium states based on verifications
            if num_large_accepted < num_medium_accepted:
                # Reset medium state to match large's verified tokens
                medium_state_dict = build_draft_inputs(
                    large_state.output_ids, 
                    medium_state.input_ids,
                    medium_state.output_ids,
                    medium_state.past_key_values,
                    medium_state.num_input_tokens
                )
                medium_state = ModelState.from_dict(medium_state_dict)
                
                # Reset small state to match medium's verified tokens
                small_state_dict = build_draft_inputs(
                    medium_state.output_ids,
                    small_state.input_ids,
                    small_state.output_ids,
                    small_state.past_key_values,
                    small_state.num_input_tokens
                )
                small_state = ModelState.from_dict(small_state_dict)
        
        total_time = time.time() - start_time
        tokens_generated = large_state.output_ids.size(1)
        
        metrics.update({
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "time_per_token": total_time / tokens_generated
        })
        
        if return_metrics:
            return large_state.output_ids[0], metrics
            
        return large_state.output_ids[0]

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

def put_sm_tensor(src_tensor, sm_tensor):
    src_tensor = src_tensor.squeeze(0)
    sm_tensor[0] = src_tensor.size(0)
    sm_tensor[1:src_tensor.size(0)+1] = src_tensor

def get_sm_tensor(sm_tensor):
    return sm_tensor[1:sm_tensor[0].item()+1].unsqueeze(0)

def update_log(times_tensor, index_tensor, idx, elapsed_time):
    if idx >= times_tensor.size(0):
        raise ValueError("Exceeded maximum log entries")
    times_tensor[idx] = elapsed_time
    idx += 1
    index_tensor[0] = idx

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
        self.draft_process = self.pool.apply_async(self._draft_process, (
            self.draft_config, self.shared_memory,
            self.draft_ready_event, self.draft_reset_event, self.start_event, self.completion_event, 
            self.close_event
        ))
        self.verify_process = self.pool.apply_async(self._verify_process, (
            self.verify_config, self.shared_memory,
            self.verify_ready_event, self.start_event, self.completion_event, 
            self.close_event, self.max_new_tokens
        ))

        self.draft_ready_event.wait()
        self.verify_ready_event.wait()

    @staticmethod
    def _draft_process(config: ModelConfig, sm: SharedMemory, 
                       ready_event, reset_event, start_event, completion_event, close_event):
        try:
            draft_model, _ = init_model(config)
            ready_event.set()

            while True:
                start_event.wait()
                if close_event.is_set():
                    break

                # Initialize draft state
                input_ids = get_sm_tensor(sm.input_tensor).to(config.device)
                draft_state = step(draft_model, input_ids)
                
                # Write to shared memory
                put_sm_tensor(draft_state["output_ids"], sm.draft_tensor)
            
                local_index = 0
                handled_rollback = False
                while not completion_event.is_set():
                    start_time = time.time()

                    # Rollback Triggered: update draft state with verify tokens
                    if sm.rollback_requests[0] > sm.rollback_responses[0] and not handled_rollback:
                        verify_output_ids = get_sm_tensor(sm.verify_tensor)
                        draft_state = build_draft_inputs(verify_output_ids, **draft_state)
                        handled_rollback = True

                    # Compute next draft tokens
                    draft_state = step(draft_model, **draft_state)
                    put_sm_tensor(draft_state["output_ids"], sm.draft_tensor)

                    # Rollback Complete: signal verify process that rollback is complete
                    if sm.rollback_requests[0] > sm.rollback_responses[0] and handled_rollback:
                        sm.rollback_responses[0] += 1
                        handled_rollback = False
                
                    end_time = time.time()

                    # Log draft time
                    elapsed_time = end_time - start_time
                    update_log(sm.draft_times, sm.draft_index, local_index, elapsed_time)
                    local_index += 1

                sm.draft_index[0] = local_index
                reset_event.set()
        except Exception as e:
            print(f"Exception in draft process: {e}")
            traceback.print_exc()

    @staticmethod
    def _verify_process(config: ModelConfig, sm: SharedMemory, 
                        ready_event, start_event, completion_event, close_event, 
                        max_new_tokens):
        try:
            verify_model, tok = init_model(config)
            ready_event.set()
            
            while True:
                start_event.wait()
                if close_event.is_set():
                    break

                # Initialize verify state
                input_ids = get_sm_tensor(sm.input_tensor).to(config.device)
                verify_state = step(verify_model, input_ids)

                # Write to shared memory
                put_sm_tensor(verify_state["output_ids"], sm.verify_tensor)
                
                local_index = 0
                while verify_state["output_ids"].size(1) < max_new_tokens:
                    start_time = time.time()

                    # Wait for draft to confirm rollback
                    while sm.rollback_requests[0] > sm.rollback_responses[0]:
                        time.sleep(0.001)

                    # Get draft tokens
                    draft_output_ids = get_sm_tensor(sm.draft_tensor)
                    verify_state = build_verify_inputs(draft_output_ids, **verify_state)
                    verify_state = verify_step(verify_model, **verify_state)

                    # Write to shared memory
                    put_sm_tensor(verify_state["output_ids"], sm.verify_tensor)

                    if not verify_state["all_tokens_were_verified"] or \
                        verify_state["num_verified_tokens"] == 1:
                        sm.rollback_requests[0] += 1

                    end_time = time.time()

                    # Log metrics
                    elapsed_time = end_time - start_time
                    update_log(sm.verify_times, sm.verify_index, local_index, elapsed_time)
                    sm.accepted_tokens[local_index] = verify_state["num_verified_tokens"]
                    local_index += 1
                    
                    # Check only newly generated tokens for eos_token_id
                    end_slice = verify_state["output_ids"][0, -verify_state["num_verified_tokens"]:]
                    if tok.eos_token_id in end_slice:
                        # find eos_token_id
                        eos_index = (verify_state["output_ids"][0] == tok.eos_token_id).nonzero(as_tuple=True)[0][-1].item()
                        # trim output_ids
                        verify_state["output_ids"] = verify_state["output_ids"][:, :eos_index+1]
                        break

                
                # Write final results to shared memory
                put_sm_tensor(verify_state["output_ids"], sm.verify_tensor)
                sm.verify_index[0] = local_index
                start_event.clear()
                completion_event.set()
        except Exception as e:
            print(f"Exception in verify process: {e}")
            traceback.print_exc()

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

@dataclass
class SharedChainMemory:
    """Shared memory matching baseline functionality"""
    input_tensor: torch.Tensor
    chain_tensors: List[torch.Tensor]  # Replaces draft/verify tensors
    rollback_requests: torch.Tensor    # Keep counter-based approach
    rollback_responses: torch.Tensor   # Keep counter-based approach
    execution_times: torch.Tensor      # Combined times tensor
    accepted_tokens: torch.Tensor
    position_indices: torch.Tensor     # Track indices per position

@dataclass
class ChainState:
    """Represents state for a single model in the chain"""
    curr: Optional[torch.Tensor] = None      # Current tensor being processed
    past_key_values: Optional[DynamicCache] = None  # KV cache for the model
    num_input_tokens: int = -1              # Number of input tokens processed
    
    @property
    def last_token(self):
        """Get the last token from current tensor"""
        return self.curr[:, -1:] if self.curr is not None else None
    
    @property
    def num_tokens(self):
        """Get number of tokens in current tensor"""
        return self.curr.size(1) if self.curr is not None else 0

    def step_output(self) -> Dict[str, Any]:
        """Convert state to step output format"""
        return {
            "output_ids": self.curr,
            "past_key_values": self.past_key_values,
            "num_input_tokens": self.num_input_tokens
        }

    @classmethod
    def from_step_output(cls, outputs: Dict[str, Any]) -> 'ChainState':
        """Create state from step output"""
        return cls(
            curr=outputs["output_ids"],
            past_key_values=outputs["past_key_values"],
            num_input_tokens=outputs["num_input_tokens"]
        )

@dataclass
class ChainEvents:
    """Event system matching baseline behavior"""
    def __init__(self, manager: mp.Manager, num_positions: int):
        self.start = manager.Event()
        self.completion = manager.Event()
        self.close = manager.Event()
        self.ready = [manager.Event() for _ in range(num_positions)]
        self.reset = [manager.Event() for _ in range(num_positions)]
        
    def wait_all_ready(self):
        for event in self.ready:
            event.wait()
            
    def wait_reset(self, position: int):
        """Wait for position reset"""
        self.reset[position].wait()
        self.reset[position].clear()

class ChainSpeculativeDecoder:
    """Chain-based speculative decoder with improved event handling"""
    def __init__(self, draft_config: ModelConfig, verify_config: ModelConfig,
                 max_new_tokens: int = 4096, max_length: int = 8096,
                 max_log_entries: int = 10000):
        self.model_configs = [draft_config, verify_config]
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.max_log_entries = max_log_entries
        self.num_positions = len(self.model_configs)
        self._init_shared_memory()
        self._init_multiprocessing()
    
    def _init_shared_memory(self):
        self.shared_memory = SharedChainMemory(
            input_tensor=torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_(),
            chain_tensors=[torch.zeros(self.max_length + 1, dtype=torch.long).share_memory_() 
                          for _ in range(self.num_positions)],
            rollback_requests=torch.zeros(1, dtype=torch.int32).share_memory_(),
            rollback_responses=torch.zeros(1, dtype=torch.int32).share_memory_(),
            execution_times=torch.zeros((self.num_positions, self.max_log_entries), 
                                     dtype=torch.float32).share_memory_(),
            accepted_tokens=torch.zeros(self.max_log_entries, dtype=torch.int32).share_memory_(),
            position_indices=torch.zeros(self.num_positions, dtype=torch.int32).share_memory_()
        )

    def _init_multiprocessing(self):
        self.manager = mp.Manager()
        self.events = ChainEvents(self.manager, self.num_positions)
        self.pool = mp.Pool(processes=self.num_positions)
        self._start_processes()

    def _start_processes(self):
        self.processes = []
        for i, config in enumerate(self.model_configs):
            process = self.pool.apply_async(self._chain_process, 
                                          (i, config, self.shared_memory, self.events, self.max_new_tokens))
            self.processes.append(process)
        
        self.events.wait_all_ready()

    @staticmethod
    def _chain_process(position: int, model_config: ModelConfig,
                      sm: SharedChainMemory, events: ChainEvents, max_new_tokens: int):
        try:
            model, tokenizer = init_model(model_config)
            events.ready[position].set()
            
            while True:
                events.start.wait()
                if events.close.is_set():
                    break
                    
                state = ChainState()
                if position == 0:  # Draft
                    input_ids = get_sm_tensor(sm.input_tensor).to(model_config.device)
                    outputs = step(model, input_ids)
                    state.curr = outputs["output_ids"]
                    state.past_key_values = outputs["past_key_values"]
                    state.num_input_tokens = outputs["num_input_tokens"]
                    
                    put_sm_tensor(state.curr, sm.chain_tensors[position])
                    
                    local_index = 0
                    handled_rollback = False
                    while not events.completion.is_set():
                        start_time = time.time()

                        # Handle rollback using counter-based approach
                        if sm.rollback_requests[0] > sm.rollback_responses[0] and not handled_rollback:
                            verify_output_ids = get_sm_tensor(sm.chain_tensors[position + 1])
                            # Moved _handle_rollback logic here
                            draft_inputs = build_draft_inputs(
                                verify_output_ids,
                                state.curr[:, -1:],
                                state.curr,
                                state.past_key_values,
                                state.num_input_tokens
                            )
                            state.curr = draft_inputs["output_ids"]
                            state.past_key_values = draft_inputs["past_key_values"]
                            handled_rollback = True

                        # Generate next token
                        outputs = step(model, state.curr[:, -1:],
                                     state.curr, state.past_key_values,
                                     state.num_input_tokens)
                        state.curr = outputs["output_ids"]
                        state.past_key_values = outputs["past_key_values"]
                        
                        put_sm_tensor(state.curr, sm.chain_tensors[position])

                        # Signal rollback completion
                        if sm.rollback_requests[0] > sm.rollback_responses[0] and handled_rollback:
                            sm.rollback_responses[0] += 1
                            handled_rollback = False
                        
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        update_log(sm.execution_times[position],
                                 sm.position_indices[position:position+1],
                                 local_index, elapsed_time)
                        local_index += 1

                    sm.position_indices[position] = local_index
                    events.reset[position].set()
                    
                else:  # Verify
                    input_ids = get_sm_tensor(sm.input_tensor).to(model_config.device)
                    outputs = step(model, input_ids)
                    state.curr = outputs["output_ids"]
                    state.past_key_values = outputs["past_key_values"]
                    state.num_input_tokens = outputs["num_input_tokens"]
                    
                    put_sm_tensor(state.curr, sm.chain_tensors[position])
                    
                    local_index = 0
                    while state.curr.size(1) < max_new_tokens: 
                        start_time = time.time()

                        # Wait for draft rollback completion
                        while sm.rollback_requests[0] > sm.rollback_responses[0]:
                            time.sleep(0.001)

                        # Get draft tokens and verify
                        draft_output_ids = get_sm_tensor(sm.chain_tensors[position-1])
                        verify_inputs = build_verify_inputs(draft_output_ids, 
                                                          state.curr[:, -1:],
                                                          state.curr,
                                                          state.past_key_values,
                                                          state.num_input_tokens)
                        verify_outputs = verify_step(model, **verify_inputs)
                        
                        state.curr = verify_outputs["output_ids"]
                        state.past_key_values = verify_outputs["past_key_values"]
                        
                        put_sm_tensor(state.curr, sm.chain_tensors[position])

                        # Request rollback if needed
                        if not verify_outputs["all_tokens_were_verified"] or \
                           verify_outputs["num_verified_tokens"] == 1:
                            sm.rollback_requests[0] += 1

                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        update_log(sm.execution_times[position],
                                 sm.position_indices[position:position+1],
                                 local_index, elapsed_time)
                        sm.accepted_tokens[local_index] = verify_outputs["num_verified_tokens"]
                        local_index += 1
                        
                        # Check for EOS token
                        if tokenizer.eos_token_id in state.curr[0, -verify_outputs["num_verified_tokens"]:]:
                            eos_index = (state.curr[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][-1].item()
                            state.curr = state.curr[:, :eos_index+1]
                            events.completion.set()
                            break
                    
                    sm.position_indices[position] = local_index
                
                events.start.clear()
                
        except Exception as e:
            print(f"Exception in chain process at position {position}: {e}")
            traceback.print_exc()


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
        
        # Wait for draft reset
        self.events.wait_reset(0)
        
        final_output = get_sm_tensor(self.shared_memory.chain_tensors[1])[0]
        tokens_generated = len(final_output)
        time_per_token = total_time / tokens_generated if tokens_generated > 0 else 0
        
        # Format metrics to match baseline
        metrics = {
            "draft_times": self.shared_memory.execution_times[0][:self.shared_memory.position_indices[0]].tolist(),
            "verify_times": self.shared_memory.execution_times[1][:self.shared_memory.position_indices[1]].tolist(),
            "accepted_tokens": self.shared_memory.accepted_tokens[:self.shared_memory.position_indices[1]].tolist(),
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
            tensor.fill_(0)
        self.shared_memory.rollback_requests.fill_(0)
        self.shared_memory.rollback_responses.fill_(0)
        self.shared_memory.execution_times.fill_(0)
        self.shared_memory.accepted_tokens.fill_(0)
        self.shared_memory.position_indices.fill_(0)
    
    def close(self):
        """Clean up multiprocessing resources"""
        self.events.close.set()
        self.events.start.set()
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
