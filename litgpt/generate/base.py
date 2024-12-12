# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, Optional, Tuple, List, Union, Iterator
import warnings
import datetime
import math

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning_utilities.core.imports import RequirementCache

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.prompts import has_prompt_style, load_prompt_style, PromptStyle
from litgpt.utils import (
    check_file_size_on_cpu_and_warn,
    check_valid_checkpoint_dir,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint
)


class bcolors:
    # Reset
    RESET = '\033[0m'

    # Regular Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bold Colors
    BLACK_BOLD = '\033[1;30m'
    RED_BOLD = '\033[1;31m'
    GREEN_BOLD = '\033[1;32m'
    YELLOW_BOLD = '\033[1;33m'
    BLUE_BOLD = '\033[1;34m'
    MAGENTA_BOLD = '\033[1;35m'
    CYAN_BOLD = '\033[1;36m'
    WHITE_BOLD = '\033[1;37m'

    # Underline Colors
    BLACK_UNDERLINED = '\033[4;30m'
    RED_UNDERLINED = '\033[4;31m'
    GREEN_UNDERLINED = '\033[4;32m'
    YELLOW_UNDERLINED = '\033[4;33m'
    BLUE_UNDERLINED = '\033[4;34m'
    MAGENTA_UNDERLINED = '\033[4;35m'
    CYAN_UNDERLINED = '\033[4;36m'
    WHITE_UNDERLINED = '\033[4;37m'

    # Background Colors
    BLACK_BACKGROUND = '\033[40m'
    RED_BACKGROUND = '\033[41m'
    GREEN_BACKGROUND = '\033[42m'
    YELLOW_BACKGROUND = '\033[43m'
    BLUE_BACKGROUND = '\033[44m'
    MAGENTA_BACKGROUND = '\033[45m'
    CYAN_BACKGROUND = '\033[46m'
    WHITE_BACKGROUND = '\033[47m'

    # High Intensity Colors
    BLACK_BRIGHT = '\033[90m'
    RED_BRIGHT = '\033[91m'
    GREEN_BRIGHT = '\033[92m'
    YELLOW_BRIGHT = '\033[93m'
    BLUE_BRIGHT = '\033[94m'
    MAGENTA_BRIGHT = '\033[95m'
    CYAN_BRIGHT = '\033[96m'
    WHITE_BRIGHT = '\033[97m'

    # High Intensity Background Colors
    BLACK_BACKGROUND_BRIGHT = '\033[100m'
    RED_BACKGROUND_BRIGHT = '\033[101m'
    GREEN_BACKGROUND_BRIGHT = '\033[102m'
    YELLOW_BACKGROUND_BRIGHT = '\033[103m'
    BLUE_BACKGROUND_BRIGHT = '\033[104m'
    MAGENTA_BACKGROUND_BRIGHT = '\033[105m'
    CYAN_BACKGROUND_BRIGHT = '\033[106m'
    WHITE_BACKGROUND_BRIGHT = '\033[107m'

    # Special Formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'

    # Keep your original naming conventions for compatibility
    HEADER = MAGENTA
    OKBLUE = BLUE
    OKCYAN = CYAN
    OKGREEN = GREEN
    WARNING = YELLOW
    FAIL = RED
    ENDC = RESET




def log(value, log_file='sample.log'):
    """
    Logs the given value to a file with a human-readable timestamp.

    Args:
    value: Any value to be logged
    log_file (str): The name of the log file (default is 'log.txt')
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message_str = str(value)

    # Create the log message
    log_message = f"[{timestamp}] {message_str}\n"

    # Ensure the directory exists
    # os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Write to the log file
    with open(log_file, 'a') as f:
        f.write(log_message)


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def crop_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Example:
        # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
        # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least 1 token always to prevent the case where no token is selected
        # In this case the most probable one is always kept
        sorted_indices_to_remove[-1:] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits

def crop_top_k(logits: torch.Tensor, top_k: float) -> torch.Tensor:
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)

    return logits

def crop_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = sorted_logits.softmax(dim=-1)
    
    # Find the index where probabilities fall below min_p
    cutoff_index = torch.where(probabilities < min_p)[0]
    if len(cutoff_index) > 0:
        cutoff_index = cutoff_index[0]
    else:
        cutoff_index = len(probabilities)
    
    # Create a mask for indices to keep
    indices_to_keep = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_keep[sorted_indices[:cutoff_index]] = True
    
    # Set logits of tokens below min_p to negative infinity
    logits = torch.where(indices_to_keep, logits, torch.full_like(logits, float('-inf')))
    
    return logits


def keep_highest_logits(logits: torch.Tensor, threshold_factor: float = 1.5) -> torch.Tensor:
    # Calculate the mean and standard deviation of the logits
    mean = torch.mean(logits)
    std = torch.std(logits)
    
    # Calculate the threshold
    threshold = mean + threshold_factor * std
    
    # Create a mask for indices to keep
    indices_to_keep = logits > threshold
    
    # Set logits below the threshold to negative infinity
    result = torch.where(indices_to_keep, logits, torch.full_like(logits, float('-inf')))
    
    return result


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature > 0.0:
        logits = logits / temperature
    return logits

def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: float = 1.0
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    #logits = logits[0, -1]

    logits = apply_temperature(logits,temperature)

    # optionally crop the logits to only the top k options
    logits = crop_top_k(logits,top_k)

    # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
    logits = crop_top_p(logits, top_p)


    if top_k > 0 or top_p > 0.0:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = multinomial_num_samples_1(probs)
    else:
        # argmax / greedy
        next_token =  torch.argmax(logits, dim=-1, keepdim=True)

    return next_token


def sample_adaptive(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: float = 1.0
) -> Tuple[torch.Tensor, float, float]: # token, entropy, var_entropy
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    #logits = logits[0, -1]


    # logits = apply_temperature(logits,temperature)

    # # optionally crop the logits to only the top k options
    # logits = crop_top_k(logits,top_k)

    # # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
    # logits = crop_top_p(logits, top_p)

    if top_k > 0 or top_p > 0.0:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = multinomial_num_samples_1(probs)
    else:
        # argmax / greedy
        next_token =  torch.argmax(logits, dim=-1, keepdim=True)

    entropy = calculate_entropy(logits).item()
    varentropy = calculate_varentropy(logits).item()

    return next_token, entropy, varentropy


def sample_greedy(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: float = 1.0
) -> Tuple[torch.Tensor, float, float]: # token, entropy, var_entropy
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    #logits = logits[0, -1]

    next_token =  torch.argmax(logits, dim=-1, keepdim=True)

    entropy = calculate_entropy(logits).item()
    varentropy = calculate_varentropy(logits).item()

    return next_token, entropy, varentropy


def calculate_varentropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate the varentropy (variance of entropy) of the given logits.

    Args:
    logits (torch.Tensor): Tensor of shape [batch_size, vocab_size]

    Returns:
    torch.Tensor: Varentropy values of shape [batch_size]
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10

    # Compute softmax with better numerical stability
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_exp = torch.exp(logits - max_logits)
    probs = logits_exp / torch.sum(logits_exp, dim=-1, keepdim=True)

    # Compute log probabilities
    log_probs = torch.log(probs + eps)

    # Calculate entropy
    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

    # Calculate varentropy
    varentropy = torch.sum(probs * (log_probs + entropy) ** 2, dim=-1)

    # Check for invalid values
    varentropy = torch.where(torch.isnan(varentropy), torch.zeros_like(varentropy), varentropy)

    # print("ve: ", varentropy)
    return varentropy


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:

    """
    Calculate the entropy of the given logits.

    Args:
    logits (torch.Tensor): Tensor of shape [batch_size, vocab_size]

    Returns:
    torch.Tensor: Entropy values of shape [batch_size]
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10

    # Compute softmax with better numerical stability
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_exp = torch.exp(logits - max_logits)
    probs = logits_exp / torch.sum(logits_exp, dim=-1, keepdim=True)

    # Compute log probabilities
    log_probs = torch.log(probs + eps)

    # Calculate entropy
    entropy = -torch.sum(probs * log_probs, dim=-1)

    # Check for invalid values
    entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)
    # print("e: ", entropy)
    return entropy


def apply_precision_limit(logits, vocab_size=128256)  -> torch.Tensor:
    # Constants
    u = 2**(-7)  # bfloat16 relative error
    min_logprob = math.log(u) - math.log(vocab_size) / 2
    
    # Convert min_logprob to the appropriate device and dtype
    min_logprob_tensor = torch.tensor(min_logprob, device=logits.device, dtype=logits.dtype)
    
    # Create a mask for logits below the threshold
    mask = logits < min_logprob_tensor
    
    # Set values below threshold to -inf
    logits[mask] = float('-inf')
    
    return logits


def scale_attn_entropy(logits,entropies,entropy_threshold,scale_factor)  -> torch.Tensor:

    ##### MORE CLAUDE 
    # entropy_threshold = 8.05

    # Calculate average entropy per head
    all_entropies = torch.stack(entropies)
    avg_entropy_per_head = all_entropies.mean(dim=(0, 1))  # Shape: (num_heads,)


    # scale based on attention head entropy
    # scale_factor = .25

    # Create a scaling factor for each head
    # Heads with entropy below the threshold will be scaled up, others scaled down
    head_scale = torch.where(avg_entropy_per_head >= entropy_threshold,
                             1 + scale_factor,
                             1 - scale_factor)
    
    # Ensure the scaling factor is positive
    head_scale = torch.clamp(head_scale, min=0.1)
    
    # Reshape logits to separate the head dimension
    B, T, V = logits.shape
    H = len(avg_entropy_per_head)
    logits_per_head = logits.view(B, T, H, V // H)

    # Apply the scaling to the logits
    scaled_logits = logits_per_head * head_scale.view(1, 1, H, 1)
    
    # Reshape back to original logits shape
    logits = scaled_logits.view(B, T, V)

    return logits


def crop_attn_entropy(logits,entropies,entropy_threshold) -> torch.Tensor:

    avg_entropy = torch.mean(torch.stack(entropies)).item()

    # Stack entropies and varentropies from all layers
    all_entropies = torch.stack(entropies)  # Shape: (num_layers, batch_size, num_heads)

    # Calculate average entropy per head
    all_entropies = torch.stack(entropies)
    avg_entropy_per_head = all_entropies.mean(dim=(0, 1))  # Shape: (num_heads,)
    
    # Create a mask for heads with entropy below the threshold
    head_mask = (avg_entropy_per_head <= entropy_threshold).float()
    
    # Reshape logits to separate the head dimension
    B, T, V = logits.shape
    H = len(avg_entropy_per_head)
    logits_per_head = logits.view(B, T, H, V // H)
    
    # Apply the mask to the logits
    masked_logits = logits_per_head * head_mask.view(1, 1, H, 1)
    
    # Reshape back to original logits shape
    logits = masked_logits.view(B, T, V)

    return logits

def next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    logits, entropies, varentropies = model(x, input_pos)

    avg_entropy = torch.mean(torch.stack(entropies)).item()
    avg_varentropy = torch.mean(torch.stack(varentropies)).item()

    log("--")
    # log(str(avg_entropy) + " - " + str(avg_varentropy) )

    log(f"Average Entropy: {avg_entropy:.4f}, Average Varentropy: {avg_varentropy:.4f}")
    
    # # Per-head analysis
    # num_heads = avg_entropy_per_head.size(0)
    # for head in range(num_heads):
    #     log(f"Head {head}: Entropy = {avg_entropy_per_head[head].item():.4f}, "
    #         f"Varentropy = {avg_varentropy_per_head[head].item():.4f}")
    ## end claude 

    entropy_all = calculate_entropy(logits[0, -1]).item()
    varentropy_all = calculate_varentropy(logits[0, -1]).item()
    log("ALL: " + str(entropy_all) + " " + str(varentropy_all))
    

    #logits = crop_attn_entropy(logits,entropies,8.35)
    logits = scale_attn_entropy(logits,entropies,8.2,0.6)
    log("SCALED: " + str(calculate_entropy(logits[0, -1]).item()) + " " + str(calculate_varentropy(logits[0, -1]).item()) )

    logits = logits[0, -1] # moved this out of sampler

    # entropy across all logits
    entropy_all = calculate_entropy(logits).item()
    varentropy_all = calculate_varentropy(logits).item()

    logits = keep_highest_logits(logits,2)
    log("PEAKS: " + str(calculate_entropy(logits).item()) + " " + str(calculate_varentropy(logits).item()))

    if calculate_entropy(logits).item() > 1:
        logits = apply_temperature(logits,.1)
    else:
        logits = apply_temperature(logits,.6)
    log("TEMP: " + str(calculate_entropy(logits).item()) + " " + str(calculate_varentropy(logits).item()))

    # logits = crop_min_p(logits,0.002)
    # log("MIN_P: " + str(calculate_entropy(logits).item()) + " " + str(calculate_varentropy(logits).item()))

    # # optionally crop the logits to only the top k options
    # logits = crop_top_k(logits,50)
    # log("TOP_K: " + str(calculate_entropy(logits).item()) + " " + str(calculate_varentropy(logits).item()))

    # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
    # logits = crop_top_p(logits, 0.8)
    # log("TOP_P: " + str(calculate_entropy(logits).item()) + " " + str(calculate_varentropy(logits).item()))



    if entropy_all < 0.1 and varentropy_all < 2 :
       _next, entropy, varentropy = sample_greedy(logits, **kwargs)
    else:
        _next, entropy, varentropy = sample_adaptive(logits, **kwargs)

    # log(str(entropy_all) + " " + str(varentropy_all) )
    # log(str(entropy) + " " + str(varentropy) )

    return _next.to(dtype=torch.int64)

def batched_sample(logits: list[torch.Tensor], kwargs: list[dict]) -> torch.Tensor:
    assert len(logits) == len(kwargs), "logits and kwargs must have the same length."
    return torch.stack([sample(l, **sample_args).to(dtype=torch.int64) for sample_args, l in zip(kwargs, logits)], dim=0)

def batched_next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, kwargs: Union[dict, list[dict]]) -> torch.Tensor:
    # Where:
    # input_pos is a 1d tensor of shape [seq_length...]
    # x is context tokens to add to the kvcache.
    # For prefill, x is a 2d tensor of shape [batch_size, prompt_length].
    # For subsequent tokens, x is a 2d tensor of shape [batch_size, 1].
    # kwargs is a list of dictionaries, each containing the keyword arguments for the sample function.
    # If one dictionary is passed, it's repeated for each sample in the batch.

    # In the future, we would like input_pos to be a 2d tensor of shape [batch_size, seq_length].
    # That way, we can support prompts of different sizes.
    # This means making the rope cache and kvcache forward() work with batches. Currently, they do not.
    # This is relatively complicated, given the current implementation. It will require some rewriting.
    # Relevant thread: https://discuss.pytorch.org/t/batched-index-select/9115
    # We will also need the same with tensor.index_copy_(). These do not work for batches, and the replacement
    # is somewhat nontrivial. Until then, we can only accept prompts that are all the same length.
    # After this problem is resolved, there will be another problem. That being, continuous batched prefill.
    # If you have any ideas on this, let me know. I don't think that padding input_pos is viable.

    _kwargs = kwargs if isinstance(kwargs, list) else [kwargs] * x.size(0)

    # Run the model on the batch.
    logits_stack = model(x, input_pos)

    # Unbind the logits stack into a list of logits.
    logits_list = [logits_stack] if logits_stack.ndim == 1 else logits_stack.unbind(0)
    logits_list = [l.unsqueeze(0) for l in logits_list]

    # Return the next token for each sample in the batch.
    return batched_sample(logits_list, kwargs=_kwargs)


@torch.inference_mode()
def generate_fn(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    stop_tokens: Tuple[List[int], ...] = (),
    include_prompt: bool,
    include_eos: bool,
) -> Iterator[torch.Tensor]:
    """
    Generates tokens for a single prompt.

    Args:
        model: The model to use.
        prompt: The tokenized prompt to generate from.
        max_returned_tokens: The maximum number of new tokens to return. Does not include the prompt tokens.
        temperature: The temp to pass to sample().
        top_k: The top_k to pass to sample().
        top_p: The top_p to pass to sample().
        stop_tokens: A tuple of stop sequences. If any of the sequences are generated, the generation stops early before max_returned_tokens.
        include_prompt: Whether to output the prompt tokens.
        include_eos: Whether to output the stop tokens if generation stops early.
    """



    prompt_size = prompt.size(0)
    device = prompt.device

    assert max_returned_tokens > prompt_size, f"Not enough space for {prompt_size} prompt tokens in a context length of {max_returned_tokens}."
    if model.max_seq_length < max_returned_tokens - 1:
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    # Yield the prompt if include_prompt is True
    if include_prompt:
        yield prompt

    stop_progress = [0] * len(stop_tokens)
    yielded_idx = 0

    # Generate output tokens.
    # The first token generated is the prefill token.
    # The input_pos for this token is the width of the entire prompt.
    # For subsequent iterations, it's the index in the context for the token that we're generating.
    tokens = []
    token = prompt
    prefill_token = True
    input_pos = torch.arange(0, prompt_size, device=device, dtype=torch.int64)
    for current_idx in range(max_returned_tokens - prompt_size):

        # Generate the token
        token = next_token(model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k, top_p=top_p)
        tokens.append(token)
        int_token = token.item()

        # Check for stop sequences
        # For each stop sequence, we keep a running total of how many are matched in stop_progress.
        # If the current token matches the next token in the stop sequence, we increment the
        # running total and hold off on yielding the token.
        for i, seq in enumerate(stop_tokens):
            if int_token == seq[stop_progress[i]]:
                stop_progress[i] += 1
                if stop_progress[i] == len(seq):
                    if include_eos:
                        yield from tokens[yielded_idx:]
                    return
            else:
                stop_progress[i] = 0

        # Yield tokens that are not part of a stop sequence in progress.
        # If there are no stop sequences, then that's all of them.
        if stop_tokens:
            safe_idx = len(tokens) - max(stop_progress)
        else:
            safe_idx = current_idx + 1 # include the token just generated

        if yielded_idx < safe_idx:
            y_tokens = tokens[yielded_idx : safe_idx]
            yield from y_tokens
            yielded_idx = safe_idx

        # Update input_pos for the next iteration.
        if prefill_token:
            prefill_token = False
            input_pos = torch.tensor([prompt_size], device=device, dtype=torch.int64)
        else:
            input_pos.add_(1)

    # Yield any remaining tokens
    if yielded_idx < len(tokens):
        yield from tokens[yielded_idx:]


# TODO: Make include_eos work.
# TODO: Rewrite unbatched generate_fn to use batched_generate_fn.
@torch.inference_mode()
def batched_generate_fn(
    model: GPT,
    prompts: torch.Tensor,
    max_returned_tokens: int,
    *,
    sample_args: Union[list[dict], dict],
    stop_tokens: Tuple[List[int], ...] = (),
    include_prompt: bool,
    include_eos: bool,
) -> Iterator[list[Union[torch.Tensor, None]]]:
    """
    Generates tokens for a batch of prompts.

    Args:
        model: The model to use.
        prompts: A 2D tensor of shape [batch_size, prompt_length].
        max_returned_tokens: The maximum number of new tokens to return. Does not include the prompt tokens.
        sample_args: The dictionary of kwargs to pass to sample() for each each token for each index in the batch.
        stop_tokens: A tuple of stop sequences. If any of the sequences are generated, the generation stops early before max_returned_tokens.
        include_prompt: Whether to output the prompt tokens.
        include_eos: Whether to output the stop tokens if generation stops early.

    Yields:
        A list of tokens for each prompt in the batch, or None if a stop sequence has already been encountered for that index in the batch.
    """

    if prompts.ndim == 1:
        prompts = prompts.unsqueeze(0)
    assert prompts.ndim == 2, "Prompts must be a 2D tensor."

    batch_size = prompts.size(0)
    max_prompt_size = prompts.size(1)
    device = prompts.device

    if isinstance(sample_args, dict):
        sample_args = [sample_args] * len(prompts)
    else:
        assert len(sample_args) == batch_size, "sample_args must have the length as the batch size."

    # TODO: This check (and the one in generate_fn) is not sufficient. We do the proper checks in LLM.generate().
    assert max_returned_tokens > max_prompt_size, f"Not enough space for {max_prompt_size} prompt tokens in a context length of {max_returned_tokens}."
    if model.max_seq_length < max_returned_tokens - 1:
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    # Yield the prompts if include_prompt is True
    if include_prompt:
        # TODO: Prompt length is padded, but they shouldn't all be the same length.
        for i in range(max_prompt_size):
            yield [prompt[i].view(-1) for prompt in prompts]

    stop_progresses = [[0] * len(stop_tokens) for _ in range(batch_size)] # [batch_size, ~len(stop_tokens)]
    stop_idxes = [-1] * batch_size
    yielded_idx = 0

    # Generate output tokens.
    # The first token generated is the prefill token.
    # The input_pos for this token is the width of the entire prompt.
    # For subsequent iterations, it's the index in the context for the token that we're generating.
    token_lists = [[] for _ in range(batch_size)]
    tokens: torch.Tensor = prompts
    prefill_token = True
    input_pos = torch.arange(0, max_prompt_size, device=device, dtype=torch.int64)
    for current_idx in range(max_returned_tokens - max_prompt_size):

        # Generate the next token for each prompt in the batch.
        # This is of shape [batch_size, 1].
        tokens = batched_next_token(model, input_pos, tokens, sample_args)
        for i in range(batch_size):
            token_lists[i].append(tokens[i])
        int_tokens = [token.item() for token in tokens]

        # Check for stop sequences
        # For each stop sequence, we keep a running total of how many are matched in stop_progress.
        # If the current token matches the next token in the stop sequence, we increment the
        # running total and hold off on yielding the token.
        for batch_idx, int_token in enumerate(int_tokens):
            if stop_idxes[batch_idx] != -1:
                continue
            for seq_idx, seq in enumerate(stop_tokens):
                seq_pos = stop_progresses[batch_idx][seq_idx]
                if seq_pos >= len(seq):
                    continue
                if int_token == seq[seq_pos]:
                    stop_progresses[batch_idx][seq_idx] += 1
                    if stop_progresses[batch_idx][seq_idx] == len(seq):
                        stop_idxes[batch_idx] = current_idx
                else:
                    stop_progresses[batch_idx][seq_idx] = 0

        # Yield tokens that are not part of a stop sequence in progress.
        # If there are no stop sequences, then that's all of them.
        if len(stop_tokens) != 0:
            safe_idxes = [len(token_lists[i]) - max(stop_progresses[i]) for i in range(batch_size)]
        else:
            safe_idxes = [current_idx + 1] # include the token just generated
        safe_idx = min(safe_idxes)

        if yielded_idx < safe_idx:
            for idx in range(yielded_idx, safe_idx):
                y_tokens = [token_lists[i][idx] if (stop_idxes[i] == -1 or idx < stop_idxes[i]) else None for i in range(batch_size)]
                if all(y is None for y in y_tokens):
                    return
                yield y_tokens
            yielded_idx = safe_idx

        # Update input_pos for the next iteration.
        if prefill_token:
            prefill_token = False

            # TODO: Make the model support a batched input_pos of shape [batch_size, 1].
            # The kvcache has been fixed, but the rope cache is still broken.
            input_pos = torch.tensor([max_prompt_size], device=device, dtype=torch.int64)
        else:
            input_pos.add_(1)

    # Yield any remaining tokens
    max_token_lists = max(len(l) for l in token_lists)
    if yielded_idx < max_token_lists:
        for idx in range(yielded_idx, max_token_lists):
            y_tokens = [token_lists[i][idx] if (stop_idxes[i] == -1 or idx < stop_idxes[i]) else None for i in range(batch_size)]
            if all(y is None for y in y_tokens):
                return
            yield y_tokens
    return


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    include_prompt: bool = True,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after applying the prompt style) to the output.
    """

    token_list = list(generate_fn(
        include_prompt=include_prompt,
        include_eos=True,
        model=model,
        prompt=prompt,
        max_returned_tokens=max_returned_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_tokens=(([eos_id],) if eos_id is not None else ())
    ))

    return torch.cat(token_list) if not len(token_list) == 0 else torch.Tensor()


@torch.inference_mode()
def main(
    checkpoint_dir: Path,
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Default generation option.

    Generates text samples based on a pre-trained model and tokenizer.

    Args:
        checkpoint_dir: The checkpoint directory to load.
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. "
                "This may result in errors when using quantization."
            )
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
    )

    prompt = prompt_style.apply(prompt)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead")

    model = fabric.setup_module(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
