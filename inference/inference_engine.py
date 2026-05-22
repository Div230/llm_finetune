"""
inference/inference_engine.py
==============================
Production inference using vLLM's PagedAttention engine + HuggingFace fallback.

This module implements two inference backends:
  1. vLLM: High-throughput serving using PagedAttention (production)
  2. HuggingFace: Standard generate() API (debugging / low-throughput)

vLLM ARCHITECTURE DEEP DIVE:
=============================
vLLM (Virtual Large Language Model) is a serving system from UC Berkeley
that achieves near-theoretical peak throughput for LLM inference.

Key papers/concepts:
  - PagedAttention: Manages KV cache with virtual memory pagination
  - Continuous batching: Dynamic request scheduling
  - Custom CUDA kernels: fused attention with paged blocks

KV CACHE EXPLANATION:
---------------------
During autoregressive generation, at each step the model computes:
  - Key vectors K for new token (based on current token + position)
  - Value vectors V for new token

To avoid recomputing K,V for ALL previous tokens at every step:
  → We CACHE the K,V vectors from all previous steps!
  → At step n, we only compute K,V for the NEW token (step n)
  → Attend over cached K,V from steps 0..n-1 + new K,V

MEMORY COST OF KV CACHE:
  Per layer: 2 (K and V) × num_heads × head_dim × num_tokens × bytes
  For Llama-3-8B (32 layers, 32 heads, 128 head_dim, BF16):
    Per token: 2 × 32 × 32 × 128 × 2 bytes = 524,288 bytes ≈ 0.5 MB per token!
    For 2048 tokens: ~1 GB per sequence in KV cache
    For batch of 10: ~10 GB just for KV cache!

PAGEDATTENTION SOLUTION:
  Standard KV cache: allocate MAX_SEQ_LEN × model_dim contiguous memory upfront
    Problem: sequences are usually shorter than max → massive waste
    Problem: can't know sequence length in advance → must over-allocate
  
  PagedAttention: allocate KV cache in fixed-size "pages" (16-32 tokens each)
    Each page is a contiguous block of memory for K,V of 16-32 tokens
    Pages are allocated as tokens are generated (on-demand!)
    Pages can be non-contiguous in memory (like OS virtual memory)
    Custom CUDA kernel handles the non-contiguous attention computation
    
    RESULT: 90%+ KV cache memory utilization vs 50-70% for standard
    = can serve 2× more concurrent requests on same hardware!

CONTINUOUS BATCHING:
  Naive batching: fill batch with N requests, wait for ALL to complete
    Problem: if one request needs 1000 tokens and others need 10,
             the 10-token requests wait 100× longer than needed
  
  Continuous batching: after each forward step, check if any sequences finished
    If yes: immediately fill that slot with a new request
    Result: GPU always near 100% utilization (no waiting for long sequences)
    Throughput improvement: 2-10× over naive batching for mixed requests
"""

import logging
import os
from typing import Dict, Generator, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from threading import Thread

from configs.config import InferenceConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# HUGGINGFACE INFERENCE (Fallback / Debug Backend)
# ==============================================================================

class HFInferenceEngine:
    """
    Standard HuggingFace inference using model.generate().

    USE WHEN:
    - vLLM is unavailable (installation issues, non-CUDA system)
    - Debugging (vLLM hides some errors)
    - Adapters NOT merged (vLLM requires merged model)
    - Low throughput scenarios (single request at a time)

    GENERATE() INTERNALS:
    ---------------------
    model.generate() implements:
    1. Greedy / Sampling decoding loop
    2. Stopping criteria (EOS, max_new_tokens)
    3. Multiple decoding strategies:
       - Greedy: argmax at each step (fast, deterministic)
       - Sampling: sample from distribution (diverse, stochastic)
       - Beam search: maintain top-k hypotheses (quality, slow)
       - Top-k + nucleus: best for chat applications

    KV CACHE IN HF:
    The past_key_values tuple stores (key, value) for each layer.
    Passed between generate() steps to avoid recomputation.
    Memory grows linearly with generated tokens.
    HF's KV cache uses contiguous memory (not paged) → less memory efficient than vLLM.

    PERFORMANCE:
    - Single request: similar speed to vLLM
    - Multiple requests: much slower than vLLM (no continuous batching)
    """

    def __init__(
        self,
        model_path: str,
        inference_config: InferenceConfig,
        device: str = "auto",
    ):
        self.config = inference_config
        self.device = device

        logger.info(f"Loading HF model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Switch to left padding for generation
        # WHY LEFT PADDING FOR GENERATION?
        # During batch generation, all sequences must have same length.
        # If we right-pad: [prompt, PAD, PAD] → model generates from PAD position
        # If we left-pad:  [PAD, PAD, prompt] → model generates from end of real tokens
        # Left padding is correct for autoregressive generation.
        self.tokenizer.padding_side = "left"

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(inference_config.dtype, torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation="flash_attention_2" if _flash_attn_available() else "sdpa",
        )
        self.model.eval()

        logger.info(f"HF model loaded. Device: {next(self.model.parameters()).device}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> List[str]:
        """
        Generate text for one or multiple prompts.

        BATCHED GENERATION:
        -------------------
        Processing multiple prompts simultaneously on the same GPU is more
        efficient than processing them sequentially because:
        1. GPU has thousands of cores → can process batch in parallel
        2. Memory bandwidth is amortized across batch
        3. Model weights are loaded once for the entire batch

        OPTIMAL BATCH SIZE:
        Limited by GPU memory (KV cache grows with batch size × seq length).
        For inference, the bottleneck is usually memory bandwidth, not compute.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Use defaults from config if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        repetition_penalty = repetition_penalty or self.config.repetition_penalty

        # Tokenize (with dynamic batching padding)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,  # Pad to same length for batching
            truncation=True,
            max_length=4096,
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            # AUTOCAST: use BF16 for faster inference, FP32 for accumulations
            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16,
                enabled=self.config.dtype != "float32",
            ):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    # Temperature sampling: T>0 enables sampling, T=0 = greedy
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    # Top-p nucleus sampling
                    top_p=top_p,
                    # Top-k sampling (applied BEFORE nucleus)
                    top_k=top_k,
                    # Repetition penalty: reduce logit score for repeated tokens
                    repetition_penalty=repetition_penalty,
                    # use_cache=True: enable KV cache (default)
                    use_cache=True,
                    # Stopping tokens
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        # Decode only the generated tokens (not the input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]

        # Batch decode
        texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return texts

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation: yield tokens as they're generated.

        STREAMING IMPLEMENTATION:
        -------------------------
        HuggingFace TextIteratorStreamer runs generation in a background thread
        and yields decoded tokens through a queue.

        The caller can then stream tokens to the UI/client incrementally.
        This is how ChatGPT-style streaming interfaces work:
        1. Model generates token by token
        2. Each token is sent to client immediately (via Server-Sent Events or WebSocket)
        3. Client appends token to displayed text
        
        Result: User sees response appearing character-by-character (~ChatGPT UX)

        IMPLEMENTATION DETAIL:
        TextIteratorStreamer uses a queue internally:
          - Generation thread: puts decoded tokens in queue
          - Caller thread: gets tokens from queue and yields them
          - skip_prompt=True: don't stream the input prompt back
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # Don't repeat the input
            skip_special_tokens=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "streamer": streamer,  # Pass streamer to generate()
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Run generation in a background thread
        # This allows us to yield tokens from the main thread while generation continues
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they arrive from the streamer queue
        for token_text in streamer:
            yield token_text

        thread.join()


# ==============================================================================
# vLLM INFERENCE ENGINE (Production Backend)
# ==============================================================================

class VLLMInferenceEngine:
    """
    High-throughput inference using vLLM's PagedAttention.

    PREREQUISITES:
    - Merged model (LoRA adapters merged into base model weights)
    - vLLM installed: pip install vllm
    - CUDA 11.8+ with compatible GPU

    VLLM INITIALIZATION:
    --------------------
    LLM() constructor:
      1. Loads model weights into GPU memory
      2. Allocates KV cache pages (based on gpu_memory_utilization)
      3. Sets up PagedAttention CUDA kernels
      4. Starts scheduler thread

    KV CACHE ALLOCATION:
      vLLM pre-allocates a pool of KV cache pages at startup.
      Total pages = GPU_memory × gpu_memory_utilization - model_weights
                    ÷ (page_size × 2 × num_layers × head_dim × num_heads × dtype_bytes)
      More pages → can serve longer sequences / more concurrent requests.
      Typical: gpu_memory_utilization=0.9 leaves 10% for activations.

    PERFORMANCE CHARACTERISTICS:
      - Latency: Similar to HF for single requests
      - Throughput: 5-10× higher than HF for concurrent requests
      - Memory efficiency: 90%+ KV cache utilization (vs 50% HF)
      - Best for: API serving with many concurrent users
    """

    def __init__(
        self,
        model_path: str,
        inference_config: InferenceConfig,
    ):
        self.config = inference_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        try:
            from vllm import LLM, SamplingParams
            self._LLM = LLM
            self._SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Note: vLLM requires CUDA 11.8+ and Linux. "
                "Use HFInferenceEngine for other environments."
            )

        logger.info(f"Initializing vLLM engine with model: {model_path}")
        logger.info("This may take 1-5 minutes for model loading + KV cache allocation...")

        # VLLM LLM ENGINE CONFIGURATION:
        self.llm = LLM(
            model=model_path,
            # dtype: computation dtype (bfloat16 for Ampere+ GPUs)
            dtype=inference_config.dtype,
            # tensor_parallel_size: split model across N GPUs
            # Tensor parallelism splits each weight matrix column-wise across GPUs
            # Each GPU computes its slice, results all-gathered after each layer
            # For single GPU: 1 (no splitting needed)
            # For multi-GPU: 2 or 4 (must evenly divide num_attention_heads)
            tensor_parallel_size=inference_config.tensor_parallel_size,
            # gpu_memory_utilization: fraction of GPU memory to use for KV cache
            # 0.9 = use 90% of GPU memory for model + KV cache
            # Lower if OOM: try 0.8 or 0.7
            gpu_memory_utilization=0.9,
            # max_model_len: maximum context length (prompt + generation)
            # Longer context = more KV cache pages needed
            max_model_len=4096,
            # trust_remote_code: needed for some custom architectures
            trust_remote_code=False,
            # swap_space: CPU RAM to use for KV cache overflow (in GB)
            # When GPU KV cache full, overflow to CPU (slow but prevents OOM)
            swap_space=4,
            # enforce_eager: disable CUDA graph capture (slower but more compatible)
            # CUDA graphs: vLLM captures computation graph and replays for speed
            # Disable if you have compatibility issues
            enforce_eager=False,
        )

        logger.info("vLLM engine initialized successfully!")
        logger.info(f"KV cache pages allocated. Ready for inference.")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> List[str]:
        """
        Batch generation with vLLM.

        vLLM GENERATE INTERNALS:
        ------------------------
        1. Tokenize all prompts
        2. Add requests to vLLM scheduler
        3. Scheduler uses continuous batching to process requests
        4. PagedAttention allocates KV cache pages on demand
        5. Return completed sequences

        SAMPLING PARAMS:
        vLLM's SamplingParams mirrors HF's generation config.
        Under the hood, sampling is implemented in custom CUDA kernels
        for maximum throughput.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        sampling_params = self._SamplingParams(
            max_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            top_k=top_k or self.config.top_k,
            repetition_penalty=repetition_penalty or self.config.repetition_penalty,
            # stop: stop generation at these strings
            stop=[self.tokenizer.eos_token] if self.tokenizer.eos_token else None,
        )

        # VLLM BATCH PROCESSING:
        # All prompts are submitted simultaneously to the engine.
        # vLLM schedules them using continuous batching:
        #   - Batch multiple requests together in each forward pass
        #   - When a request completes, remove it and add a new one immediately
        #   - GPU always running at maximum capacity
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract generated text (excluding the input prompt)
        texts = []
        for output in outputs:
            # Each RequestOutput has a list of CompletionOutput (for beam search)
            # We use only the first completion (greedy/sampling)
            texts.append(output.outputs[0].text)

        return texts

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation with vLLM.

        VLLM STREAMING MECHANISM:
        -------------------------
        vLLM's async_generate() returns an async generator.
        We use the synchronous streaming API here for simplicity.
        
        Each yielded RequestOutput contains the tokens generated so far.
        We track what we've already yielded to return only the NEW tokens.
        
        For production async streaming, use vLLM's AsyncLLMEngine with
        FastAPI + Server-Sent Events (SSE) — this is how Open WebUI works.
        """
        sampling_params = self._SamplingParams(
            max_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            stream=True,  # Enable streaming output
        )

        previous_text = ""

        for output in self.llm.generate(prompt, sampling_params, stream=True):
            new_text = output.outputs[0].text
            # Yield only the newly generated portion
            delta = new_text[len(previous_text):]
            if delta:
                yield delta
            previous_text = new_text


# ==============================================================================
# UNIFIED INFERENCE INTERFACE
# ==============================================================================

class InferenceEngine:
    """
    Unified interface that routes to vLLM or HF based on config and availability.

    This is the recommended interface for external code.
    It handles:
    - Backend selection (vLLM vs HF)
    - Graceful fallback if vLLM unavailable
    - Consistent API regardless of backend

    USAGE:
        engine = InferenceEngine.from_config(inference_config)

        # Single request
        response = engine.generate("What is the capital of France?")

        # Batch request
        responses = engine.generate(["Question 1", "Question 2", "Question 3"])

        # Streaming
        for token in engine.generate_stream("Tell me a story about..."):
            print(token, end="", flush=True)
    """

    def __init__(self, backend: Union[HFInferenceEngine, VLLMInferenceEngine]):
        self._backend = backend

    @classmethod
    def from_config(cls, inference_config: InferenceConfig) -> "InferenceEngine":
        """Create InferenceEngine with appropriate backend."""
        if inference_config.use_vllm:
            try:
                backend = VLLMInferenceEngine(
                    model_path=inference_config.model_path,
                    inference_config=inference_config,
                )
                logger.info("Using vLLM backend")
                return cls(backend)
            except (ImportError, Exception) as e:
                logger.warning(
                    f"vLLM unavailable ({e}). Falling back to HuggingFace backend."
                )

        backend = HFInferenceEngine(
            model_path=inference_config.model_path,
            inference_config=inference_config,
        )
        logger.info("Using HuggingFace backend")
        return cls(backend)

    def generate(
        self,
        prompts: Union[str, List[str]],
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate text. Returns single string if single prompt, list if multiple."""
        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]

        results = self._backend.generate(prompts, **kwargs)
        return results[0] if single else results

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream tokens as they're generated."""
        yield from self._backend.generate_stream(prompt, **kwargs)


# ==============================================================================
# UTILITIES
# ==============================================================================

def _flash_attn_available() -> bool:
    """Check if flash-attn is installed and GPU supports it."""
    try:
        import flash_attn  # noqa
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except ImportError:
        return False


def format_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """
    Format a single instruction into the model's expected prompt format.
    Uses tokenizer's chat_template if available, else falls back to simple format.
    """
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # Add the assistant response prefix
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}. Using fallback format.")

    # Fallback format
    return f"[INST] {instruction} [/INST]"
