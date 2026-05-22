"""
models/model_loader.py
======================
Model loading, QLoRA quantization setup, and LoRA adapter creation.

This module handles the most technically complex part of the pipeline:
loading a large model in 4-bit precision and attaching trainable LoRA adapters.

EXECUTION FLOW:
  load_base_model()
    → configure BitsAndBytesConfig (NF4 quantization parameters)
    → AutoModelForCausalLM.from_pretrained() (loads weights in 4-bit to GPU)
    → prepare_model_for_kbit_training() (PEFT setup for quantized models)
    → create_lora_config() (define LoRA adapter architecture)
    → get_peft_model() (attach LoRA adapters to model)
    → print_trainable_parameters() (verify only LoRA params are trainable)

PEFT INTERNALS:
  When get_peft_model() is called, PEFT:
  1. Identifies target modules by name matching (e.g., 'q_proj', 'v_proj')
  2. For
Untitled.ipynb
￼
￼
￼
￼
￼
￼
￼
￼
￼
 each target Linear layer, creates a LoRALayer wrapper:
     LoRALinear(
       base_layer: nn.Linear (frozen, 4-bit quantized),
       lora_A: nn.Linear(in_features, r, bias=False),  # trainable
       lora_B: nn.Linear(r, out_features, bias=False),  # trainable
       scaling: alpha / r,
     )
  3. Forward: output = base_layer(x) + scaling * lora_B(lora_A(x))
  4. Freezes base_layer parameters (requires_grad=False)
  5. Only lora_A, lora_B remain with requires_grad=True
"""

import logging
import os
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from configs.config import ModelConfig, LoRAConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# TOKENIZER LOADING
# ==============================================================================

def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer for the model.

    CRITICAL TOKENIZER CONFIGURATIONS:
    ------------------------------------
    1. PADDING TOKEN:
       Most causal LM tokenizers (GPT-style) have NO padding token by default
       because autoregressive models don't naturally need padding.
       We MUST add one for batch training (dynamic padding requires pad token).
       Common solutions:
         a) Use EOS token as pad: tokenizer.pad_token = tokenizer.eos_token
            Downside: model might stop generating early if it "sees" pad=EOS
         b) Add new token: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            Requires model.resize_token_embeddings(len(tokenizer))
       We use option (a) for simplicity; option (b) is cleaner.

    2. PADDING SIDE:
       'right' for training (causal attention, right-side padding is standard)
       'left' for inference (batch generation needs left-side padding so all
       sequences have their actual tokens at the END, not padded tokens)
       We switch sides dynamically between training and inference.

    3. CHAT TEMPLATE:
       Modern tokenizers include a Jinja2 chat template for consistent formatting.
       This template is authoritative — use it in inference if possible.
       During training, we use our explicit PROMPT_TEMPLATES for control.

    4. SLOW vs FAST TOKENIZER:
       Fast tokenizers (Rust-based HuggingFace Tokenizers) are 5-10× faster.
       use_fast=True is default and recommended.

    Returns:
        Configured tokenizer
    """
    logger.info(f"Loading tokenizer: {model_config.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=model_config.cache_dir,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,  # Rust-based fast tokenizer (5-10× faster)
    )

    # ── Configure Padding Token ───────────────────────────────────────────────
    if tokenizer.pad_token is None:
        # Use EOS as pad token — standard approach for Llama/Mistral
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token = eos_token ({tokenizer.eos_token})")

    # ── Set Padding Side ──────────────────────────────────────────────────────
    # Right padding for training (left padding would shift position IDs)
    tokenizer.padding_side = "right"

    # ── Validate Special Tokens ───────────────────────────────────────────────
    logger.info(
        f"Tokenizer vocab size: {len(tokenizer)}, "
        f"BOS: {tokenizer.bos_token}, "
        f"EOS: {tokenizer.eos_token}, "
        f"PAD: {tokenizer.pad_token}"
    )

    return tokenizer


# ==============================================================================
# BITS AND BYTES QUANTIZATION CONFIG
# ==============================================================================

def create_bnb_config(lora_config: LoRAConfig) -> Optional[BitsAndBytesConfig]:
    """
    Create BitsAndBytes 4-bit quantization configuration.

    BITSANDBYTES INTERNALS:
    -----------------------
    bitsandbytes is a CUDA extension library by Tim Dettmers (QLoRA author).
    It provides:
      1. 4-bit quantization: stores weights as 4-bit NF4/FP4, dequants to BF16 for compute
      2. 8-bit quantization: INT8 storage, dequants during matmul
      3. Paged optimizers: CPU-offloaded Adam states

    4-BIT QUANTIZATION PROCESS (per block of 64 weights):
      1. Normalize block to [-1, 1] by dividing by max |value|
      2. Map each normalized value to nearest NF4 quantization level
         (16 levels for 4 bits = 2⁴ levels)
      3. Store 4-bit indices + FP32 scaling factor per block

    DEQUANTIZATION (during forward pass compute):
      1. Load 4-bit indices from VRAM
      2. Look up NF4 table: index → [-1, 1] float value
      3. Multiply by block scale factor → original magnitude restored
      4. Compute in BF16

    This all happens transparently inside CUDA kernels.
    The Linear layer appears normal to PyTorch but internally handles quantization.

    Returns:
        BitsAndBytesConfig if use_4bit=True, else None
    """
    if not lora_config.use_4bit:
        return None

    # Map string dtype to torch dtype
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(
        lora_config.bnb_4bit_compute_dtype, torch.bfloat16
    )

    # Check BF16 support
    if compute_dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            logger.warning(
                "GPU doesn't support BF16. Falling back to FP16 for compute dtype."
            )
            compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        # ── 4-bit Loading ─────────────────────────────────────────────────────
        load_in_4bit=True,
        # Quantization type: 'nf4' (NormalFloat4) or 'fp4' (Float4)
        # NF4 is better for normally-distributed weights (all LLM weights)
        bnb_4bit_quant_type=lora_config.bnb_4bit_quant_type,
        # Computation happens in BF16 even though storage is 4-bit
        # This preserves numerical accuracy during forward/backward passes
        bnb_4bit_compute_dtype=compute_dtype,
        # Double quantization: quantize the scale factors too (saves ~0.37 bits/param)
        bnb_4bit_use_double_quant=lora_config.use_double_quantization,
    )

    logger.info(
        f"Created 4-bit config: type={lora_config.bnb_4bit_quant_type}, "
        f"compute={lora_config.bnb_4bit_compute_dtype}, "
        f"double_quant={lora_config.use_double_quantization}"
    )

    return bnb_config


# ==============================================================================
# BASE MODEL LOADING
# ==============================================================================

def load_base_model(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    bnb_config: Optional[BitsAndBytesConfig],
) -> PreTrainedModel:
    """
    Load the base LLM with optional 4-bit quantization.

    DEVICE MAPPING:
    ---------------
    device_map="auto" tells Accelerate to automatically distribute model layers
    across available GPUs (and CPU/disk if needed).

    For single GPU: all layers go to cuda:0
    For multiple GPUs: layers split to maximize VRAM utilization
    Algorithm: greedy layer-by-layer assignment filling GPU 0 first, then GPU 1, etc.

    ACCELERATE DEVICE MAP INTERNALS:
    ---------------------------------
    Accelerate's device_map uses a dispatch table:
      model.hf_device_map = {
        "model.embed_tokens": "cuda:0",
        "model.layers.0": "cuda:0",
        ...
        "model.layers.20": "cuda:1",  # overflow to GPU 1
        ...
      }
    Hooks are inserted to move tensors between devices during forward pass.
    This is less efficient than tensor parallelism but much simpler to implement.

    TORCH DTYPE FOR LOADING:
    -----------------------
    When loading in 4-bit: torch_dtype affects the BF16 compute buffer size
    When loading in BF16: torch_dtype=torch.bfloat16 halves model memory vs FP32

    Returns:
        Loaded model (possibly quantized)
    """
    logger.info(f"Loading model: {model_config.model_name_or_path}")

    # Determine compute dtype
    if bnb_config is not None:
        torch_dtype = bnb_config.bnb_4bit_compute_dtype
    elif torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # ── Model Loading ─────────────────────────────────────────────────────────
    # This is where HuggingFace downloads the model shards, dequantizes/maps them,
    # and loads into VRAM. For a 7B model in 4-bit: ~4 GB VRAM needed.
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        # Quantization config (None = load in full precision)
        quantization_config=bnb_config,
        # Device placement: 'auto' = Accelerate decides, or specify 'cuda:0'
        device_map="auto",
        # Compute dtype for non-quantized computations
        torch_dtype=torch_dtype,
        # FLASH ATTENTION 2 via attn_implementation
        # 'flash_attention_2': requires flash-attn installed, Ampere+ GPU
        # 'sdpa': PyTorch Scaled Dot Product Attention (built-in, Ampere+)
        # 'eager': standard HF attention (always works, slowest)
        attn_implementation=(
            "flash_attention_2"
            if model_config.use_flash_attention_2
            else "sdpa"
        ),
        # Trust remote code: needed for models with custom architectures
        trust_remote_code=model_config.trust_remote_code,
        # Cache directory for model weights
        cache_dir=model_config.cache_dir,
        # pretraining_tp=1: Disable tensor parallelism (we use data parallelism instead)
    )

    logger.info(f"Model loaded. dtype: {torch_dtype}")
    _log_model_memory(model)

    return model


# ==============================================================================
# LORA ADAPTER SETUP
# ==============================================================================

def _auto_detect_target_modules(model: PreTrainedModel) -> List[str]:
    """
    Auto-detect appropriate LoRA target modules for the model architecture.

    DETECTION STRATEGY:
    -------------------
    We look for Linear layers with names matching common attention/MLP patterns.
    Different model architectures use different naming conventions:
      Llama/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
      Qwen: c_attn (fused QKV), c_proj, w1, w2
      Falcon: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
      Phi: Wqkv, out_proj, fc1, fc2

    We target ALL Linear layers that match these patterns.
    For production, it's better to explicitly specify target modules.
    """
    # Common attention projection patterns across architectures
    attention_patterns = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Llama, Mistral, Gemma
        "query_key_value",  # Falcon
        "c_attn",  # Qwen, GPT-2 style
        "Wqkv",  # Phi
        "qkv_proj",  # Some models
    ]

    # MLP patterns
    mlp_patterns = [
        "gate_proj", "up_proj", "down_proj",  # Llama SwiGLU MLP
        "fc1", "fc2",  # Standard MLP
        "dense_h_to_4h", "dense_4h_to_h",  # Falcon
        "c_fc", "c_proj",  # GPT style
        "w1", "w2", "w3",  # Qwen
    ]

    all_patterns = attention_patterns + mlp_patterns
    found_modules = set()

    # Walk all named modules and check for matches
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Check if the module's last name component matches any pattern
        module_name = name.split(".")[-1]
        if module_name in all_patterns:
            found_modules.add(module_name)

    if not found_modules:
        logger.warning(
            "Could not auto-detect LoRA target modules. "
            "Using ['q_proj', 'v_proj'] as safe default. "
            "Please specify target_modules explicitly for best results."
        )
        return ["q_proj", "v_proj"]

    # Filter to attention only (safer default) unless MLP modules detected are standard
    attn_found = found_modules.intersection(attention_patterns)
    mlp_found = found_modules.intersection(mlp_patterns)

    target = list(attn_found) + list(mlp_found)
    logger.info(f"Auto-detected LoRA target modules: {sorted(target)}")
    return sorted(target)


def setup_lora(
    model: PreTrainedModel,
    lora_config_params: LoRAConfig,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[PreTrainedModel, LoraConfig]:
    """
    Set up LoRA adapters on the base model.

    PREPARE_MODEL_FOR_KBIT_TRAINING:
    --------------------------------
    This PEFT function does several important things for 4-bit training:
    1. Enables gradient computation for non-quantized parameters
    2. Casts LayerNorm layers to FP32 (quantized LayerNorm is numerically unstable)
    3. Sets up input embedding gradient computation
    4. Enables gradient checkpointing if requested
    
    WHY LAYERNORM IN FP32?
    The LayerNorm operation computes: y = (x - μ) / √(σ² + ε) × γ + β
    With only 4-bit precision in LayerNorm, the running statistics and scale
    factors lose too much precision → training instability or NaN.
    FP32 for LayerNorm is essentially free (LayerNorm has negligible memory cost).

    Returns:
        (peft_model, lora_config) tuple
    """
    # ── Step 1: Prepare quantized model for k-bit training ────────────────────
    if lora_config_params.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # use_reentrant=False: new PyTorch checkpoint API, more stable
            # use_reentrant=True: legacy, may cause issues with some models
        )

    # ── Step 2: Resolve target modules ───────────────────────────────────────
    target_modules = lora_config_params.target_modules
    if target_modules is None:
        target_modules = _auto_detect_target_modules(model)
    
    logger.info(f"LoRA target modules: {target_modules}")

    # ── Step 3: Create LoRA configuration ─────────────────────────────────────
    # LORA CONFIG MATH REFERENCE:
    # For module W ∈ ℝ^{d_out × d_in}:
    #   lora_A ∈ ℝ^{r × d_in}: initialized from N(0, 1/r) (Kaiming normal)
    #   lora_B ∈ ℝ^{d_out × r}: initialized to zeros (output is zero at start)
    #   forward: h = W₀x + (alpha/r) × lora_B(lora_A(x))
    # Training: only lora_A and lora_B are updated

    lora_config = LoraConfig(
        r=lora_config_params.lora_r,
        lora_alpha=lora_config_params.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_config_params.lora_dropout,
        bias=lora_config_params.bias,
        task_type=TaskType.CAUSAL_LM,
        # init_lora_weights=True: use default initialization (A~N, B=0)
        # init_lora_weights="gaussian": both A, B from N(0, 1/r)
        # init_lora_weights="loftq": LoftQ initialization (better convergence)
        init_lora_weights=True,
        # inference_mode=False: enable LoRA dropout during training
        inference_mode=False,
    )

    # ── Step 4: Apply LoRA to model ───────────────────────────────────────────
    # get_peft_model() wraps specified Linear layers with LoRALinear
    # and freezes all non-LoRA parameters
    model = get_peft_model(model, lora_config)

    # ── Step 5: Verify parameter counts ──────────────────────────────────────
    print_trainable_parameters(model)

    return model, lora_config


# ==============================================================================
# UTILITIES
# ==============================================================================

def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Print a summary of trainable vs total parameters.

    This is essential verification after LoRA setup.
    Expected output for 7B model with r=16:
      All params: 6,738,415,616 | Trainable: 84,394,496 | Trainable%: 1.25%

    If trainable% is 100%, LoRA wasn't applied correctly.
    If trainable% is 0%, no parameters are being trained (bug!).
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(
        f"All params: {all_params:,} | "
        f"Trainable: {trainable_params:,} | "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )
    print(
        f"\n{'='*60}\n"
        f"MODEL PARAMETER SUMMARY\n"
        f"{'='*60}\n"
        f"  Total parameters:     {all_params:>15,}\n"
        f"  Trainable parameters: {trainable_params:>15,}\n"
        f"  Frozen parameters:    {all_params - trainable_params:>15,}\n"
        f"  Trainable percentage: {100 * trainable_params / all_params:>14.2f}%\n"
        f"{'='*60}\n"
    )


def _log_model_memory(model: PreTrainedModel) -> None:
    """Log estimated model memory usage."""
    param_memory = 0
    for name, param in model.named_parameters():
        # Estimate dtype byte size
        if param.dtype in (torch.float32,):
            bytes_per_param = 4
        elif param.dtype in (torch.float16, torch.bfloat16):
            bytes_per_param = 2
        else:
            bytes_per_param = 0.5  # 4-bit

        param_memory += param.numel() * bytes_per_param

    param_memory_gb = param_memory / (1024 ** 3)
    logger.info(f"Estimated model weight memory: {param_memory_gb:.2f} GB")


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    merge: bool = False,
) -> None:
    """
    Save LoRA adapters or merged model.

    SAVING STRATEGIES:
    ------------------
    Option A: Save LoRA adapters only (default, ~100 MB for r=16)
      model.save_pretrained(output_dir)  → saves adapter_config.json + adapter_model.bin
      tokenizer.save_pretrained(output_dir)
      
      To load: PeftModel.from_pretrained(base_model, output_dir)
      USE CASE: When you want to swap adapters, share adapters without base model,
               or use multiple LoRA adapters with same base model.

    Option B: Merge adapters into base model and save full model (~14 GB for 7B BF16)
      merged = model.merge_and_unload()  → computes W = W₀ + (α/r)BA, removes adapters
      merged.save_pretrained(output_dir)
      
      MERGE MATH: W_final = W_frozen + (alpha/r) × B × A
      W_final is a standard Linear layer — no PEFT overhead at inference time.
      USE CASE: Deploy with vLLM, TGI, or any standard inference framework.

    SAFETENSORS FORMAT:
    -------------------
    We save in safetensors format (not pickle):
      - Safe: no arbitrary code execution (pickle can run arbitrary Python)
      - Fast: memory-mapped loading, faster than pickle for large models
      - Compatible with all HuggingFace tooling
    """
    os.makedirs(output_dir, exist_ok=True)

    if merge:
        logger.info("Merging LoRA adapters into base model...")
        # MERGE AND UNLOAD:
        # 1. Computes W_final = W_base + (alpha/r) * B * A for each LoRA module
        # 2. Replaces LoRALinear with standard nn.Linear containing W_final
        # 3. Removes adapter weights from model
        # Result: standard model with no PEFT overhead
        merged_model = model.merge_and_unload()

        # Dequantize if still in 4-bit (merged model should be in compute dtype)
        # Note: merging automatically handles dequantization

        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,  # Save as safetensors
            max_shard_size="5GB",  # Shard into 5GB files for large models
        )
        logger.info(f"Merged model saved to {output_dir}")
    else:
        # Save LoRA adapters only
        model.save_pretrained(output_dir)
        logger.info(f"LoRA adapters saved to {output_dir}")

    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")


def load_trained_model(
    base_model_path: str,
    adapter_path: str,
    lora_config_params: LoRAConfig,
) -> PreTrainedModel:
    """
    Load a previously trained model with LoRA adapters for inference or continued training.

    LOADING FLOW:
    1. Load base model (quantized if use_4bit=True)
    2. Load LoRA adapters on top
    3. Return model ready for inference

    For inference: model.eval() and generate()
    For continued training: setup_lora() not needed (adapters already attached)
    """
    # Load base model (same config as original training)
    bnb_config = create_bnb_config(lora_config_params)
    
    from configs.config import ModelConfig
    model_config = ModelConfig(model_name_or_path=base_model_path)
    base_model = load_base_model(model_config, lora_config_params, bnb_config)

    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,  # Inference mode
    )

    return model
