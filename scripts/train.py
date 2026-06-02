"""
scripts/train.py
================
Main entry point for the fine-tuning pipeline.

USAGE:
------
# Basic run (uses all defaults from configs/config.py):
python scripts/train.py

# Custom model and dataset:
python scripts/train.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --output_dir ./outputs/mistral-ultrachat

# Resume from checkpoint:
python scripts/train.py --resume_from_checkpoint ./outputs/checkpoint-step500

# Multi-GPU (4 GPUs):
accelerate launch --num_processes 4 scripts/train.py \
    --per_device_train_batch_size 4

# With config YAML (recommended for experiments):
python scripts/train.py --config_file configs/llama3_guanaco.yaml
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import HfArgumentParser, set_seed

from configs.config import ModelConfig, LoRAConfig, DataConfig, TrainingConfig, WandbConfig
from data.dataset import load_and_prepare_dataset, create_dataloaders
from models.model_loader import (
    load_tokenizer,
    create_bnb_config,
    load_base_model,
    setup_lora,
    save_model,
)
from training.trainer import train

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse CLI arguments using HuggingFace's HfArgumentParser.

    HfArgumentParser ADVANTAGE:
    ----------------------------
    Generates argparse arguments automatically from dataclass field definitions.
    Field `help` metadata → --help text.
    Field `default` → argument default.
    Field type annotation → argument type.

    This means: every field in our config dataclasses automatically becomes a
    CLI argument WITHOUT manually defining add_argument() calls.
    """
    parser = HfArgumentParser((ModelConfig, LoRAConfig, DataConfig, TrainingConfig, WandbConfig))

    # Add extra arguments not in dataclasses
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to YAML config file. Overrides individual flags.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        choices=["llama3", "mistral", "chatml", "alpaca", "gemma"],
        help="Prompt template. Auto-detected from model name if not set.",
    )
    parser.add_argument(
        "--merge_after_training",
        action="store_true",
        help="Merge LoRA adapters into base model after training completes.",
    )

    # Parse known args first (allows unknown args to be ignored)
    if "--config_file" in sys.argv:
        # If config file provided, load from YAML first then override with CLI
        args, remaining = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        return args
    else:
        args = parser.parse_args_into_dataclasses()
        return args


def setup_environment(training_config: TrainingConfig) -> None:
    """
    Configure environment for reproducible training.

    REPRODUCIBILITY:
    ----------------
    Neural network training involves randomness from:
      1. Weight initialization (if training from scratch)
      2. Data shuffling order
      3. Dropout mask selection
      4. Sampling (temperature/top-p in generation)
      5. CUDA non-deterministic ops (atomics in parallel reduction)

    For reproducible results across runs, we fix all random seeds.
    NOTE: Full CUDA reproducibility requires CUBLAS_WORKSPACE_CONFIG=:4096:8
    and torch.use_deterministic_algorithms(True), which is slower.
    For production training, exact bit-for-bit reproducibility is usually
    not required — we just want statistically similar results across runs.
    """
    # Python built-in random
    import random
    random.seed(training_config.seed)

    # NumPy (used by datasets for shuffling)
    import numpy as np
    np.random.seed(training_config.seed)

    # PyTorch CPU + CUDA
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    # HuggingFace Transformers (uses this for generation)
    set_seed(training_config.seed)

    # Deterministic CUDA operations (optional, slower but reproducible)
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # TF32 precision on Ampere GPUs (A100, RTX 30/40 series)
    # TF32 uses FP32 range but only 10 bits of mantissa (vs 23 for FP32)
    # Gives ~3× speedup for matmuls with negligible accuracy loss
    # Enabled by default in PyTorch 1.7+, but let's be explicit
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Benchmark cudnn for best convolution algorithms (not relevant for LLMs,
    # but doesn't hurt and sometimes used in preprocessing)
    torch.backends.cudnn.benchmark = False  # False for reproducibility
    torch.backends.cudnn.deterministic = True

    logger.info(f"Random seeds set to {training_config.seed}")
    logger.info(f"TF32 enabled for matmul and cudnn")


def log_system_info() -> None:
    """Log hardware and software configuration at training start."""
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)

    # Python and PyTorch versions
    import platform
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")

    # CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} | "
                f"{props.total_memory / 1e9:.1f} GB | "
                f"Compute {props.major}.{props.minor}"
            )
        logger.info(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        logger.warning("No CUDA GPU available! Training will be very slow on CPU.")

    # Package versions
    try:
        import transformers
        logger.info(f"transformers: {transformers.__version__}")
    except ImportError:
        pass
    try:
        import peft
        logger.info(f"peft: {peft.__version__}")
    except ImportError:
        pass
    try:
        import bitsandbytes as bnb
        logger.info(f"bitsandbytes: {bnb.__version__}")
    except ImportError:
        logger.warning("bitsandbytes not found — 4-bit quantization unavailable!")
    try:
        import flash_attn
        logger.info(f"flash-attn: {flash_attn.__version__}")
    except ImportError:
        logger.warning("flash-attn not found — using PyTorch SDPA instead.")

    logger.info("=" * 60)


def main():
    # ── Parse Config ──────────────────────────────────────────────────────────
    parsed = parse_args()

    # HfArgumentParser returns a tuple of dataclass instances
    if isinstance(parsed, tuple):
        model_config, lora_config, data_config, training_config, wandb_config = parsed[:5]
        extra_args = parsed[5] if len(parsed) > 5 else None
    else:
        model_config = parsed[0]
        lora_config = parsed[1]
        data_config = parsed[2]
        training_config = parsed[3]
        wandb_config = parsed[4]
        extra_args = None

    # Create output and log directories
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ── Environment Setup ─────────────────────────────────────────────────────
    log_system_info()
    setup_environment(training_config)

    logger.info(f"Model: {model_config.model_name_or_path}")
    logger.info(f"Dataset: {data_config.dataset_name}")
    logger.info(f"LoRA rank: {lora_config.lora_r}, alpha: {lora_config.lora_alpha}")
    logger.info(f"Output dir: {training_config.output_dir}")

    # ── Step 1: Load Tokenizer ────────────────────────────────────────────────
    logger.info("\n[1/6] Loading tokenizer...")
    tokenizer = load_tokenizer(model_config)

    # ── Step 2: Prepare Dataset ───────────────────────────────────────────────
    logger.info("\n[2/6] Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(
        data_config=data_config,
        tokenizer=tokenizer,
        model_name=model_config.model_name_or_path,
        prompt_template=getattr(extra_args, "prompt_template", None) if extra_args else None,
    )
    logger.info(f"Train: {len(dataset['train'])} examples")
    logger.info(f"Validation: {len(dataset['validation'])} examples")

    # ── Step 3: Create DataLoaders ────────────────────────────────────────────
    logger.info("\n[3/6] Creating DataLoaders...")

    # Handle sequence packing
    if data_config.use_packing:
        from data.dataset import PackedDataset
        train_ds = PackedDataset(
            dataset["train"],
            tokenizer=tokenizer,
            max_seq_length=data_config.max_seq_length,
        )
        val_ds = PackedDataset(
            dataset["validation"],
            tokenizer=tokenizer,
            max_seq_length=data_config.max_seq_length,
        )
    else:
        train_ds = dataset["train"]
        val_ds = dataset["validation"]

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        tokenizer=tokenizer,
        data_config=data_config,
        training_config=training_config,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Step 4: Load Model ────────────────────────────────────────────────────
    logger.info("\n[4/6] Loading base model...")
    bnb_config = create_bnb_config(lora_config)
    base_model = load_base_model(model_config, lora_config, bnb_config)

    # ── Step 5: Setup LoRA ────────────────────────────────────────────────────
    logger.info("\n[5/6] Setting up LoRA adapters...")
    model, peft_lora_config = setup_lora(base_model, lora_config, tokenizer)

    # Optional: torch.compile for forward pass speedup
    # Disabled by default due to PEFT compatibility issues
    if model_config.torch_compile and hasattr(torch, 'compile'):
        logger.info("Applying torch.compile...")
        model = torch.compile(model, mode='default')

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    logger.info("\n[6/6] Starting training...")
    trained_model = train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        lora_config_params=lora_config,
        wandb_config=wandb_config,
        resume_from_checkpoint=getattr(extra_args, "resume_from_checkpoint", None),
    )

    # ── Post-Training: Merge (Optional) ───────────────────────────────────────
    merge = getattr(extra_args, "merge_after_training", False) or lora_config.merge_adapters
    final_dir = os.path.join(training_config.output_dir, "final_model")

    logger.info(f"\nSaving final model to {final_dir}...")
    save_model(
        model=trained_model,
        tokenizer=tokenizer,
        output_dir=final_dir,
        merge=merge,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {final_dir}")
    if merge:
        logger.info("LoRA adapters merged into base model.")
        logger.info(f"Run inference: python scripts/inference.py --model_path {final_dir}")
    else:
        logger.info("LoRA adapters saved separately.")
        logger.info(f"To merge later: python scripts/merge.py --adapter_path {final_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
