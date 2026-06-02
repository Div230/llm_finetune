"""
training/trainer.py
===================
Complete training loop with full experiment tracking, multi-GPU support,
gradient management, and checkpointing.

This module implements a production-quality training loop that goes beyond
the HuggingFace Trainer abstraction. Understanding this level of detail is
essential for debugging training issues and extending to research settings.

HF TRAINER VS CUSTOM LOOP — WHEN TO USE WHICH:
===============================================
HuggingFace Trainer:
  Pros: One-liner setup, handles distributed training, logging, checkpoints
  Cons: Hard to debug, limited customization, hidden behavior
  USE FOR: Standard fine-tuning where you trust the defaults

Custom Training Loop (this file):
  Pros: Full control, easy to debug, easy to add custom logic
  Cons: More boilerplate, manual distributed training setup
  USE FOR: Research, debugging, custom objectives, curriculum learning

ACCELERATE INTEGRATION:
=======================
We use HuggingFace Accelerate as a thin wrapper over:
  - DistributedDataParallel (DDP) for multi-GPU
  - DeepSpeed integration (optional, for extreme scale)
  - Mixed precision (BF16/FP16)
  - Device placement (CPU/GPU/TPU)

Accelerate's philosophy: write single-GPU code, Accelerate handles distribution.
The key calls are:
  accelerator.prepare(model, optimizer, train_loader, val_loader)
  accelerator.backward(loss)  # handles mixed precision scaling
  accelerator.gather(predictions)  # collect results from all GPUs
"""

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

from configs.config import TrainingConfig, WandbConfig, LoRAConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# TRAINING METRICS TRACKER
# ==============================================================================

@dataclass
class TrainingMetrics:
    """Accumulates metrics over a logging interval for averaged reporting."""
    loss_sum: float = 0.0
    grad_norm_sum: float = 0.0
    n_steps: int = 0
    tokens_sum: int = 0
    time_sum: float = 0.0

    def update(self, loss: float, grad_norm: float, n_tokens: int, elapsed: float):
        self.loss_sum += loss
        self.grad_norm_sum += grad_norm
        self.tokens_sum += n_tokens
        self.time_sum += elapsed
        self.n_steps += 1

    def average(self) -> Dict[str, float]:
        if self.n_steps == 0:
            return {}
        return {
            "train/loss": self.loss_sum / self.n_steps,
            "train/perplexity": math.exp(min(self.loss_sum / self.n_steps, 20)),
            "train/grad_norm": self.grad_norm_sum / self.n_steps,
            "train/tokens_per_sec": self.tokens_sum / max(self.time_sum, 1e-6),
        }

    def reset(self):
        self.loss_sum = 0.0
        self.grad_norm_sum = 0.0
        self.n_steps = 0
        self.tokens_sum = 0
        self.time_sum = 0.0


# ==============================================================================
# WANDB SETUP
# ==============================================================================

def setup_wandb(
    training_config: TrainingConfig,
    wandb_config: WandbConfig,
    lora_config_params: LoRAConfig,
    extra_config: Optional[Dict] = None,
) -> None:
    """
    Initialize Weights & Biases experiment tracking.

    WANDB INTEGRATION PHILOSOPHY:
    ------------------------------
    Log everything that might matter for debugging or reproduction:
      1. All hyperparameters (LR, rank, batch size, etc.)
      2. Hardware info (GPU type, memory)
      3. Model architecture details
      4. Training metrics at every logging_steps
      5. Validation metrics at every eval_steps
      6. Generated text samples (qualitative evaluation)
      7. GPU memory usage
      8. Learning rate over time

    WHY GENERATED SAMPLES MATTER:
    Loss curves can look good even when the model is outputting nonsense.
    Logging actual generated text at regular intervals catches:
      - Mode collapse (repetitive outputs)
      - Format regression (lost instruction following)
      - Language mixing (unexpected language switches)
      - Hallucination patterns
    """
    config = {
        # Training hyperparameters
        "learning_rate": training_config.learning_rate,
        "num_epochs": training_config.num_train_epochs,
        "batch_size_per_device": training_config.per_device_train_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "effective_batch_size": (
            training_config.per_device_train_batch_size
            * training_config.gradient_accumulation_steps
        ),
        "warmup_ratio": training_config.warmup_ratio,
        "weight_decay": training_config.weight_decay,
        "max_grad_norm": training_config.max_grad_norm,
        "lr_scheduler": training_config.lr_scheduler_type,
        "optimizer": training_config.optim,
        "bf16": training_config.bf16,
        "fp16": training_config.fp16,
        # LoRA config
        "lora_r": lora_config_params.lora_r,
        "lora_alpha": lora_config_params.lora_alpha,
        "lora_dropout": lora_config_params.lora_dropout,
        "use_4bit": lora_config_params.use_4bit,
        "quant_type": lora_config_params.bnb_4bit_quant_type,
        "double_quant": lora_config_params.use_double_quantization,
        # GPU info
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "num_gpus": torch.cuda.device_count(),
        **(extra_config or {}),
    }

    run_name = training_config.run_name or f"qlora-r{lora_config_params.lora_r}"

    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=run_name,
        config=config,
        tags=wandb_config.tags or ["qlora", "fine-tuning"],
        # resume="allow": allows resuming interrupted runs
        resume="allow",
    )

    logger.info(f"WandB initialized: {wandb.run.url}")


# ==============================================================================
# OPTIMIZER AND SCHEDULER
# ==============================================================================

def create_optimizer_and_scheduler(
    model: PreTrainedModel,
    training_config: TrainingConfig,
    num_training_steps: int,
) -> Tuple:
    """
    Create optimizer and learning rate scheduler.

    OPTIMIZER SELECTION:
    --------------------
    bitsandbytes provides quantized optimizer variants:
      - adamw_bnb_8bit: 8-bit Adam, good for full fine-tuning
      - paged_adamw_32bit: 32-bit Adam with CPU paging (best for QLoRA)
      - paged_adamw_8bit: 8-bit Adam with CPU paging
      - adamw_torch: PyTorch native (fastest when memory allows)
      - adamw_torch_fused: Fused kernel (10% faster, requires PyTorch 2.0+)

    FOR QLORA (our setup):
    LoRA params are only ~84M for 7B model. Adam states for 84M params:
      FP32: 84M × 2 × 4 bytes = 672 MB (manageable!)
    So for QLoRA, paged_adamw_32bit is recommended for numerical stability.
    8-bit optimizer is more useful for full fine-tuning (7B params → 56 GB optimizer states).

    PARAMETER GROUPING:
    -------------------
    We use different weight decay for different parameter types:
      - Weight matrices (nn.Linear.weight): apply weight decay
      - Bias terms (nn.Linear.bias): NO weight decay (standard practice)
      - Normalization parameters (LayerNorm): NO weight decay
    This follows the AdamW paper's recommendation.

    LR SCHEDULER:
    -------------
    Cosine with warmup is the gold standard for LLM fine-tuning:
      Phase 1 (warmup): LR goes from 0 → peak over warmup_steps
      Phase 2 (decay): LR follows cosine curve from peak → 0 over remaining steps
      
    The cosine decay ensures a smooth final convergence.
    """
    # ── Parameter Groups (with/without weight decay) ──────────────────────────
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        # Weight decay on weights, not biases or normalization params
        if any(nd in name for nd in ["bias", "layer_norm", "layernorm", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": training_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # ── Optimizer Creation ────────────────────────────────────────────────────
    optimizer_name = training_config.optim.lower()

    if "paged_adamw_8bit" in optimizer_name:
        try:
            import bitsandbytes as bnb
            logger.info("Using 8bit adamw paging using bitsandbytes.")
            optimizer = bnb.optim.PagedAdamW8bit(
                optimizer_grouped_parameters,
                lr=training_config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        except ImportError:
            logger.warning("bitsandbytes not available. Using PyTorch AdamW.")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=training_config.learning_rate
            )

    elif "paged_adamw_32bit" in optimizer_name:
        try:
            import bitsandbytes as bnb
            logger.info("Using 32bit adamw paging using bitsandbytes.")
            optimizer = bnb.optim.PagedAdamW32bit(
                optimizer_grouped_parameters,
                lr=training_config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        except ImportError:
            logger.warning("bitsandbytes not available. Using PyTorch AdamW.")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=training_config.learning_rate
            )

    elif "adamw_torch_fused" in optimizer_name:
        # Fused kernel: combines multiple CUDA ops into one → ~10% speedup
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_config.learning_rate,
            fused=True,
        )

    else:  # Default: standard AdamW
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    warmup_steps = int(num_training_steps * training_config.warmup_ratio)

    # get_scheduler returns the appropriate scheduler from transformers
    # Supported: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial',
    #            'constant', 'constant_with_warmup'
    scheduler = get_scheduler(
        name=training_config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    logger.info(
        f"Optimizer: {optimizer_name}, "
        f"Scheduler: {training_config.lr_scheduler_type}, "
        f"Warmup steps: {warmup_steps}/{num_training_steps}"
    )

    return optimizer, scheduler


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(
    model: PreTrainedModel,
    val_loader: DataLoader,
    accelerator: Accelerator,
    max_batches: int = 100,
) -> Dict[str, float]:
    """
    Run validation loop and compute metrics.

    EVALUATION BEST PRACTICES:
    ---------------------------
    1. @torch.no_grad(): Disables gradient computation → 2× faster, less memory
    2. model.eval(): Disables dropout, uses running BatchNorm stats
    3. accelerator.gather(): Collects predictions from all GPUs in multi-GPU setup
    4. Limit batches: Full validation can take too long. Sample 100 batches.

    PERPLEXITY:
    -----------
    Perplexity = exp(cross_entropy_loss)
    Intuition: "How many tokens does the model consider equally likely at each step?"
    Perplexity = 10 means: on average, the model is as uncertain as if it had 10 equally
    likely options at every token position.
    Lower is better. For reference:
      Language model on English: ~20-50 (decent)
      GPT-4 on standard benchmarks: ~5-15
      Random model on 50k vocab: ~50,000

    PERPLEXITY MATH:
    ----------------
    PP = exp(1/N × Σᵢ -log P(xᵢ|x₁,...,xᵢ₋₁))
    where N = total tokens, xᵢ = token at position i
    = exp(average_cross_entropy_loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break

        # Forward pass (gradients disabled by @torch.no_grad())
        outputs = model(**batch)
        loss = outputs.loss

        # Gather losses from all GPUs (for multi-GPU)
        gathered_loss = accelerator.gather(loss.unsqueeze(0))
        total_loss += gathered_loss.mean().item()

        # Count non-padding tokens (those with label != -100)
        # For accurate perplexity, we need token count not sequence count
        n_real_tokens = (batch["labels"] != -100).sum().item()
        total_tokens += n_real_tokens
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Clamp to prevent overflow

    model.train()  # Restore training mode

    return {
        "val/loss": avg_loss,
        "val/perplexity": perplexity,
    }


# ==============================================================================
# GENERATE SAMPLES FOR QUALITATIVE EVALUATION
# ==============================================================================

@torch.no_grad()
def generate_sample(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list,
    max_new_tokens: int = 200,
) -> list:
    """
    Generate text samples for qualitative monitoring.

    WHY QUALITATIVE MONITORING?
    ---------------------------
    Quantitative metrics (loss, perplexity) are necessary but not sufficient.
    A model can have excellent loss but still:
      - Repeat the same phrase endlessly
      - Switch languages mid-sentence
      - Follow the instruction format but give nonsensical answers
      - Generate correct format but wrong factual content

    Logging actual generated text to WandB every N steps provides
    early warning of these failure modes.

    GENERATION DURING TRAINING:
    ---------------------------
    We switch to eval mode, generate, then switch back to train mode.
    This disables dropout for clean generations.
    We also move tokenizer to left-padding for generation
    (right-padding causes issues: the model starts generating from the padding side).
    """
    model.eval()

    # Switch to left padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    generated_texts = []
    for prompt in prompts[:3]:  # Limit to 3 samples per generation call
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(model.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode only the new tokens (not the input prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)

    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    model.train()

    return generated_texts


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: TrainingConfig,
    lora_config_params: LoRAConfig,
    wandb_config: WandbConfig,
    resume_from_checkpoint: Optional[str] = None,
) -> PreTrainedModel:
    """
    Main training loop with full monitoring and checkpointing.

    ACCELERATE MULTI-GPU OVERVIEW:
    ================================
    ACCELERATE wraps standard PyTorch code for distributed training.
    
    Single GPU code (what you write):
      model = MyModel()
      optimizer = AdamW(model.parameters())
      for batch in dataloader:
          outputs = model(batch)
          loss = outputs.loss
          loss.backward()
          optimizer.step()
    
    Multi-GPU code (what Accelerate generates internally):
      - Wraps model in DistributedDataParallel (DDP)
      - Each GPU runs a copy of the model on its subset of data
      - Gradients are all-reduced (summed + averaged) across GPUs after backward()
      - All GPUs update weights synchronously → equivalent to larger batch size
      
    DDP ALL-REDUCE:
      After backward(): Σ gradients across all GPUs / num_GPUs
      This gives: gradient as if trained on N_GPUs × batch_size data
      Communication cost: ~2 × model_size bytes per step (ring all-reduce)
      Overlap: DDP overlaps gradient communication with backward computation (bucket comm)
    
    HOW TO USE ACCELERATE FOR MULTI-GPU:
      Single GPU: python train.py
      Multi-GPU:  accelerate launch --num_processes 4 train.py
      DeepSpeed:  accelerate launch --config_file ds_config.yaml train.py

    TRAINING LOOP DESIGN:
    =====================
    Key decisions in this loop:
    1. Gradient accumulation: accumulate N batches before optimizer.step()
    2. Mixed precision: backward() through autocast for BF16 gradients
    3. Gradient clipping: clip_grad_norm_ before optimizer.step()
    4. Checkpointing: save_pretrained() every save_steps
    5. Evaluation: run val loop every eval_steps
    6. WandB logging: log metrics every logging_steps

    Args:
        model: PEFT model with LoRA adapters
        tokenizer: Model tokenizer
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        training_config: Training hyperparameters
        lora_config_params: LoRA configuration
        wandb_config: WandB settings
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        Trained model
    """
    # ── Accelerator Setup ─────────────────────────────────────────────────────
    # Accelerator handles:
    #   - Mixed precision (BF16/FP16)
    #   - Multi-GPU distribution
    #   - Gradient scaling (for FP16)
    #   - Device placement
    from accelerate import GradScalerKwargs
    kwargs_handlers = []

    if training_config.fp16:
        # GradScaler: scales loss up before backward, scales gradients down after
        # Prevents FP16 underflow (small gradients becoming 0 in FP16)
        # NOT needed for BF16 (BF16 has same range as FP32)
        kwargs_handlers.append(GradScalerKwargs(
            init_scale=2**16,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=2000,
        ))

    mixed_precision = "bf16" if training_config.bf16 else ("fp16" if training_config.fp16 else "no")

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        log_with="wandb" if training_config.report_to == "wandb" else None,
        kwargs_handlers=kwargs_handlers if kwargs_handlers else None,
    )

    # Set reproducibility seed
    set_seed(training_config.seed)

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / training_config.gradient_accumulation_steps
    )
    num_training_steps = training_config.num_train_epochs * num_update_steps_per_epoch

    logger.info(
        f"Training steps per epoch: {num_update_steps_per_epoch}, "
        f"Total training steps: {num_training_steps}"
    )

    # ── Optimizer and Scheduler ───────────────────────────────────────────────
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, training_config, num_training_steps
    )

    # ── Accelerator Prepare ───────────────────────────────────────────────────
    # This single call handles all distributed training setup:
    # - Wraps model in DDP for multi-GPU
    # - Distributes data across GPUs
    # - Handles device placement
    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, val_loader, scheduler
    )

    # ── WandB Initialization ──────────────────────────────────────────────────
    if accelerator.is_main_process and training_config.report_to == "wandb":
        setup_wandb(training_config, wandb_config, lora_config_params, {
            "num_training_steps": num_training_steps,
            "num_update_steps_per_epoch": num_update_steps_per_epoch,
        })

    # ── Resume from Checkpoint ────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0

    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        # Extract step number from checkpoint directory name
        checkpoint_name = os.path.basename(resume_from_checkpoint)
        if "step" in checkpoint_name:
            global_step = int(checkpoint_name.split("step")[-1])
        logger.info(f"Resumed at global step {global_step}")

    # ── Gradient Checkpointing ────────────────────────────────────────────────
    if training_config.gradient_checkpointing:
        # enable_input_require_grads: required for gradient flow through
        # the quantized frozen base model to LoRA adapters
        # Without this, gradients don't flow back through the embedding layer
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # ── torch.compile (optional) ──────────────────────────────────────────────
    # TORCH.COMPILE INTERNALS:
    # ------------------------
    # Compiles the model using TorchDynamo → AOT Autograd → TorchInductor
    # TorchDynamo: captures Python bytecode into FX graph (handles control flow)
    # AOT Autograd: traces joint forward+backward graph
    # TorchInductor: generates optimized Triton/CUDA kernels
    #
    # mode options:
    #   'default': balanced (good first choice)
    #   'reduce-overhead': minimize Python overhead (better for small models)
    #   'max-autotune': maximize performance (long compilation time)
    
    # NOTE: torch.compile has known issues with some PEFT patterns.
    # Uncomment cautiously and test for correctness.
    # if getattr(training_config, 'torch_compile', False):
    #     model = torch.compile(model, mode='default')
    #     logger.info("torch.compile enabled")

    # ── Training Metrics ──────────────────────────────────────────────────────
    metrics = TrainingMetrics()
    best_val_loss = float("inf")

    # Sample prompts for qualitative evaluation
    eval_prompts = [
        "What is machine learning? Explain it simply.",
        "Write a Python function to sort a list of dictionaries by a key.",
        "What are the main differences between transformers and RNNs?",
    ]

    # ── MAIN TRAINING LOOP ────────────────────────────────────────────────────
    logger.info(f"{'='*60}")
    logger.info("STARTING TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"  Epochs: {training_config.num_train_epochs}")
    logger.info(f"  Steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"  Total steps: {num_training_steps}")
    logger.info(f"  Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    logger.info(f"{'='*60}")

    model.train()

    for epoch in range(start_epoch, training_config.num_train_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()

        for step, batch in enumerate(train_loader):
            step_start_time = time.time()

            # GRADIENT ACCUMULATION CONTEXT:
            # accelerator.accumulate() context manager handles gradient accumulation.
            # When step % gradient_accumulation_steps != 0:
            #   - DDP: skips gradient synchronization (no_sync context) → faster
            # When step % gradient_accumulation_steps == 0:
            #   - DDP: runs gradient all-reduce → update step
            with accelerator.accumulate(model):

                # ── Forward Pass ──────────────────────────────────────────────
                # MIXED PRECISION AUTOCAST:
                # Inside this context, PyTorch automatically casts:
                #   - Matrix multiplications → BF16 (fast, low memory)
                #   - Reductions (sum, mean) → FP32 (accurate)
                #   - Comparisons → FP32
                # This is the key operation for mixed precision training.
                # Memory savings: activations stored in BF16 (2 bytes vs 4 for FP32)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs.loss

                # ── Backward Pass ─────────────────────────────────────────────
                # accelerator.backward():
                # - For BF16: standard loss.backward() (no scaling needed)
                # - For FP16: loss_scaler.scale(loss).backward() + unscale
                #   FP16 SCALING EXPLAINED:
                #   FP16 gradients can underflow to 0 for small values.
                #   Solution: multiply loss by scale_factor (e.g., 2^16) before backward
                #   → gradients scaled up by scale_factor
                #   → after backward, divide by scale_factor (unscale)
                #   → if gradient overflow detected: reduce scale_factor, skip update
                accelerator.backward(loss)

                # ── Gradient Clipping ─────────────────────────────────────────
                # After unscaling (for FP16), clip gradient norms.
                # accelerator.clip_grad_norm_() handles the multi-GPU case:
                # In DDP, gradients are already all-reduced, so clipping is applied
                # to the globally summed gradient.
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(),
                        training_config.max_grad_norm,
                    )
                else:
                    grad_norm = 0.0

                # ── Optimizer Step ────────────────────────────────────────────
                optimizer.step()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()  # Clear gradients for next accumulation

            # ── Track per-step metrics ─────────────────────────────────────────
            step_elapsed = time.time() - step_start_time
            n_tokens = (batch["labels"] != -100).sum().item()

            metrics.update(
                loss=loss.item(),
                grad_norm=float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                n_tokens=n_tokens,
                elapsed=step_elapsed,
            )
            epoch_loss += loss.item()

            # Only count update steps (not accumulation micro-steps)
            if accelerator.sync_gradients:
                global_step += 1

                # ── Logging ───────────────────────────────────────────────────
                if global_step % training_config.logging_steps == 0:
                    avg_metrics = metrics.average()
                    current_lr = scheduler.get_last_lr()[0]

                    # GPU memory stats
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        avg_metrics["system/gpu_memory_allocated_gb"] = allocated
                        avg_metrics["system/gpu_memory_reserved_gb"] = reserved
                        avg_metrics["system/gpu_utilization"] = _get_gpu_utilization()

                    avg_metrics["train/learning_rate"] = current_lr
                    avg_metrics["train/epoch"] = epoch + step / len(train_loader)
                    avg_metrics["train/global_step"] = global_step

                    if accelerator.is_main_process:
                        if training_config.report_to == "wandb" and wandb.run:
                            wandb.log(avg_metrics, step=global_step)

                        logger.info(
                            f"Step {global_step}/{num_training_steps} | "
                            f"Loss: {avg_metrics['train/loss']:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tokens/s: {avg_metrics.get('train/tokens_per_sec', 0):.0f} | "
                            f"GPU: {avg_metrics.get('system/gpu_memory_allocated_gb', 0):.1f}GB"
                        )

                    metrics.reset()

                # ── Evaluation ────────────────────────────────────────────────
                if global_step % training_config.eval_steps == 0:
                    logger.info(f"Running evaluation at step {global_step}...")
                    val_metrics = evaluate(model, val_loader, accelerator)

                    if accelerator.is_main_process:
                        val_metrics["val/global_step"] = global_step

                        if training_config.report_to == "wandb" and wandb.run:
                            # Log generated samples to WandB for qualitative eval
                            samples = generate_sample(
                                accelerator.unwrap_model(model),
                                tokenizer,
                                eval_prompts,
                            )
                            sample_table = wandb.Table(
                                columns=["prompt", "generation"],
                                data=[[p, s] for p, s in zip(eval_prompts, samples)],
                            )
                            val_metrics["val/generated_samples"] = sample_table
                            wandb.log(val_metrics, step=global_step)

                        logger.info(
                            f"Validation - Loss: {val_metrics['val/loss']:.4f} | "
                            f"Perplexity: {val_metrics['val/perplexity']:.2f}"
                        )

                        # Save best model
                        if val_metrics["val/loss"] < best_val_loss:
                            best_val_loss = val_metrics["val/loss"]
                            best_model_path = os.path.join(
                                training_config.output_dir, "best_model"
                            )
                            _save_checkpoint(
                                model, tokenizer, best_model_path, accelerator, global_step
                            )
                            logger.info(
                                f"New best model saved! val_loss={best_val_loss:.4f}"
                            )

                # ── Periodic Checkpointing ────────────────────────────────────
                if global_step % training_config.save_steps == 0:
                    checkpoint_path = os.path.join(
                        training_config.output_dir, f"checkpoint-step{global_step}"
                    )
                    _save_checkpoint(
                        model, tokenizer, checkpoint_path, accelerator, global_step
                    )
                    _cleanup_old_checkpoints(
                        training_config.output_dir,
                        training_config.save_total_limit,
                    )

        # ── End of Epoch ──────────────────────────────────────────────────────
        epoch_elapsed = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch + 1}/{training_config.num_train_epochs} complete | "
            f"Avg Loss: {avg_epoch_loss:.4f} | "
            f"Time: {epoch_elapsed:.0f}s"
        )

        if accelerator.is_main_process and training_config.report_to == "wandb" and wandb.run:
            wandb.log({
                "epoch/loss": avg_epoch_loss,
                "epoch/time_seconds": epoch_elapsed,
                "epoch/num": epoch + 1,
            }, step=global_step)

    # ── Final Save ────────────────────────────────────────────────────────────
    final_path = os.path.join(training_config.output_dir, "final_model")
    _save_checkpoint(model, tokenizer, final_path, accelerator, global_step)
    logger.info(f"Training complete. Final model saved to {final_path}")

    if accelerator.is_main_process and training_config.report_to == "wandb" and wandb.run:
        wandb.finish()

    return accelerator.unwrap_model(model)


# ==============================================================================
# CHECKPOINT UTILITIES
# ==============================================================================

def _save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    accelerator: Accelerator,
    step: int,
) -> None:
    """Save model checkpoint. Only main process saves."""
    if not accelerator.is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Unwrap from DDP wrapper to get actual model
    unwrapped_model = accelerator.unwrap_model(model)

    # Save LoRA adapters (not the full model — just the small adapter weights)
    # This calls PEFT's save_pretrained which saves adapter_config.json + adapter_model.safetensors
    unwrapped_model.save_pretrained(
        output_dir,
        save_function=accelerator.save,  # Use accelerator's save for DDP compatibility
        safe_serialization=True,  # safetensors format
    )
    tokenizer.save_pretrained(output_dir)

    # Save training state for resumption (optimizer + scheduler states)
    accelerator.save_state(os.path.join(output_dir, "training_state"))

    logger.info(f"Checkpoint saved: {output_dir} (step {step})")


def _cleanup_old_checkpoints(output_dir: str, keep_total: int) -> None:
    """Remove old checkpoints, keeping only the most recent keep_total."""
    import glob
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-step*")),
        key=lambda x: int(x.split("step")[-1]) if x.split("step")[-1].isdigit() else 0,
    )
    if len(checkpoints) > keep_total:
        for old_checkpoint in checkpoints[:-keep_total]:
            import shutil
            shutil.rmtree(old_checkpoint, ignore_errors=True)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")


def _get_gpu_utilization() -> float:
    """Get GPU utilization percentage using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0.0
