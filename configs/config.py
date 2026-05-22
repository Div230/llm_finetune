"""
configs/config.py
=================
Central configuration system for the entire fine-tuning pipeline.

WHY DATACLASSES?
----------------
We use Python dataclasses + HuggingFace's HfArgumentParser instead of raw dicts or YAML for
several important reasons:
  1. Type safety: IDE autocomplete + runtime type hints catch bugs early
  2. CLI integration: HfArgumentParser auto-generates argparse from dataclass fields
  3. Serialization: dataclasses serialize cleanly to JSON/YAML for experiment tracking
  4. Validation: post_init hooks validate interdependencies between fields
  5. Documentation: field descriptions become CLI --help text automatically

PRODUCTION USAGE:
  In production, configs are usually stored in YAML/JSON and loaded at job submission time.
  The dataclass pattern here mirrors Meta's fairseq, EleutherAI's lm-evaluation-harness,
  and the HuggingFace Trainer approach, all of which follow this same pattern.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for the base model and architecture choices.

    MODEL SELECTION GUIDE:
    ----------------------
    The choice of base model is the single most impactful decision in fine-tuning.
    Consider:
      - Parameter count: Larger = smarter but slower + more memory
      - Context length: How long are your training examples?
      - License: Commercial use? Check carefully (Llama 3, Gemma require agreements)
      - Architecture: Does it support Flash Attention 2? RoPE scaling?
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2-7B-Instruct",
        metadata={
            "help": (
                "HuggingFace model ID or local path. "
                "Good defaults: "
                "'meta-llama/Meta-Llama-3-8B-Instruct' (best for instruction following), "
                "'mistralai/Mistral-7B-Instruct-v0.3' (strong, less restricted), "
                "'Qwen/Qwen2-7B-Instruct' (excellent multilingual), "
                "'google/gemma-2-9b-it' (competitive, Apache 2.0 license)"
            )
        }
    )

    # ── Attention Backend ─────────────────────────────────────────────────────
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "FLASH ATTENTION 2: The single most impactful optimization for transformer training.\n"
                "\n"
                "WHAT IT IS:\n"
                "  Standard attention computes the full N×N attention matrix in HBM (GPU VRAM).\n"
                "  FlashAttention2 uses tiling + recomputation to avoid materializing this matrix,\n"
                "  keeping all intermediate values in SRAM (fast on-chip cache).\n"
                "\n"
                "MATHEMATICAL INTUITION:\n"
                "  Standard: O(N²) memory, O(N²) FLOPs for N tokens\n"
                "  FlashAttn2: O(N) memory (tiles of size ~64), same O(N²) FLOPs but\n"
                "  with 10-100× better memory bandwidth utilization because SRAM is\n"
                "  ~100× faster than HBM.\n"
                "\n"
                "MEMORY IMPACT:\n"
                "  For sequence length 2048: saves ~1-2 GB on 7B model\n"
                "  For sequence length 8192: saves ~16-32 GB (necessary at this length)\n"
                "\n"
                "SPEED IMPACT:\n"
                "  2-4× faster than standard PyTorch attention on modern GPUs\n"
                "  Scales sub-quadratically with sequence length in practice\n"
                "\n"
                "TRADEOFFS:\n"
                "  - Requires Ampere+ GPU (A100, RTX 3090+, your RTX 4050 supports it)\n"
                "  - Requires flash-attn package (C++ compile, takes ~5 min)\n"
                "  - Slightly different numerical outputs (non-deterministic at fp16)\n"
                "\n"
                "PRODUCTION: Used universally in all modern LLM training. Non-negotiable."
            )
        }
    )

    # ── torch.compile ─────────────────────────────────────────────────────────
    torch_compile: bool = field(
        default=False,
        metadata={
            "help": (
                "TORCH.COMPILE (TorchDynamo + TorchInductor):\n"
                "\n"
                "WHAT IT IS:\n"
                "  PyTorch 2.0+ feature that JIT-compiles your model to optimized Triton/CUDA\n"
                "  kernels using a graph capture + backend compilation pipeline.\n"
                "  Think: like XLA for JAX but for PyTorch, with Python flexibility preserved.\n"
                "\n"
                "HOW IT WORKS:\n"
                "  1. TorchDynamo: Captures Python bytecode into FX graphs\n"
                "  2. AOT Autograd: Traces both forward + backward into a single graph\n"
                "  3. TorchInductor: Generates optimized Triton kernels from the graph\n"
                "\n"
                "SPEED IMPACT:\n"
                "  Typically 10-30% speedup on training, up to 2× on inference\n"
                "  Fuses operations: e.g., LayerNorm + GELU into single kernel\n"
                "\n"
                "TRADEOFFS:\n"
                "  - First call triggers compilation (30-120 sec warmup)\n"
                "  - Graph breaks occur with dynamic control flow → reduced benefit\n"
                "  - Not fully compatible with some PEFT patterns (hence default=False)\n"
                "  - Incompatible with gradient checkpointing in some configs\n"
                "\n"
                "PRODUCTION: Used in inference; training usage growing but still experimental."
            )
        }
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow execution of model repo code. Required for some models like Qwen."}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to cache downloaded model weights."}
    )


# ==============================================================================
# QLORA / PEFT CONFIGURATION
# ==============================================================================

@dataclass
class LoRAConfig:
    """
    QLoRA and LoRA adapter configuration.

    LORA — LOW-RANK ADAPTATION:
    ===========================
    WHAT IT IS:
      LoRA is a parameter-efficient fine-tuning technique that inserts trainable
      low-rank matrices into frozen transformer layers instead of updating all weights.

    WHY IT EXISTS:
      Fine-tuning a 7B model's full weights requires:
        - 7B params × 4 bytes (fp32) = 28 GB just for weights
        - + 28 GB for gradients
        - + 84 GB for Adam optimizer states (m + v vectors, fp32)
        = ~140 GB total. Impossible on any single consumer GPU.

      LoRA insight: The "task-specific knowledge" learned during fine-tuning lives in a
      low-rank subspace of the weight update matrix ΔW. We don't need to update the
      full d×d matrix — we can approximate ΔW ≈ B·A where:
        - A ∈ ℝ^(r×d): projects input DOWN to rank r
        - B ∈ ℝ^(d×r): projects back UP to full dimension
        - r << d (e.g., r=16, d=4096)

    MATHEMATICAL INTUITION:
      Original: h = W₀x (frozen)
      LoRA:     h = W₀x + (α/r)·BAx
        - W₀: original frozen weight matrix d×d (or d_out×d_in)
        - B: d×r matrix, initialized to zeros
        - A: r×d matrix, initialized from N(0, σ²)
        - α: scaling factor (usually = r, making α/r = 1)
        - Why B=0 init? Ensures LoRA output is zero at start → training begins from
          pretrained behavior, not random noise.

    MEMORY IMPACT:
      LoRA parameters for a 7B model with r=16:
        - Each weight matrix: 2 matrices of size (d, r) ≈ 4096×16×2 = 131K params
        - ~20 attention/MLP matrices per layer × 32 layers = ~640 LoRA modules
        - Total LoRA params: ~640 × 131K = ~84M params
        - vs 7B base params → only 1.2% of parameters are trained!
      Memory for gradients + Adam states: 84M × 12 bytes ≈ ~1 GB instead of ~140 GB

    COMPUTATIONAL IMPACT:
      Forward pass overhead: BAx adds 2 matmuls of size (r, d) = tiny
      Backward pass: Only through B, A matrices (base model frozen)
      Speed overhead: ~5-10% slower than frozen inference, vs 3-5× slower for full FT

    TRADEOFFS:
      Pros: Dramatically lower memory, faster training, easy adapter swapping
      Cons: Slightly lower performance ceiling than full fine-tuning for complex tasks.
      For most tasks with <100K examples, LoRA matches full FT quality.

    QLORA = QLoRA = Quantized LoRA:
      Uses 4-bit quantized base model + LoRA on top.
      The base model weights are stored in 4-bit NF4 format (see below).
      Only LoRA adapters (r=16 matrices) are trained in BF16/FP16.
      This allows fine-tuning 65B models on a single A100-80GB.
    """
    # ── Core LoRA Hyperparameters ─────────────────────────────────────────────
    lora_r: int = field(
        default=16,
        metadata={
            "help": (
                "LoRA rank r. Controls how many dimensions the low-rank subspace has.\n"
                "RANK SELECTION GUIDE:\n"
                "  r=4:   Ultra-lightweight, good for style/tone shifts, short fine-tunes\n"
                "  r=8:   Good default for most tasks, standard in many papers\n"
                "  r=16:  Better for complex tasks, instruction following (recommended)\n"
                "  r=32:  Complex tasks with diverse outputs, more LoRA capacity\n"
                "  r=64:  Near full-FT capacity, rarely needed\n"
                "MEMORY: Each doubling of r doubles LoRA parameter count (and Adam states)\n"
                "MATH: ΔW ≈ BA where B∈ℝ^{d×r}, A∈ℝ^{r×k}. rank=r controls expressivity."
            )
        }
    )

    lora_alpha: int = field(
        default=16,
        metadata={
            "help": (
                "LoRA alpha scaling factor. Controls the magnitude of LoRA updates.\n"
                "FORMULA: effective_update = (alpha/r) * B*A\n"
                "COMMON PRACTICE:\n"
                "  alpha = r: scaling = 1.0 (balanced, no amplification)\n"
                "  alpha = 2r: scaling = 2.0 (amplified updates, faster learning)\n"
                "  alpha = r/2: smaller updates, more conservative\n"
                "INTUITION: Higher alpha → LoRA adapters have more influence on predictions\n"
                "          but may overfit faster. Lower alpha → more stable but slower.\n"
                "RECOMMENDATION: Start with alpha=r (so alpha=16 if r=16)."
            )
        }
    )

    lora_dropout: float = field(
        default=0.05,
        metadata={
            "help": (
                "Dropout probability applied to LoRA layers during training.\n"
                "REGULARIZATION: Randomly zeroes LoRA activations to prevent overfitting.\n"
                "Small datasets: 0.05-0.1 (more regularization needed)\n"
                "Large datasets: 0.0-0.05\n"
                "INTERACTION: Works ALONGSIDE weight decay, not instead of it."
            )
        }
    )

    # ── Target Modules ────────────────────────────────────────────────────────
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Which weight matrices to apply LoRA to. Default=None means auto-detect.\n"
                "\n"
                "THEORY: LoRA is most effective on attention projection matrices because\n"
                "attention weights encode relational knowledge that needs task-specific\n"
                "adjustment. MLP matrices encode factual knowledge (less task-specific).\n"
                "\n"
                "COMMON CONFIGURATIONS:\n"
                "  Minimal (attention only):\n"
                "    ['q_proj', 'v_proj']\n"
                "    Why q+v: Query determines what to attend to, Value what to output\n"
                "    Key is less impactful (confirmed by ablations in LoRA paper)\n"
                "\n"
                "  Standard (all attention):\n"
                "    ['q_proj', 'k_proj', 'v_proj', 'o_proj']\n"
                "\n"
                "  Full (attention + MLP) — best quality:\n"
                "    ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']\n"
                "\n"
                "  Llama/Mistral module names: q_proj, k_proj, v_proj, o_proj,\n"
                "                              gate_proj, up_proj, down_proj\n"
                "  Qwen module names:          c_attn, c_proj, w1, w2\n"
                "  Falcon module names:        query_key_value, dense, dense_h_to_4h, dense_4h_to_h\n"
                "\n"
                "MEMORY IMPACT: Each additional target module adds ~(d×r + r×k) params"
            )
        }
    )

    bias: str = field(
        default="none",
        metadata={
            "help": (
                "Which biases to add LoRA to. Options: 'none', 'all', 'lora_only'.\n"
                "'none' is standard — biases have minimal impact and waste parameters."
            )
        }
    )

    task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "PEFT task type. CAUSAL_LM for autoregressive models (GPT-style)."}
    )

    # ── Quantization (the Q in QLoRA) ─────────────────────────────────────────
    use_4bit: bool = field(
        default=True,
        metadata={
            "help": (
                "NF4 4-BIT QUANTIZATION (the Q in QLoRA):\n"
                "\n"
                "WHAT IT IS:\n"
                "  Store model weights in 4-bit integers instead of 16/32-bit floats.\n"
                "  Each weight uses 4 bits → 8× less memory than FP32, 4× less than BF16.\n"
                "\n"
                "NF4 (NormalFloat4) vs INT4:\n"
                "  Standard INT4: uniform grid [-8, -7, ..., 7, 8] — poor for weights\n"
                "  NF4: Non-uniform grid optimized for normally-distributed weights.\n"
                "  Why? Neural network weights approximately follow N(0, σ²).\n"
                "  NF4 places quantization levels at points that minimize quantization\n"
                "  error for this distribution (more levels near 0, fewer at extremes).\n"
                "\n"
                "  MATHEMATICAL INTUITION:\n"
                "  NF4 levels = quantiles of N(0,1) distribution mapped to [-1, 1]:\n"
                "  levels = {x : Φ(x) ∈ {0, 1/15, 2/15, ..., 15/15}}\n"
                "  where Φ is the standard normal CDF. This is information-theoretically\n"
                "  optimal for normally distributed inputs.\n"
                "\n"
                "MEMORY IMPACT:\n"
                "  7B model × 4 bits/param = 3.5 GB (vs 14 GB in BF16 or 28 GB in FP32)\n"
                "  Enables fitting 70B models on single A100-80GB with QLoRA!\n"
                "\n"
                "COMPUTATIONAL IMPACT:\n"
                "  Weights are dequantized to BF16 DURING forward pass computation.\n"
                "  The 4-bit storage is in VRAM; compute happens in BF16.\n"
                "  This means: no loss in numerical precision during actual computation,\n"
                "  only at storage time. Very elegant design.\n"
                "  Overhead: ~10-20% slower than pure BF16 due to dequant ops.\n"
                "\n"
                "TRADEOFFS:\n"
                "  Pros: 4× memory reduction, enables large model fine-tuning\n"
                "  Cons: Slight quality degradation vs BF16 (usually <1 perplexity point)\n"
                "        Not supported on all hardware (needs CUDA + bitsandbytes)\n"
                "\n"
                "PRODUCTION: Standard for consumer GPU fine-tuning. Not used in datacenter\n"
                "training from scratch, but common for fine-tuning + inference serving."
            )
        }
    )

    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type: 'nf4' (recommended) or 'fp4' (less optimal)"}
    )

    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": (
                "BF16 VS FP16 — A Critical Choice:\n"
                "\n"
                "FP16 (Half Precision):\n"
                "  Format: 1 sign bit, 5 exponent bits, 10 mantissa bits\n"
                "  Range: ~±65,504\n"
                "  Problem: Small exponent range causes OVERFLOW during training\n"
                "  (loss scaling needed to prevent gradients vanishing to 0)\n"
                "  Overflow example: loss × 1000 → overflow → NaN → training collapse\n"
                "\n"
                "BF16 (Brain Float 16 — Google Brain format):\n"
                "  Format: 1 sign bit, 8 exponent bits, 7 mantissa bits\n"
                "  Range: same as FP32 (~±3.4×10³⁸) — no overflow risk!\n"
                "  Tradeoff: Less precision (7 mantissa bits vs 10 in FP16)\n"
                "  But LLM training is surprisingly robust to reduced mantissa precision\n"
                "\n"
                "WHY BF16 IS BETTER FOR LLM TRAINING:\n"
                "  1. No loss scaling needed (range matches FP32)\n"
                "  2. Gradients and activations can be large without overflow\n"
                "  3. Supported natively on A100, H100, RTX 3090/4090 (and your RTX 4050)\n"
                "\n"
                "WHEN TO USE FP16:\n"
                "  On older GPUs (V100, GTX 1080 Ti) that don't support BF16\n"
                "  Must use GradScaler with FP16!\n"
                "\n"
                "MEMORY: Both FP16 and BF16 are 2 bytes/value (vs 4 for FP32)\n"
                "PRODUCTION: All modern LLM training uses BF16. Use 'float16' as fallback."
            )
        }
    )

    use_double_quantization: bool = field(
        default=True,
        metadata={
            "help": (
                "DOUBLE QUANTIZATION (QLoRA innovation):\n"
                "\n"
                "WHAT IT IS:\n"
                "  4-bit quantization requires storing a scaling factor for each block\n"
                "  of weights (to normalize them before quantization). These scale factors\n"
                "  themselves take up memory. Double quantization quantizes THESE scale\n"
                "  factors too.\n"
                "\n"
                "HOW IT WORKS:\n"
                "  Normal 4-bit: weights quantized in blocks of 64, each block has one\n"
                "  FP32 scale factor → 32 bits per 64 params = 0.5 bits extra overhead\n"
                "\n"
                "  Double quant: quantize the scale factors themselves to 8-bit,\n"
                "  then store one FP32 scale-of-scales per 256 scale factors.\n"
                "  → Only 8 bits per 64 params + 32 bits per 256×64 params\n"
                "  = 8/64 + 32/(256×64) = 0.125 + 0.002 = ~0.127 bits overhead\n"
                "  vs 0.5 bits without double quant\n"
                "\n"
                "MEMORY SAVINGS:\n"
                "  Reduces quantization overhead by ~0.37 bits/param\n"
                "  For 7B model: 7B × 0.37 bits ≈ 2.6 Gb ≈ ~0.32 GB saved\n"
                "  Not huge, but free quality-neutral savings.\n"
                "\n"
                "TRADEOFFS:\n"
                "  Marginal compute overhead for the second dequantization step.\n"
                "  Quality impact is negligible (quantizing already-quantized scales is safe).\n"
                "\n"
                "PRODUCTION: Always enabled when available. Pure upside."
            )
        }
    )

    # ── Adapter Saving ────────────────────────────────────────────────────────
    merge_adapters: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to merge LoRA weights into base model after training.\n"
                "WHEN TO MERGE:\n"
                "  True: Deploy a single model, use with vLLM/TGI for serving\n"
                "  False: Keep adapters separate for easy swapping between tasks\n"
                "MERGE MATH: W_final = W_frozen + (alpha/r) * B*A\n"
                "TRADEOFF: Merged = faster inference (no adapter overhead)\n"
                "          Separate = swap tasks without reloading base model"
            )
        }
    )


# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================

@dataclass
class DataConfig:
    """
    Dataset loading, preprocessing, and tokenization configuration.

    DATASET PIPELINE OVERVIEW:
    ==========================
    Raw HF Dataset → Format prompt → Tokenize → Pad/Pack → DataLoader → Training

    The pipeline design matters enormously for both training quality and speed.
    Poor data pipelines are the #1 source of training bugs in real projects.
    """
    dataset_name: str = field(
        default="timdettmers/openassistant-guanaco",
        metadata={
            "help": (
                "HuggingFace dataset ID. Good options for instruction following:\n"
                "  'timdettmers/openassistant-guanaco' — OpenAssistant conversations\n"
                "  'HuggingFaceH4/ultrachat_200k' — UltraChat 200K (high quality)\n"
                "  'teknium/OpenHermes-2.5' — OpenHermes, diverse instruction data\n"
                "  'Open-Orca/OpenOrca' — OpenOrca, reasoning-focused\n"
                "  'smangrul/code-chat-assistant-v1' — Code assistant data\n"
                "Can also be a local path: '/home/user/my_dataset'"
            )
        }
    )

    dataset_split: str = field(
        default="train",
        metadata={"help": "Which split to use for training (typically 'train')."}
    )

    val_split_size: float = field(
        default=0.1,
        metadata={
            "help": (
                "Fraction of data to use for validation (if no dedicated val split).\n"
                "0.1 = 10% validation, 90% training.\n"
                "For large datasets (>100K), even 0.01 (1%) gives reliable validation metrics."
            )
        }
    )

    text_column: str = field(
        default="text",
        metadata={"help": "Column name containing the text in the dataset."}
    )

    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum sequence length for training examples.\n"
                "\n"
                "MEMORY IMPACT IS QUADRATIC (before FlashAttn) or LINEAR (with FlashAttn):\n"
                "  Without Flash Attention: memory ∝ N² (attention matrix)\n"
                "  With Flash Attention 2: memory ∝ N (tiled computation)\n"
                "\n"
                "PRACTICAL LIMITS (with QLoRA + FlashAttn on RTX 4050 6GB):\n"
                "  max_seq_length=512:  Fast training, misses long-range patterns\n"
                "  max_seq_length=1024: Good balance for most chat datasets\n"
                "  max_seq_length=2048: Standard for instruction fine-tuning\n"
                "  max_seq_length=4096: Needs FlashAttn2 + gradient checkpointing\n"
                "\n"
                "TRUNCATION: Sequences longer than max_seq_length are truncated.\n"
                "Strategy matters: truncate from the LEFT (lose early context) or\n"
                "RIGHT (lose response end) — see truncation_side."
            )
        }
    )

    # ── Sequence Packing ──────────────────────────────────────────────────────
    use_packing: bool = field(
        default=True,
        metadata={
            "help": (
                "SEQUENCE PACKING:\n"
                "\n"
                "WHAT IT IS:\n"
                "  Instead of padding short sequences to max_seq_length (wasteful),\n"
                "  pack multiple short sequences into a single training example up to\n"
                "  max_seq_length, separated by EOS tokens.\n"
                "\n"
                "  Example (max_len=1024):\n"
                "  Without packing: [seq1(200)] [PAD×824] → 80% padding tokens!\n"
                "  With packing:    [seq1(200)][EOS][seq2(300)][EOS][seq3(500)][EOS]\n"
                "                   → 0% padding, full utilization\n"
                "\n"
                "WHY IT MATTERS:\n"
                "  Padding tokens contribute ZERO gradient signal but still cost:\n"
                "  - Memory: pad tokens occupy VRAM in KV cache + activations\n"
                "  - Compute: attention computed over pad tokens (even if masked)\n"
                "  Packing converts wasted compute into real signal.\n"
                "\n"
                "THROUGHPUT IMPACT:\n"
                "  For datasets with short avg sequence length (e.g., 200 tokens):\n"
                "  Packing can give 3-5× higher throughput on same hardware!\n"
                "\n"
                "CAVEAT (Cross-contamination):\n"
                "  Without careful attention masking, the model might attend across\n"
                "  packed sequence boundaries (seq2 attending to seq1 context).\n"
                "  TRL's SFTTrainer handles this correctly with position_ids reset.\n"
                "\n"
                "WHEN TO DISABLE:\n"
                "  - When sequences are already near max_seq_length (no benefit)\n"
                "  - When debugging (makes error tracing harder)\n"
                "  - For tasks sensitive to sequence boundary contamination\n"
                "\n"
                "PRODUCTION: Standard in modern LLM training (used in LLaMA, Mistral training)."
            )
        }
    )

    # ── Streaming ─────────────────────────────────────────────────────────────
    streaming: bool = field(
        default=False,
        metadata={
            "help": (
                "STREAMING DATASETS:\n"
                "\n"
                "WHY IT EXISTS:\n"
                "  Large datasets (e.g., The Pile, 800 GB) can't fit in RAM.\n"
                "  Streaming loads examples one-by-one from disk/network on demand.\n"
                "\n"
                "HOW IT WORKS:\n"
                "  HF Datasets IterableDataset: reads shards sequentially, yields examples\n"
                "  lazily. Compatible with PyTorch DataLoader via custom collator.\n"
                "\n"
                "TRADEOFFS:\n"
                "  Pros: No memory limit on dataset size, start training immediately\n"
                "  Cons: No random shuffling (only buffer shuffling), can't know dataset\n"
                "        size in advance (affects LR scheduler), slower with network datasets\n"
                "\n"
                "RECOMMENDATION: Use streaming=True for datasets >10 GB or >1M examples.\n"
                "For smaller datasets, loading to memory gives better shuffling + speed."
            )
        }
    )

    truncation_side: str = field(
        default="right",
        metadata={
            "help": (
                "Truncation strategy for sequences exceeding max_seq_length.\n"
                "'right': truncate the end (lose response tail) — OK for instruction tuning\n"
                "'left': truncate the beginning (lose early context) — better for completion"
            )
        }
    )

    num_workers: int = field(
        default=4,
        metadata={"help": "Number of DataLoader worker processes for parallel data loading."}
    )

    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Workers for dataset map() tokenization preprocessing."}
    )


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Core training hyperparameters and optimization settings.
    """
    output_dir: str = field(
        default="./outputs",
        metadata={"help": "Directory to save checkpoints, adapter weights, and logs."}
    )

    # ── Core Hyperparameters ──────────────────────────────────────────────────
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of complete passes through the training dataset."}
    )

    per_device_train_batch_size: int = field(
        default=4,
        metadata={
            "help": (
                "Batch size per GPU. With QLoRA on 7B model:\n"
                "  RTX 4050 6GB: batch_size=1-2\n"
                "  RTX 3090 24GB: batch_size=4-8\n"
                "  A100 80GB: batch_size=16-32\n"
                "COMBINED with gradient_accumulation_steps for effective batch size."
            )
        }
    )

    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU during evaluation."}
    )

    # ── Gradient Accumulation ─────────────────────────────────────────────────
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help": (
                "GRADIENT ACCUMULATION:\n"
                "\n"
                "WHAT IT IS:\n"
                "  Instead of updating weights after every batch, accumulate gradients\n"
                "  over N batches, then perform one large optimizer step.\n"
                "\n"
                "WHY IT EXISTS:\n"
                "  Effective batch size = per_device_batch_size × num_GPUs × gradient_accumulation_steps\n"
                "  Small GPUs can't fit large batches. Gradient accumulation simulates\n"
                "  large batches on small GPU memory.\n"
                "\n"
                "MATHEMATICAL INTUITION:\n"
                "  Normal (batch=32): compute grad(L(x₁,...,x₃₂)), update weights\n"
                "  Grad accum (batch=4, accum=8): compute grad(L(x₁,...,x₄)),\n"
                "    accumulate, repeat 7 more times, sum gradients, update weights\n"
                "  The gradient estimates converge: E[Σᵢ grad(Lᵢ)] ≈ grad(L(all))\n"
                "\n"
                "MEMORY IMPACT:\n"
                "  None! Gradient accumulation doesn't change peak memory usage.\n"
                "  It's purely a step-counting trick.\n"
                "\n"
                "SPEED IMPACT:\n"
                "  Slightly slower per-step (same forward+backward, but N-1 steps\n"
                "  don't do optimizer update → lower GPU utilization between updates)\n"
                "  Often 5-10% overhead vs equivalent large batch.\n"
                "\n"
                "TRADEOFFS:\n"
                "  Larger effective batch = more stable gradients, smoother loss\n"
                "  Too large = loss of stochasticity benefits (worse generalization)\n"
                "  Sweet spot: effective batch of 32-128 for most LLM fine-tuning\n"
                "\n"
                "PRODUCTION: Universal technique. Every LLM training code uses this."
            )
        }
    )

    # ── Gradient Checkpointing ────────────────────────────────────────────────
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": (
                "GRADIENT CHECKPOINTING (Activation Checkpointing):\n"
                "\n"
                "THE PROBLEM IT SOLVES:\n"
                "  During backpropagation, PyTorch stores ALL intermediate activations\n"
                "  from the forward pass (needed to compute gradients via chain rule).\n"
                "  For a 7B model with 32 layers, these activations can use 10-20 GB!\n"
                "\n"
                "WHAT IT DOES:\n"
                "  Instead of storing all activations, store only checkpoints at regular\n"
                "  intervals (e.g., every layer boundary). During backward pass, recompute\n"
                "  intermediate activations from the nearest checkpoint.\n"
                "\n"
                "MEMORY-COMPUTE TRADEOFF:\n"
                "  Memory: O(√N) instead of O(N) for N layers (optimal checkpoint spacing)\n"
                "  Compute: ~33% more FLOPs (each activation computed ~1.33× times)\n"
                "  Speed: ~20-30% slower training\n"
                "\n"
                "MATHEMATICAL INTUITION:\n"
                "  Normal backprop: store all {a₁, a₂, ..., aₙ} → O(N) memory\n"
                "  Checkpointed: store {a₁, a_√N, a_2√N, ...} → O(√N) checkpoints\n"
                "  When backpropagating through segment k, recompute forward from aₖ√N\n"
                "  Total recomputation cost: O(√N) × O(√N segment) = O(N) extra compute\n"
                "  But memory drops from O(N) to O(√N)!\n"
                "\n"
                "IMPACT FOR 7B MODEL:\n"
                "  Without: ~18 GB for activations alone\n"
                "  With: ~4-6 GB for activations, but 25% slower\n"
                "\n"
                "PRODUCTION: Always enabled for models >3B. Non-negotiable for consumer GPU."
            )
        }
    )

    # ── Learning Rate Schedule ────────────────────────────────────────────────
    learning_rate: float = field(
        default=2e-4,
        metadata={
            "help": (
                "Peak learning rate for the optimizer.\n"
                "QLORA RECOMMENDATIONS:\n"
                "  r=16: 2e-4 (standard)\n"
                "  r=8: 3e-4 to 5e-4\n"
                "  r=64: 1e-4 (lower rank → lower LR)\n"
                "NOTE: QLoRA uses higher LR than full fine-tuning (1e-4 to 3e-4)\n"
                "because only LoRA params (small fraction) are updated."
            )
        }
    )

    lr_scheduler_type: str = field(
        default="cosine",
        metadata={
            "help": (
                "Learning rate schedule type.\n"
                "'cosine': LR decays from peak to 0 following cos(πt/T)/2 + 0.5\n"
                "  Best default. Smooth decay, often better final performance.\n"
                "'linear': LR decays linearly from peak to 0\n"
                "  Simple, predictable, good for shorter training runs\n"
                "'constant': Fixed LR (no decay) — rarely optimal\n"
                "'cosine_with_restarts': Cyclic cosine (experimental)\n"
                "\n"
                "INTUITION: High LR early = fast movement toward good region.\n"
                "Low LR late = fine-grained refinement without overshooting."
            )
        }
    )

    warmup_ratio: float = field(
        default=0.03,
        metadata={
            "help": (
                "Fraction of training steps to use for linear LR warmup.\n"
                "0.03 = 3% warmup (e.g., 30 warmup steps for 1000 total steps)\n"
                "\n"
                "WHY WARMUP?\n"
                "  At training start, gradients are large and noisy (model randomly\n"
                "  initialized or at pretrained weights with new task context).\n"
                "  High LR at step 1 → large weight updates → potential destabilization.\n"
                "  Warmup ramps LR from 0 → peak over warmup_steps, stabilizing training.\n"
                "\n"
                "WITHOUT WARMUP: Training sometimes collapses in first 100 steps.\n"
                "WITH WARMUP: Smooth convergence, rarely see early collapse."
            )
        }
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optim: str = field(
        default="paged_adamw_8bit",
        metadata={
            "help": (
                "PAGED ADAMW 8-BIT (bitsandbytes optimizer):\n"
                "\n"
                "STANDARD ADAMW:\n"
                "  Adam with weight decay. Maintains two moment estimates per parameter:\n"
                "    mₜ = β₁mₜ₋₁ + (1-β₁)gₜ   [first moment = gradient moving average]\n"
                "    vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²  [second moment = squared gradient avg]\n"
                "  Weight update: θ ← θ - lr × (mₜ/√vₜ + ε) - wd×θ\n"
                "  Memory cost: 2 × model_size in FP32 for m and v vectors\n"
                "  For 7B params: 7B × 2 × 4 bytes = 56 GB just for optimizer states!\n"
                "\n"
                "8-BIT ADAM (bitsandbytes):\n"
                "  Store optimizer states in 8-bit instead of 32-bit.\n"
                "  Uses block-wise quantization for stability.\n"
                "  Memory: 56 GB → 14 GB (4× savings)\n"
                "  Quality: Nearly identical to FP32 Adam (block quantization preserves stats)\n"
                "\n"
                "BUT WAIT — we're training LoRA (only ~84M params), so optimizer states\n"
                "for LoRA are tiny! The 8-bit savings mainly matters for full fine-tuning.\n"
                "For QLoRA, use 'paged_adamw_32bit' or 'adamw_torch' for better precision.\n"
                "\n"
                "PAGED OPTIMIZER (QLoRA innovation):\n"
                "  'Paged' means CPU RAM is used as overflow for optimizer states.\n"
                "  When GPU memory fills up, optimizer states are paged to CPU RAM\n"
                "  (like virtual memory in OS). This prevents OOM crashes on long sequences.\n"
                "  Overhead: ~5-10% when paging occurs, 0% when not needed.\n"
                "\n"
                "OPTIONS:\n"
                "  'paged_adamw_8bit': Memory efficient, good for large models\n"
                "  'paged_adamw_32bit': More precise, recommended for QLoRA\n"
                "  'adamw_torch': PyTorch native, fastest, use when memory allows\n"
                "  'adamw_torch_fused': Fused kernel version, ~10% faster"
            )
        }
    )

    weight_decay: float = field(
        default=0.001,
        metadata={
            "help": (
                "L2 regularization coefficient (AdamW weight decay).\n"
                "Small values (0.001-0.01) work well for fine-tuning.\n"
                "Larger values (0.1) used in pretraining from scratch.\n"
                "Prevents overfitting by penalizing large weights."
            )
        }
    )

    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": (
                "GRADIENT CLIPPING:\n"
                "\n"
                "WHAT: If ||gradients||₂ > max_grad_norm, scale gradients to have\n"
                "      exactly max_grad_norm magnitude. This prevents gradient explosions.\n"
                "\n"
                "WHY: In deep networks, gradients can occasionally spike to huge values,\n"
                "     causing catastrophic weight updates that destroy training progress.\n"
                "\n"
                "MATH: g_clipped = g × min(1, max_grad_norm / ||g||₂)\n"
                "\n"
                "VALUE CHOICE:\n"
                "  0.3: Conservative (QLoRA standard), prevents destabilization\n"
                "  1.0: Standard for pretraining (less restrictive)\n"
                "  Very small (<0.1): May slow down learning unnecessarily"
            )
        }
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every N training steps."}
    )

    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every N training steps."}
    )

    logging_steps: int = field(
        default=10,
        metadata={"help": "Log training metrics every N steps."}
    )

    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep. Older ones are deleted."}
    )

    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load the checkpoint with best validation loss at end of training."}
    )

    # ── Mixed Precision ───────────────────────────────────────────────────────
    bf16: bool = field(
        default=True,
        metadata={"help": "Enable BF16 mixed precision if supported (Ampere+ GPUs)."}
    )

    fp16: bool = field(
        default=False,
        metadata={"help": "Enable FP16 mixed precision (fallback for pre-Ampere GPUs)."}
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility (Python, NumPy, PyTorch, CUDA)."}
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    report_to: str = field(
        default="wandb",
        metadata={"help": "Experiment tracking: 'wandb', 'tensorboard', 'none'."}
    )

    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name. Auto-generated if None."}
    )

    # ── Multi-GPU ─────────────────────────────────────────────────────────────
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "DataLoader worker processes."}
    )

    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Keep all dataset columns (False = don't auto-remove)."}
    )

    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": (
                "DDP option to find unused parameters during backward pass.\n"
                "False: Faster DDP, assumes all parameters are used.\n"
                "True: Needed if some modules aren't used in every forward pass."
            )
        }
    )

    # ── Group by Length (Dynamic Padding) ────────────────────────────────────
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": (
                "DYNAMIC PADDING (Group by Length):\n"
                "\n"
                "WHAT IT IS:\n"
                "  Group similar-length sequences together in the same batch.\n"
                "  Pad only to the MAXIMUM LENGTH WITHIN EACH BATCH, not globally.\n"
                "\n"
                "EXAMPLE:\n"
                "  Dataset lengths: [50, 52, 48, 900, 850, 920, 100, 95, 110]\n"
                "  Without grouping:\n"
                "    Batch 1: [50, 900, 95] → pad all to 900 → 77% padding waste\n"
                "  With grouping (by length):\n"
                "    Batch 1: [48, 50, 52] → pad to 52 → 3% padding waste\n"
                "    Batch 2: [850, 900, 920] → pad to 920 → 3% waste\n"
                "\n"
                "THROUGHPUT IMPACT:\n"
                "  Can 2-3× throughput improvement on datasets with varied lengths.\n"
                "  Less impactful when using sequence packing (packing already handles this).\n"
                "\n"
                "CAVEAT:\n"
                "  Batches are no longer uniformly random → slight bias in gradient estimates.\n"
                "  Use with shuffle=True to mitigate this.\n"
                "\n"
                "PRODUCTION: Standard practice. HF Trainer supports this natively."
            )
        }
    )


# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================

@dataclass
class InferenceConfig:
    """
    Configuration for inference with vLLM and HuggingFace.

    VLLM ARCHITECTURE:
    ==================
    vLLM is a high-throughput inference server for LLMs, developed at UC Berkeley.
    Key innovations vs naive HF generate():

    1. PAGEDATTENTION:
       Standard KV cache allocates contiguous memory blocks per sequence.
       Problem: Sequences have unknown length at request start → over-allocate (50% wasted).
       PagedAttention uses non-contiguous "pages" like OS virtual memory.
       Each page holds K,V vectors for a fixed number of tokens.
       Pages are allocated on-demand and can be SHARED between parallel beam search paths.
       Result: 90%+ KV cache utilization vs ~50-70% for standard caching.

    2. CONTINUOUS BATCHING:
       Naive: process requests in fixed batches, wait for ALL to finish.
       Problem: Short requests waste cycles waiting for long ones.
       Continuous batching: dynamically add new requests to the batch when slots free up.
       Result: 2-5× higher throughput vs naive batching for mixed request lengths.

    3. FUSED CUDA KERNELS:
       vLLM implements custom CUDA kernels for attention with PagedAttention blocks.
       Combined with FP16/BF16 inference: very fast compute.
    """
    model_path: str = field(
        default="./outputs/final_model",
        metadata={"help": "Path to the merged fine-tuned model for vLLM inference."}
    )

    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens to generate."}
    )

    temperature: float = field(
        default=0.7,
        metadata={
            "help": (
                "TEMPERATURE SAMPLING:\n"
                "\n"
                "WHAT IT IS:\n"
                "  Scales logits before softmax, controlling output randomness.\n"
                "  new_logit = logit / temperature\n"
                "\n"
                "MATH:\n"
                "  P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)\n"
                "  T→0: deterministic (always pick highest probability token)\n"
                "  T=1: unmodified model distribution\n"
                "  T→∞: uniform random across all tokens\n"
                "\n"
                "INTUITION:\n"
                "  Low T (0.1-0.4): precise, factual, repetitive — good for code/math\n"
                "  Medium T (0.6-0.8): balanced creativity and coherence — good for chat\n"
                "  High T (0.9-1.2): creative, diverse, sometimes incoherent — for creative writing"
            )
        }
    )

    top_p: float = field(
        default=0.9,
        metadata={
            "help": (
                "NUCLEUS SAMPLING (Top-P):\n"
                "\n"
                "Sample only from the smallest set of tokens whose\n"
                "cumulative probability exceeds p.\n"
                "At each step: sort tokens by prob, take top-k until sum > p.\n"
                "\n"
                "EXAMPLE: p=0.9, token probs=[0.5, 0.3, 0.1, 0.05, 0.04, 0.01, ...]\n"
                "  Cumulative: [0.5, 0.8, 0.9] → stop at 3rd token (0.9 reached)\n"
                "  Sample from these 3 tokens only, rescaled to sum to 1.\n"
                "\n"
                "WHY BETTER THAN TOP-K:\n"
                "  Top-K always picks same number of candidates regardless of distribution.\n"
                "  Top-P adapts: when model is confident (peaked dist), few candidates;\n"
                "  when uncertain (flat dist), many candidates. More natural."
            )
        }
    )

    top_k: int = field(
        default=50,
        metadata={
            "help": (
                "TOP-K SAMPLING:\n"
                "Sample only from the top-k most probable tokens at each step.\n"
                "Prevents sampling very low-probability (likely nonsense) tokens.\n"
                "k=50: standard safe value. k=1: greedy (deterministic)."
            )
        }
    )

    repetition_penalty: float = field(
        default=1.1,
        metadata={
            "help": (
                "REPETITION PENALTY:\n"
                "Penalizes tokens that have already appeared in the context.\n"
                "logit_penalized = logit / penalty  (if token already appeared)\n"
                "1.0: no penalty, 1.1: mild, 1.3: strong, >1.5: too aggressive\n"
                "Prevents loops like 'the the the the...'"
            )
        }
    )

    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs for tensor parallelism in vLLM. 1 for single GPU."}
    )

    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for vLLM: 'bfloat16', 'float16', 'float32'."}
    )

    use_vllm: bool = field(
        default=True,
        metadata={"help": "Use vLLM for inference. False = use HuggingFace generate()."}
    )


# ==============================================================================
# WANDB CONFIGURATION
# ==============================================================================

@dataclass
class WandbConfig:
    """
    Weights & Biases experiment tracking configuration.

    WHY WANDB?
    ----------
    Training LLMs without proper experiment tracking is flying blind.
    W&B provides:
      1. Real-time loss curves: spot divergence within minutes
      2. Hyperparameter comparison: which lr/rank/batch combination worked best?
      3. GPU/memory monitoring: catch OOM risks early
      4. Artifact versioning: log model checkpoints and datasets
      5. Team sharing: everyone can see experiment results
    """
    project: str = field(
        default="llm-finetune",
        metadata={"help": "WandB project name."}
    )

    entity: Optional[str] = field(
        default=None,
        metadata={"help": "WandB entity (username or team). None = default entity."}
    )

    tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Tags for this run (e.g., ['qlora', 'llama3', '7b'])."}
    )

    log_model: bool = field(
        default=False,
        metadata={"help": "Log model checkpoints as W&B artifacts (large storage!)."}
    )

    log_freq: int = field(
        default=10,
        metadata={"help": "Log custom metrics every N steps."}
    )


# ==============================================================================
# UNIFIED CONFIG (combines all sub-configs)
# ==============================================================================

@dataclass
class FinetuneConfig:
    """
    Master configuration that combines all sub-configurations.
    Used when loading from YAML or passing programmatically.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        """Validate config interdependencies."""
        # BF16/FP16 mutual exclusion
        if self.training.bf16 and self.training.fp16:
            raise ValueError("Cannot use both bf16 and fp16. Choose one.")

        # Packing + group_by_length warning
        if self.data.use_packing and self.training.group_by_length:
            import warnings
            warnings.warn(
                "use_packing=True and group_by_length=True both set. "
                "Packing already handles length-based batching; group_by_length is redundant. "
                "Consider disabling group_by_length."
            )

        # Output dir creation
        os.makedirs(self.training.output_dir, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        """Load config from YAML file."""
        import yaml
        from dacite import from_dict
        with open(path) as f:
            data = yaml.safe_load(f)
        return from_dict(data_class=cls, data=data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
