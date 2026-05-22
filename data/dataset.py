"""
data/dataset.py
===============
Complete data pipeline: loading → formatting → tokenizing → packing → DataLoader.

This module handles the entire data journey from raw HuggingFace dataset to
GPU-ready batches. Every design choice here has real impact on training quality
and speed.

PIPELINE OVERVIEW:
  HF Dataset
    → apply_chat_template() or format_prompt()
    → tokenizer(text, truncation=True, max_length=N)
    → ConstantLengthDataset (packing) OR DataCollatorForSeq2Seq (dynamic padding)
    → DataLoader
    → GPU batch
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict, IterableDataset
from transformers import PreTrainedTokenizer

from configs.config import DataConfig, TrainingConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT FORMATTING
# ==============================================================================

# These templates define how raw text is structured for the model.
# The model learns to produce text in this format, so using the CORRECT template
# for your model is critical. Using the wrong template degrades performance.

PROMPT_TEMPLATES = {
    # ── Llama 3 Instruct Template ─────────────────────────────────────────────
    # Llama 3 uses special tokens: <|begin_of_text|>, <|eot_id|>, etc.
    # These are part of the tokenizer vocabulary and must match the model exactly.
    "llama3": (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "{instruction}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{response}<|eot_id|>"
    ),

    # ── Mistral / Alpaca Template ─────────────────────────────────────────────
    # Simpler format: [INST] ... [/INST] wrapper
    "mistral": (
        "[INST] {instruction} [/INST] {response}</s>"
    ),

    # ── ChatML Format ─────────────────────────────────────────────────────────
    # Used by Qwen, Yi, and many open models.
    # <|im_start|> = "imaginary monologue start" (historical naming)
    "chatml": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n{response}<|im_end|>\n"
    ),

    # ── Alpaca Format ─────────────────────────────────────────────────────────
    # Classic Stanford Alpaca format. Widely understood by many models.
    "alpaca": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{response}"
    ),

    # ── Gemma IT Format ───────────────────────────────────────────────────────
    "gemma": (
        "<bos><start_of_turn>user\n"
        "{instruction}<end_of_turn>\n"
        "<start_of_turn>model\n"
        "{response}<end_of_turn>\n"
    ),
}


def get_prompt_formatter(
    model_name: str,
    template: Optional[str] = None,
    system_prompt: str = "You are a helpful, respectful and honest assistant.",
) -> Callable[[Dict], str]:
    """
    Returns a function that formats a dataset example into a prompt string.

    WHY A FACTORY FUNCTION?
    -----------------------
    Different models need different formats. Rather than if-else in the training loop,
    we create a specialized formatter once and reuse it. This keeps the training loop
    clean and makes the formatting logic testable independently.

    CHAT TEMPLATES:
    ---------------
    Modern HuggingFace tokenizers include a built-in chat_template (a Jinja2 template).
    We prefer the tokenizer's own template when available because it's authoritative.
    The templates dict above are fallbacks.

    Args:
        model_name: Model name to auto-detect template (if template not specified)
        template: Override template name ('llama3', 'mistral', 'chatml', 'alpaca', 'gemma')
        system_prompt: Default system prompt text

    Returns:
        Callable that takes a dataset row dict and returns formatted string
    """
    # Auto-detect template from model name
    if template is None:
        name_lower = model_name.lower()
        if "llama-3" in name_lower or "llama3" in name_lower:
            template = "llama3"
        elif "mistral" in name_lower:
            template = "mistral"
        elif "qwen" in name_lower or "yi" in name_lower:
            template = "chatml"
        elif "gemma" in name_lower:
            template = "gemma"
        else:
            template = "alpaca"
        logger.info(f"Auto-detected template: {template}")

    prompt_template = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["alpaca"])

    def formatter(example: Dict) -> str:
        """Format a single dataset example into a training prompt."""
        # Handle different dataset schema patterns
        if "text" in example:
            # Already formatted (e.g., guanaco dataset has pre-formatted text)
            return example["text"]

        # Extract instruction/input/output from common schema patterns
        instruction = example.get("instruction", example.get("question", example.get("input", "")))
        response = example.get("output", example.get("response", example.get("answer", "")))
        system = example.get("system", system_prompt)

        # Handle multi-turn conversations (common in ShareGPT format)
        if "conversations" in example:
            return _format_conversations(example["conversations"], template, system)

        if "messages" in example:
            return _format_messages(example["messages"], template, system)

        # Single turn
        try:
            return prompt_template.format(
                instruction=instruction,
                response=response,
                system=system,
            )
        except KeyError:
            # Fallback: use text field or repr
            return example.get("text", str(example))

    return formatter


def _format_conversations(conversations: List[Dict], template: str, system: str) -> str:
    """
    Format ShareGPT-style conversations to prompt string.
    ShareGPT format: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]

    Multi-turn conversation handling is crucial for fine-tuning chat models.
    The model must learn to handle context from previous turns.
    """
    prompt_template = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["alpaca"])
    result = ""
    for i in range(0, len(conversations) - 1, 2):
        user_msg = conversations[i]["value"] if conversations[i]["from"] == "human" else ""
        if i + 1 < len(conversations):
            asst_msg = conversations[i + 1]["value"] if conversations[i + 1]["from"] == "gpt" else ""
        else:
            asst_msg = ""

        result += prompt_template.format(
            instruction=user_msg,
            response=asst_msg,
            system=system if i == 0 else "",
        )
    return result


def _format_messages(messages: List[Dict], template: str, system: str) -> str:
    """
    Format OpenAI-style messages to prompt string.
    Messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    prompt_template = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["alpaca"])
    result = ""
    i = 0
    sys_msg = system

    if messages and messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        i = 1

    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            result += prompt_template.format(
                instruction=messages[i]["content"],
                response=messages[i + 1]["content"],
                system=sys_msg if i <= 1 else "",
            )
            i += 2
        else:
            i += 1
    return result


# ==============================================================================
# TOKENIZATION
# ==============================================================================

def tokenize_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    text_column: str = "text",
    add_eos_token: bool = True,
) -> Dict[str, List]:
    """
    Tokenize a batch of examples.

    IMPORTANT DESIGN DECISIONS:
    ---------------------------
    1. TRUNCATION: We truncate to max_seq_length. The choice of what to truncate
       (beginning vs end) affects quality. For instruction following, right-side
       truncation (losing response tail) is usually preferable to left-side truncation
       (losing the instruction).

    2. EOS TOKEN: We MUST add EOS at sequence end to teach the model when to stop.
       Without EOS, the model generates indefinitely during inference.

    3. LABELS = INPUT_IDS: For causal language modeling, the target is the input
       shifted by one position. The Trainer handles this shift internally.
       labels[i] = input_ids[i] means: "predict token at position i given tokens 0..i-1"

    4. ATTENTION MASK: 1 for real tokens, 0 for padding. Prevents attending to pad tokens.

    WHY BATCH TOKENIZATION?
    -----------------------
    Tokenizing examples one-by-one is 5-10× slower than batched tokenization.
    dataset.map(tokenize_fn, batched=True) processes 1000 examples per call,
    taking advantage of parallelism in the Rust tokenizer backend.
    """
    # Apply formatter to get text strings
    texts = examples[text_column]

    # Add EOS token if not present (teaches model to stop generating)
    if add_eos_token:
        texts = [
            t + tokenizer.eos_token if not t.endswith(tokenizer.eos_token) else t
            for t in texts
        ]

    # Tokenize batch
    # padding=False: we'll pad during collation (dynamic padding is more efficient)
    # truncation=True: cut sequences to max_seq_length
    tokenized = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        padding=False,  # Dynamic padding in collator
        return_tensors=None,  # Return Python lists, not tensors (more memory efficient)
    )

    # Labels = input_ids for causal LM
    # MASKING PADDING IN LABELS:
    # We set label=-100 for padding tokens. CrossEntropyLoss ignores -100 labels.
    # This ensures padding doesn't contribute to gradient computation.
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def load_and_prepare_dataset(
    data_config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    prompt_template: Optional[str] = None,
) -> DatasetDict:
    """
    Complete dataset preparation pipeline.

    Steps:
    1. Load raw dataset from HuggingFace Hub or local disk
    2. Apply prompt formatting (converts raw data to model-input strings)
    3. Tokenize (text → token IDs)
    4. Create train/validation split
    5. Return DatasetDict ready for training

    Args:
        data_config: DataConfig with dataset parameters
        tokenizer: Initialized tokenizer for the model
        model_name: Model name for template auto-detection
        prompt_template: Override prompt template name

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    logger.info(f"Loading dataset: {data_config.dataset_name}")

    # ── Step 1: Load Dataset ──────────────────────────────────────────────────
    load_kwargs = {
        "path": data_config.dataset_name,
        "split": data_config.dataset_split,
        "cache_dir": "./.dataset_cache",
    }

    # STREAMING vs IN-MEMORY:
    # Streaming uses IterableDataset which loads lazily from disk/network.
    # In-memory caches everything in RAM for faster, random-access shuffling.
    if data_config.streaming:
        load_kwargs["streaming"] = True

    raw_dataset = load_dataset(**load_kwargs)

    logger.info(f"Raw dataset loaded: {raw_dataset}")

    # ── Step 2: Format Prompts ────────────────────────────────────────────────
    # Apply prompt formatter to convert dataset rows to training strings
    formatter = get_prompt_formatter(
        model_name=model_name,
        template=prompt_template,
    )

    # Check if dataset already has 'text' column (pre-formatted)
    if data_config.text_column not in raw_dataset.column_names:
        logger.info("Formatting prompts...")

        def format_and_rename(examples: Dict) -> Dict:
            """Apply formatter to batch and create 'text' column."""
            formatted = []
            for i in range(len(next(iter(examples.values())))):
                single = {k: v[i] for k, v in examples.items()}
                formatted.append(formatter(single))
            return {"text": formatted}

        if data_config.streaming:
            raw_dataset = raw_dataset.map(
                format_and_rename,
                batched=True,
                remove_columns=raw_dataset.column_names,
            )
        else:
            raw_dataset = raw_dataset.map(
                format_and_rename,
                batched=True,
                num_proc=data_config.preprocessing_num_workers,
                remove_columns=raw_dataset.column_names,
                desc="Formatting prompts",
            )

    # ── Step 3: Tokenize ─────────────────────────────────────────────────────
    if not data_config.use_packing:
        # Only tokenize here if NOT packing.
        # If packing, ConstantLengthDataset handles tokenization internally.
        logger.info("Tokenizing dataset...")

        tokenize_fn = partial(
            tokenize_function,
            tokenizer=tokenizer,
            max_seq_length=data_config.max_seq_length,
            text_column="text",
        )

        if data_config.streaming:
            tokenized_dataset = raw_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=["text"],
            )
        else:
            tokenized_dataset = raw_dataset.map(
                tokenize_fn,
                batched=True,
                num_proc=data_config.preprocessing_num_workers,
                remove_columns=["text"],
                desc="Tokenizing",
            )
    else:
        tokenized_dataset = raw_dataset  # Packing handles tokenization

    # ── Step 4: Train/Validation Split ───────────────────────────────────────
    if data_config.streaming:
        # Can't split streaming datasets deterministically
        # Use a workaround: take N examples for validation
        # This is an approximation and known limitation of streaming
        logger.warning(
            "Streaming datasets don't support clean train/val splits. "
            "Using a fixed number of examples as validation set."
        )
        train_dataset = tokenized_dataset.skip(1000)
        val_dataset = tokenized_dataset.take(1000)
        return DatasetDict({"train": train_dataset, "validation": val_dataset})

    split_dataset = tokenized_dataset.train_test_split(
        test_size=data_config.val_split_size,
        seed=42,
    )
    logger.info(
        f"Dataset split: {len(split_dataset['train'])} train, "
        f"{len(split_dataset['test'])} validation examples"
    )

    return DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"],
    })


# ==============================================================================
# SEQUENCE PACKING (ConstantLengthDataset)
# ==============================================================================

class PackedDataset(Dataset):
    """
    Packs multiple short sequences into fixed-length chunks of exactly max_seq_length.

    MOTIVATION:
    -----------
    Consider a chat dataset where average conversation = 200 tokens.
    With max_seq_length=2048:
      - Without packing: 200/2048 = 9.7% GPU utilization! 90% is padding.
      - With packing: pack ~10 conversations per example → near 100% utilization.

    HOW IT WORKS:
    -------------
    1. Iterate through all examples in order
    2. Concatenate their token sequences with EOS separators
    3. When accumulated tokens ≥ max_seq_length, yield a chunk
    4. Reset buffer and continue

    ATTENTION MASKING FOR PACKED SEQUENCES:
    ----------------------------------------
    CRITICAL: We must ensure attention doesn't cross sequence boundaries!
    If seq_A tokens attend to seq_B tokens, the model learns wrong patterns.
    
    SOLUTION: Use position_ids reset + causal mask within each sub-sequence.
    TRL's SFTTrainer handles this correctly via DataCollatorForCompletionOnlyLM.
    In this implementation, we use a simple approach: separate with EOS and
    rely on the model learning the EOS→new-sequence pattern.

    For perfect cross-contamination prevention, use TRL SFTTrainer with
    packing=True which implements proper attention mask segmentation.
    """

    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        text_column: str = "text",
        infinite: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_column = text_column
        self.infinite = infinite

        # Pre-pack all sequences
        self.packed_sequences = self._pack_sequences(dataset)
        logger.info(
            f"Packed {len(dataset)} examples into {len(self.packed_sequences)} "
            f"chunks of length {max_seq_length}"
        )

    def _pack_sequences(self, dataset) -> List[Dict[str, torch.Tensor]]:
        """Pack dataset examples into fixed-length chunks."""
        packed = []
        buffer_ids = []
        buffer_labels = []
        eos_id = self.tokenizer.eos_token_id

        for example in dataset:
            text = example[self.text_column]

            # Tokenize individual example (no max_length truncation here —
            # we'll handle chunking in the packing loop)
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,  # Still truncate single very long examples
                add_special_tokens=True,
            )["input_ids"]

            # Add EOS separator between sequences
            if buffer_ids and tokens[0] != eos_id:
                buffer_ids.append(eos_id)
                buffer_labels.append(-100)  # Don't compute loss on separator EOS
                # Actually we DO want loss on EOS (teaches model to stop)
                # Let's correct: use eos_id for labels too
                buffer_labels[-1] = eos_id

            buffer_ids.extend(tokens)
            buffer_labels.extend(tokens)  # Labels = tokens for causal LM

            # Emit chunks of exactly max_seq_length
            while len(buffer_ids) >= self.max_seq_length:
                chunk_ids = buffer_ids[:self.max_seq_length]
                chunk_labels = buffer_labels[:self.max_seq_length]

                packed.append({
                    "input_ids": torch.tensor(chunk_ids, dtype=torch.long),
                    "labels": torch.tensor(chunk_labels, dtype=torch.long),
                    "attention_mask": torch.ones(self.max_seq_length, dtype=torch.long),
                })

                buffer_ids = buffer_ids[self.max_seq_length:]
                buffer_labels = buffer_labels[self.max_seq_length:]

        # Handle remaining buffer (pad to max_seq_length or discard)
        if len(buffer_ids) > 0:
            pad_length = self.max_seq_length - len(buffer_ids)
            if pad_length > 0:
                buffer_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                buffer_labels.extend([-100] * pad_length)  # -100 = ignore in loss

            packed.append({
                "input_ids": torch.tensor(buffer_ids[:self.max_seq_length], dtype=torch.long),
                "labels": torch.tensor(buffer_labels[:self.max_seq_length], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [1] * (self.max_seq_length - pad_length) + [0] * pad_length,
                    dtype=torch.long,
                ),
            })

        return packed

    def __len__(self) -> int:
        return len(self.packed_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.packed_sequences[idx]


# ==============================================================================
# DATA COLLATOR
# ==============================================================================

class LLMDataCollator:
    """
    Custom data collator for LLM fine-tuning.

    WHY A CUSTOM COLLATOR?
    ----------------------
    The default HuggingFace DataCollatorWithPadding is generic and doesn't handle:
      1. The labels tensor (needed for loss computation)
      2. Masking padding in labels with -100
      3. Custom padding strategies

    DYNAMIC PADDING:
    ----------------
    Instead of padding all sequences to max_seq_length globally, we pad only
    to the MAXIMUM LENGTH in each BATCH. For a batch with lengths [50, 80, 60]:
      Global padding → pad everything to 2048 → 97% of batch is padding
      Dynamic padding → pad to 80 → 33% padding

    Combined with group_by_length (sorting by length before batching), this can
    reduce padding from 80%+ to <10%.

    IMPLEMENTATION:
    ---------------
    This collator is deliberately simple and educational.
    For production, consider using TRL's DataCollatorForCompletionOnlyLM
    which also handles masking the instruction part (only computing loss on
    assistant responses, not user messages).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        pad_to_multiple_of: Optional[int] = 8,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # Padding to multiple of 8 or 16 improves GPU throughput (tensor core alignment)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of examples into a batch tensor.

        PADDING LOGIC:
        - For input_ids and attention_mask: pad with pad_token_id and 0 respectively
        - For labels: pad with -100 (CrossEntropy ignores -100 indices)
        """
        # Find max length in this batch (dynamic padding)
        max_length = max(len(f["input_ids"]) for f in features)

        # Align to multiple of 8 for tensor core efficiency
        if self.pad_to_multiple_of:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []

        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_length = max_length - seq_len

            # Convert to lists for padding
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature.get("attention_mask", [1] * seq_len))
            labels = list(feature["labels"])

            # Pad on the right (standard for causal LM)
            # WHY RIGHT PADDING? Left padding changes position indices and confuses
            # RoPE (Rotary Position Embeddings) used in Llama/Mistral.
            # Always right-pad for training.
            input_ids += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
            labels += [-100] * pad_length  # -100 ignored in cross-entropy loss

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ==============================================================================
# DATALOADER CREATION
# ==============================================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    tokenizer: PreTrainedTokenizer,
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> tuple:
    """
    Create PyTorch DataLoaders for training and validation.

    DATALOADER DESIGN CHOICES:
    --------------------------
    1. shuffle=True for training: essential for gradient decorrelation
       Without shuffling, model sees same sequence of examples every epoch
       → memorizes order patterns, poor generalization

    2. pin_memory=True: pre-pins batch tensors to pinned CPU memory
       → faster CPU→GPU transfers via DMA (direct memory access)
       → typically 20-30% faster data loading on modern systems

    3. persistent_workers=True: keeps worker processes alive between epochs
       → avoids Python process spawn overhead (~1-2 sec/epoch)
       → recommended when num_workers > 0

    4. prefetch_factor=2: each worker pre-loads 2 batches ahead
       → hides data loading latency behind GPU compute

    Returns:
        (train_loader, val_loader) tuple
    """
    collator = LLMDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_config.max_seq_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.per_device_train_batch_size,
        shuffle=not data_config.streaming,  # Can't shuffle streaming datasets
        collate_fn=collator,
        num_workers=data_config.num_workers,
        pin_memory=True,
        persistent_workers=data_config.num_workers > 0,
        prefetch_factor=2 if data_config.num_workers > 0 else None,
        drop_last=True,  # Drop last incomplete batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.per_device_eval_batch_size,
        shuffle=False,  # Never shuffle validation
        collate_fn=collator,
        num_workers=data_config.num_workers,
        pin_memory=True,
        persistent_workers=data_config.num_workers > 0,
        prefetch_factor=2 if data_config.num_workers > 0 else None,
    )

    return train_loader, val_loader
