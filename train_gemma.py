import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import json
import os
import sys
import gc

# --- CONFIGURATION FOR STABILITY ---
# Reduced from 1024 to 768 to save VRAM on RTX 4050 (6GB)
# If it freezes again, lower this to 512.
MAX_LENGTH = 768
MODEL_NAME = "google/gemma-2b-it"
OUTPUT_DIR = "./gemma-mtg-combo-finder"


# Explicitly clean memory callback
class EmptyCacheCallback(TrainerCallback):
    """Frees up memory after every few steps to prevent VRAM creep"""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()


def format_instruction(example):
    """Format with clear structure for reasoning"""
    return f"""<start_of_turn>user
{example['instruction']}

{example['input']}<end_of_turn>
<start_of_turn>model
{example['output']}<end_of_turn>"""


def prepare_dataset(tokenizer):
    """Load and tokenize training data"""
    print("Loading combo training data...")

    if not os.path.exists("data/combo_training_data.json"):
        print("ERROR: data/combo_training_data.json not found!")
        sys.exit(1)

    if not os.path.exists("data/training_data.json"):
        print("ERROR: data/training_data.json not found!")
        sys.exit(1)

    with open("data/combo_training_data.json", "r") as f:
        combo_data = json.load(f)

    with open("data/training_data.json", "r") as f:
        general_data = json.load(f)

    all_data = combo_data + general_data

    print(f"Loaded {len(all_data)} training examples")

    formatted_all = [{"text": format_instruction(ex)} for ex in all_data]

    # Save formatted data temporarily
    os.makedirs("data", exist_ok=True)
    with open("data/formatted_all.json", "w") as f:
        json.dump(formatted_all, f)

    all_dataset = load_dataset(
        "json", data_files="data/formatted_all.json", split="train"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized_dataset = all_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=all_dataset.column_names,
    )

    return tokenized_dataset.train_test_split(test_size=0.1)


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint to resume from"""
    if not os.path.exists(output_dir):
        return None

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest = os.path.join(output_dir, checkpoints[-1])
    print(f"Found checkpoint: {latest}")
    return latest


def main():
    print("=" * 60)
    print("GEMMA MTG PAUPER COMBO FINDER - STABLE TRAINING MODE")
    print("=" * 60)

    # Force cleanup before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Check for existing checkpoint
    resume_from = find_latest_checkpoint(OUTPUT_DIR)
    if resume_from:
        print(f"\n✓ Resuming from: {resume_from}")

    # Check CUDA
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem:.2f} GB")
        if mem < 7.0:
            print("⚠ LOW VRAM DETECTED: Running in high-efficiency mode")

    # Quantization config - Optimized for stability
    print("\nConfiguring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing manually on the model first
    model.gradient_checkpointing_enable()

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for training
    print("\nPreparing model for kbit training...")
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    print("\nPreparing dataset...")
    dataset = prepare_dataset(tokenizer)
    print(f"✓ Training samples: {len(dataset['train'])}")
    print(f"✓ Validation samples: {len(dataset['test'])}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training arguments - OPTIMIZED FOR STABILITY
    print("\nConfiguring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,  # Keep strict at 1
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        bf16=True,
        # Stability settings
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # CRITICAL FIX
        dataloader_num_workers=0,  # Windows fix
        # Logging & Saving
        save_steps=10,
        logging_steps=2,
        save_total_limit=3,  # Reduced to save disk space
        eval_strategy="steps",
        eval_steps=20,
        warmup_steps=100,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        report_to="none",
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_safetensors=True,
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=data_collator,
        callbacks=[EmptyCacheCallback()],  # Add memory cleaner
    )

    print("✓ Trainer initialized")
    print("\n" + "=" * 60)
    print("STARTING TRAINING (Safe Mode)")
    print("=" * 60)

    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
        print("Saving current state...")
    except Exception as e:
        print(f"\n⚠ ERROR during training: {e}")
        print("Attempting to save model...")

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
