import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def train_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    dataset_path = "./data/processed/train_data.jsonl"
    output_dir = "./models/lora_adapter"

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=10, max_steps=100, optim="paged_adamw_8bit",
        save_strategy="no"
    )

    trainer = SFTTrainer(
        model=model, train_dataset=dataset, peft_config=peft_config,
        dataset_text_field="text", max_seq_length=256, tokenizer=tokenizer, args=training_args,
    )

    print("Starting Training...")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training Complete!")

if __name__ == "__main__":
    train_model()