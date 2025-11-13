# finetune_llama2.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load GPT-2 base model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load your Shoolini University dataset
train_path = "data/shoolini.txt"  # ensure this file exists
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)

# Create a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./finetuned_gpt2_shoolini",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=1,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save the final model
trainer.save_model("./finetuned_gpt2_shoolini")
print("✅ Fine-tuning completed and model saved!")
