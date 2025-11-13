from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load dataset
dataset = load_dataset("text", data_files={"train": "my_data.txt"})

# Step 2: Load tokenizer and model (multilingual BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

# Step 3: Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Data collator (for Masked Language Modeling)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir="./pretrained_model",
    overwrite_output_dir=True,
    num_train_epochs=1,   # increase later if you have more data
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=1,
    logging_steps=10
)

# Step 6: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# Step 7: Start pretraining
trainer.train()

# Step 8: Save your fine-tuned/pretrained model
trainer.save_model("./my_pretrained_model")
tokenizer.save_pretrained("./my_pretrained_model")
