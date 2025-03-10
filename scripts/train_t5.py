import os
import json
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,  # ADD THIS LINE
    Trainer,
    TrainingArguments
)
from datasets import Dataset

def main():
    # Load tokenized dataset
    data_dir = os.path.join("data", "processed")
    with open(os.path.join(data_dir, "t5_train_dataset.json"), "r") as f:
        t5_data = json.load(f)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(t5_data)

    # Initialize T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join("models", "t5_finetuned"),
        per_device_train_batch_size=4,
        num_train_epochs=5,
        evaluation_strategy="no",
        logging_steps=100,
        save_steps=500,
        fp16=True,  # Enable mixed precision
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(os.path.join("models", "t5_finetuned"))
    tokenizer.save_pretrained(os.path.join("models", "t5_finetuned"))

if __name__ == "__main__":
    main()