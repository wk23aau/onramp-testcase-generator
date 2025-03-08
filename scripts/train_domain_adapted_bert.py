# File: scripts/train_domain_adapted_bert.py

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def main():
    # 1. Config
    corpus_file = os.path.join("data", "processed", "domain_knowledge_corpus.txt")
    model_name = "bert-base-uncased"  # or "roberta-base", "distilbert-base-uncased", etc.
    output_dir = os.path.join("models", "domain_adapted_bert")

    # 2. Load dataset from text file
    # Each line in the text file is considered one "example".
    raw_dataset = load_dataset(
        "text",
        data_files={"train": corpus_file},
        split="train"
    )

    # 3. Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # 4. Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # 5. Tokenize dataset
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"]
    )

    # 6. Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no",   # no dev set, purely unsupervised
        do_train=True,
        do_eval=False,
    )

    # 8. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # purely unsupervised
        data_collator=data_collator
    )

    # 9. Train
    trainer.train()

    # 10. Save final model & tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Domain-adapted model saved to {output_dir}")

if __name__ == "__main__":
    main()
