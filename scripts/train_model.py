import os
import json
import re
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def clean_text(text):
    """
    Clean text by:
      - Removing extra whitespace.
      - Normalizing newlines.
      - Removing non-printable characters.
    """
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\\n\s*', '\\n', text)
    return text

def prepare_examples(data):
    examples = []
    for item in data:
        # Clean metadata fields
        title = clean_text(item.get("Title", ""))
        area_path = clean_text(item.get("Area Path", ""))
        state = clean_text(item.get("State", ""))
        
        # Build an enhanced prompt without the "Assigned To" field
        prompt = (
            f"Test Case Details:\n"
            f"Title: {title}\n"
            f"Area Path: {area_path}\n"
            f"State: {state}\n\n"
            "Based on the above, generate a detailed, step-by-step test case in the following format:\n"
            "Step 1: <Step Action> -> Expected: <Step Expected>\n"
            "Step 2: <Step Action> -> Expected: <Step Expected>\n"
            "...\n\n"
            "Test Steps:"
        )
        
        # Clean and concatenate test steps
        steps = []
        for step in item.get("Test Steps", []):
            step_num = str(step.get("Step", "")).strip()
            action = clean_text(step.get("Step Action", ""))
            expected = clean_text(step.get("Step Expected", ""))
            step_text = f"Step {step_num}: {action}"
            if expected:
                step_text += f" -> Expected: {expected}"
            steps.append(step_text)
        target = "\n".join(steps)
        target = clean_text(target)
        
        examples.append({"prompt": prompt, "target": target})
    return examples

def main():
    # Update this path to your merged JSON file for the specific system (e.g., EC)
    input_json = "data/processed/ONRAMP TCs/EC/merged.json"
    data = load_json_file(input_json)
    examples = prepare_examples(data)
    
    ds = Dataset.from_list(examples)
    ds = ds.train_test_split(test_size=0.1)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]
    
    model_name = "t5-base"
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Use a data collator for dynamic padding in sequence-to-sequence tasks.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    def tokenize_function(example):
        # Tokenize the prompt and target; no fixed padding here as the collator will handle it.
        model_inputs = tokenizer(example["prompt"], max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example["target"], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="models/t5_testcase_generator",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,  # using dynamic padding
    )
    
    trainer.train()
    
    model.save_pretrained("models/t5_testcase_generator")
    tokenizer.save_pretrained("models/t5_testcase_generator")
    print("Model training complete and saved.")

if __name__ == "__main__":
    main()
