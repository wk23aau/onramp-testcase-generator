import os
import json
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

def retrieve_similar_test_cases(user_story, top_k=3):
    # Load FAISS index and metadata
    data_dir = os.path.join("data", "processed")
    index = faiss.read_index(os.path.join(data_dir, "retrieval_index.faiss"))
    with open(os.path.join(data_dir, "retrieval_metadata.json"), "r") as f:
        metadata = json.load(f)

    # Encode user story
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Or use your domain model
    query_embedding = model.encode([user_story])
    
    # Retrieve top-k similar test cases
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [metadata[i] for i in indices[0]]
    return retrieved

def build_t5_prompt(user_story, retrieved_tcs):
    prompt = f"Generate test case for: {user_story['description']}\n\n"
    prompt += "Acceptance Criteria:\n" + "\n".join(user_story['acceptance_criteria'])
    prompt += "\n\nSimilar Test Cases:\n" + "\n".join([str(tc["TestSteps"]) for tc in retrieved_tcs])
    return prompt

def main():
    # Load split datasets
    data_dir = os.path.join("data", "processed")
    with open(os.path.join(data_dir, "merged_train.json"), "r") as f:
        train_data = json.load(f)

    # Initialize T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Prepare dataset for T5
    t5_data = []
    for example in tqdm(train_data):
        # Fix key names and handle acceptance criteria
        acceptance_criteria = example.get("AcceptanceCriteria", "")
        if isinstance(acceptance_criteria, str):
            acceptance_criteria = acceptance_criteria.split("\n")

        user_story = {
        "title": example["UserStoryTitle"],
        "description": example["UserStoryDescription"],
        "acceptance_criteria": example["AcceptanceCriteria"]
        }
        retrieved = retrieve_similar_test_cases(user_story)
        prompt = build_t5_prompt(user_story, retrieved)
        target = json.dumps(example["TestSteps"])  # Ensure this matches your data's key
        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512).input_ids
        labels = tokenizer(target, return_tensors="pt", padding="max_length", truncation=True, max_length=512).input_ids

        t5_data.append({
            "input_ids": input_ids.squeeze().tolist(),
            "labels": labels.squeeze().tolist()
        })

    # Save tokenized dataset
    with open(os.path.join(data_dir, "t5_train_dataset.json"), "w") as f:
        json.dump(t5_data, f)

if __name__ == "__main__":
    main()