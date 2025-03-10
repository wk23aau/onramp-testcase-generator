import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    # Load processed data
    data_dir = os.path.join("data", "processed")
    with open(os.path.join(data_dir, "merged_train.json"), "r") as f:
        train_data = json.load(f)

    # Load domain-adapted BERT (replace with your model)
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Or use your domain model
    model.eval()

    # Generate embeddings for test cases
    tc_texts = [json.dumps(tc["TestSteps"]) for tc in train_data]  # Use test case steps as context
    tc_embeddings = model.encode(tc_texts, show_progress_bar=True)

    # Build FAISS index
    dimension = tc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(tc_embeddings)

    # Save index and metadata
    faiss.write_index(index, os.path.join(data_dir, "retrieval_index.faiss"))
    with open(os.path.join(data_dir, "retrieval_metadata.json"), "w") as f:
        json.dump(train_data, f)

    print(f"FAISS index and metadata saved to {data_dir}")

if __name__ == "__main__":
    main()