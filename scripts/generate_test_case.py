from transformers import T5ForConditionalGeneration, T5Tokenizer
import faiss
from sentence_transformers import SentenceTransformer
import json

class TestCaseGenerator:
    def __init__(self):
        # Load models
        self.retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.t5_model = T5ForConditionalGeneration.from_pretrained("models/t5_finetuned")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("models/t5_finetuned")

        # Load FAISS index and metadata
        self.index = faiss.read_index("data/processed/retrieval_index.faiss")
        with open("data/processed/retrieval_metadata.json", "r") as f:
            self.metadata = json.load(f)

    def retrieve(self, user_story_text, top_k=3):
        query_embedding = self.retrieval_model.encode([user_story_text])
        _, indices = self.index.search(query_embedding, top_k)
        return [self.metadata[i] for i in indices[0]]

    def generate(self, user_story):
        # Retrieve similar test cases
        retrieved = self.retrieve(user_story["UserStoryDescription"])
        
        # Build prompt
        prompt = f"Generate test case for: {user_story['UserStoryDescription']}\n\n"
        prompt += "Acceptance Criteria:\n" + user_story["AcceptanceCriteria"].replace('\n', '\n- ')
        prompt += "\n\nSimilar Test Cases:\n" + "\n".join([tc["TestSteps"] for tc in retrieved])

        # Generate with T5
        inputs = self.t5_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.t5_model.generate(**inputs, max_length=512)
        return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)