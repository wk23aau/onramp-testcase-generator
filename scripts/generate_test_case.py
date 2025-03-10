import os
import json
import faiss
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

class TestCaseGenerator:
    def __init__(self):
        self.retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join("models", "t5_finetuned")
        )
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            os.path.join("models", "t5_finetuned")
        )

        data_dir = os.path.join("data", "processed")
        self.index = faiss.read_index(os.path.join(data_dir, "retrieval_index.faiss"))
        with open(os.path.join(data_dir, "retrieval_metadata.json"), "r") as f:
            self.metadata = json.load(f)

    def retrieve(self, user_story_text, top_k=3):
        query_embedding = self.retrieval_model.encode([user_story_text])
        _, indices = self.index.search(query_embedding, top_k)
        return [self.metadata[i] for i in indices[0]]

    def generate(self, user_story):
        # FIXED: Use "UserStoryDescription" and "TestSteps"
        retrieved = self.retrieve(user_story["UserStoryDescription"])
        
        prompt = f"Generate test case for: {user_story['UserStoryDescription']}\n\n"
        prompt += "Acceptance Criteria:\n" + user_story["AcceptanceCriteria"].replace('\n', '\n- ')
        prompt += "\n\nSimilar Test Cases:\n" + "\n".join([tc["TestSteps"] for tc in retrieved])

        inputs = self.t5_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.t5_model.generate(**inputs, max_length=512)
        return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    generator = TestCaseGenerator()
    user_story = {
        "UserStoryTitle": "Sample User Story",
        "UserStoryDescription": "As a user, I want to reset my password via email.",
        "AcceptanceCriteria": "Given valid email\nWhen I request a password reset\nThen a confirmation email is sent"
    }
    test_case = generator.generate(user_story)
    print("Generated Test Case:", test_case)