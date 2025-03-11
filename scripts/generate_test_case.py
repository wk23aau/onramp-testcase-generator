import os
import json
import faiss
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TestCaseGenerator:
    def __init__(self):
        try:
            logging.info("Loading retrieval model...")
            self.retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logging.info("Loading T5 model from: models/t5_finetuned")
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                os.path.join("models", "t5_finetuned")
            )
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                os.path.join("models", "t5_finetuned")
            )
            
            data_dir = os.path.join("data", "processed")
            logging.info("Loading FAISS index...")
            self.index = faiss.read_index(os.path.join(data_dir, "retrieval_index.faiss"))
            
            logging.info("Loading metadata...")
            with open(os.path.join(data_dir, "retrieval_metadata.json"), "r") as f:
                self.metadata = json.load(f)
                
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    def retrieve(self, user_story_text, top_k=3):
        # Fetch more examples and deduplicate
        query_embedding = self.retrieval_model.encode([user_story_text])
        _, indices = self.index.search(query_embedding, top_k * 2)  # Fetch more candidates
        retrieved = [self.metadata[i] for i in indices[0]]
        
        # Deduplicate by TestSteps
        seen = set()
        filtered = []
        for tc in retrieved:
            steps = tc["TestSteps"]
            if steps not in seen:
                seen.add(steps)
                filtered.append(tc)
            if len(filtered) >= top_k:
                break
        return filtered

    def generate(self, user_story):
        try:
            logging.info("Generating test case...")
            retrieved = self.retrieve(user_story["UserStoryDescription"])
            
            prompt = f"Generate test case for: {user_story['UserStoryDescription']}\n\n"
            prompt += "Acceptance Criteria:\n" + user_story["AcceptanceCriteria"].replace('\n', '\n- ')
            prompt += "\n\nSimilar Test Cases:\n" + "\n".join([tc["TestSteps"] for tc in retrieved])
            
            logging.info("Tokenizing input...")
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            logging.info("Generating output...")
            outputs = self.t5_model.generate(**inputs, max_length=512)
            
            result = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Generation complete: {result}")
            return result
        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            return "Error during generation"

# Example usage
if __name__ == "__main__":
    try:
        logging.info("Starting test case generation...")
        generator = TestCaseGenerator()
        
        user_story = {
            "UserStoryTitle": "Sample User Story",
            "UserStoryDescription": "As a user, I want to reset my password via email.",
            "AcceptanceCriteria": "Given valid email\nWhen I request a password reset\nThen a confirmation email is sent"
        }
        
        logging.info("Calling generate() method...")
        test_case = generator.generate(user_story)
        
        print("\n\n=== FINAL OUTPUT ===")
        print("Generated Test Case:", test_case)
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")