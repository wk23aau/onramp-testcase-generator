# OnRamp Testcase Generator

## Overview
The OnRamp Testcase Generator is a research project focused on automating the creation of detailed manual test cases from raw user stories and associated test cases. It uses advanced NLP techniques to:
- **Understand** user stories and test case language.
- **Retrieve** similar test cases via embedding similarity.
- **Generate** complete test cases (titles, steps, expected outcomes) using a generative model.

---

## Data Sources
Raw data is stored in `data/raw/ONRAMP TCs/combined` and includes:
- **Test Cases**: CSV files with IDs, titles, area paths, and XML/HTML-formatted steps.
- **User Stories**: CSV files with IDs, titles, descriptions, and acceptance criteria.

---

## Research & Approach

### Model Selection
#### **T5 (Text-to-Text Transfer Transformer)**  
Chosen for its generative capabilities, T5 converts tasks into a text-to-text format, making it ideal for generating test cases from user stories. It combines BERT-like language understanding with generation.

#### **Why Not BERT or ABSA?**  
- **BERT**: Strong for understanding but requires additional components for generation.  
- **ABSA**: Focuses on sentiment/opinion extraction, not procedural text generation.

---

### Retrieval-Augmented Generation
1. **Embedding-Based Similarity**:  
   - Use `SentenceTransformer` (e.g., `all-MiniLM-L6-v2`) to compute embeddings for user stories and test cases.  
   - Retrieve similar test cases via cosine similarity.  

2. **Retrieval-Augmented Prompt**:  
   Inject retrieved test cases into the prompt to guide T5's output style/structure.  

3. **Contrastive Learning**:  
   Refine embeddings to improve relevance of retrieved test cases.

---

### Pipeline Overview
1. **Data Loading**: Merge CSV files into a JSON dataset pairing user stories with test cases.  
2. **Preprocessing**: Clean text, extract key fields (e.g., `UserStoryTitle`, `TestSteps`).  
3. **Retrieval Module**: Fetch similar test cases using embeddings.  
4. **Prompt Construction**: Combine user story, retrieved test case, and instructions into a T5-friendly format.  
5. **Model Training**: Fine-tune T5 using Hugging Face's `Trainer`.

---

## Project Structure
```plaintext
onramp-testcase-generator/
├── data/
│   ├── raw/
│   │   └── ONRAMP TCs/combined/  # Raw CSV files
│   └── processed/
│       └── merged_testcases_userstories.json  # Preprocessed dataset
├── models/
│   └── t5_retrieval_generator/  # Fine-tuned model
├── scripts/
│   ├── train_retrieval_gen.py  # Training script
│   └── ...  # Additional scripts
├── README.md
└── requirements.txt
```
---

## Training Script
The script scripts/train_retrieval_gen.py handles:

- Data loading/preprocessing.
- Retrieval-augmented prompt construction.
- T5 tokenization and fine-tuning.
- Dependencies
- Install required packages:

```bash
pip install transformers sentence-transformers datasets torch
```
---
## Future Enhancements
- Dual Encoder Models: Train joint embeddings for user stories and test cases.
- Contrastive Learning: Improve retrieval relevance.
- Multi-Stage Generation: Refine output quality with cross-attention mechanisms.
---
## Conclusion
This project automates test case generation by combining T5's generative power with retrieval-augmented prompts, producing structured test cases aligned with user story requirements.
