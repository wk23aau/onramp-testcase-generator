# File: scripts/build_domain_corpus.py

import json
import os

def main():
    input_json = os.path.join("data", "processed", "domain_knowledge.json")
    output_txt = os.path.join("data", "processed", "domain_knowledge_corpus.txt")

    with open(input_json, "r", encoding="utf-8") as f:
        domain_data = json.load(f)

    with open(output_txt, "w", encoding="utf-8") as f_out:
        for record in domain_data:
            # Combine the text fields we care about
            text_fields = [
                record.get("Description", ""),
                record.get("Acceptance Criteria", ""),
                record.get("Repro Steps", ""),
                record.get("Resolution", "")
            ]
            # Filter out empty strings, join with newlines
            combined_text = "\n".join(t for t in text_fields if t)
            # Write to file, plus a blank line separator
            if combined_text.strip():
                f_out.write(combined_text.strip() + "\n\n")

    print(f"Saved combined domain text to: {output_txt}")

if __name__ == "__main__":
    main()
