# File: scripts/preprocess_domainknowledge.py

import sys
import csv
import json
import re
import os
import html
from bs4 import BeautifulSoup

# Increase CSV field size limit to handle large fields.
csv.field_size_limit(2147483647)

def clean_text(text):
    """
    Clean text by:
      - Converting None to an empty string.
      - Unescaping HTML entities.
      - Removing HTML/XML tags using BeautifulSoup (with newline separators to preserve structure).
      - Removing extra whitespace.
    """
    if not text:
        return ""
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator="\n", strip=True)
    lines = [re.sub(r'\s+', ' ', line).strip() for line in cleaned.split("\n")]
    return "\n".join(line for line in lines if line)

def load_csv(filepath):
    """
    Load a CSV file and return a list of dictionaries.
    """
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)

def preprocess_record(row, doc_type):
    """
    Clean common fields for a single row from Requirements, Features, or Bugs.
    Returns a dictionary with a DocType appended.
    """
    cleaned = {
        "ID": clean_text(row.get("ID", "")),
        "Work Item Type": clean_text(row.get("Work Item Type", "")),
        "Title": clean_text(row.get("Title", "")),
        "State": clean_text(row.get("State", "")),
        "Description": clean_text(row.get("Description", "")),
        "Acceptance Criteria": clean_text(row.get("Acceptance Criteria", "")),
        "Repro Steps": clean_text(row.get("Repro Steps", "")),
        "Resolution": clean_text(row.get("Resolution", "")),
        "DocType": doc_type
    }
    return cleaned

def process_csv_if_exists(filepath, doc_type, domain_data):
    """
    If the given CSV file exists, load and process its rows, appending
    them to the domain_data list with a 'DocType' field.
    """
    if os.path.exists(filepath):
        records = load_csv(filepath)
        for row in records:
            domain_data.append(preprocess_record(row, doc_type))
        print(f"Processed {len(records)} {doc_type} records from {os.path.basename(filepath)}.")
    else:
        print(f"{os.path.basename(filepath)} not found. Skipping {doc_type} processing.")

def main():
    base_raw = os.path.join("data", "raw", "ONRAMP TCs", "combined")

    # Define the CSV file paths
    requirements_csv = os.path.join(base_raw, "Requirements.csv")
    features_csv = os.path.join(base_raw, "Features.csv")
    bugs_csv = os.path.join(base_raw, "Bugs.csv")

    # Accumulate all domain knowledge here
    domain_data = []

    # Process each CSV if it exists
    process_csv_if_exists(requirements_csv, "requirement", domain_data)
    process_csv_if_exists(features_csv, "feature", domain_data)
    process_csv_if_exists(bugs_csv, "bug", domain_data)

    # Define output path
    output_path = os.path.join("data", "processed", "domain_knowledge.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the combined domain data as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(domain_data, f, indent=2)

    print(f"Combined domain knowledge saved to {output_path}")
    print(f"Total records: {len(domain_data)}")

if __name__ == "__main__":
    main()
