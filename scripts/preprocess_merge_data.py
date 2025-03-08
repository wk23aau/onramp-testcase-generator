# File: scripts/preprocess_merge_data.py

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
      - Removing HTML/XML tags using BeautifulSoup (with newline separators to preserve block structure).
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

def extract_ticket_id(title):
    """
    Extract the ticket number from the test case title.
    Example: "Ticket 50223 - WV - VOTER MANAGEMENT - ..." returns "50223".
    """
    match = re.search(r'Ticket\s+(\d+)', title)
    return match.group(1) if match else None

def extract_leading_numeric(title):
    """
    Extract a leading numeric code from the test case title.
    Example: "92616 VM - ..." returns "92616".
    """
    match = re.match(r'^(\d+)', title)
    return match.group(1) if match else None

def merge_data(userstories, testcases):
    """
    Preprocess and merge test cases with user stories by matching:
      1. 'Ticket ###' from test case title to user story ID.
      2. If not found, a leading numeric code from test case title to user story ID.
      3. If that fails, check if the user story title is a substring of the test case title.
      4. Otherwise, ignore the test case.

    All key text fields are cleaned.
    """
    # Build a dictionary for user stories keyed by their cleaned ID.
    us_dict = {}
    for row in userstories:
        us_id = clean_text(row.get("ID"))
        row["ID"] = us_id
        row["Title"] = clean_text(row.get("Title", ""))
        row["Description"] = clean_text(row.get("Description", ""))
        row["Acceptance Criteria"] = clean_text(row.get("Acceptance Criteria", ""))
        us_dict[us_id] = row

    merged = []
    for tc in testcases:
        tc["ID"] = clean_text(tc.get("ID"))
        tc["Title"] = clean_text(tc.get("Title", ""))
        tc["Area Path"] = clean_text(tc.get("Area Path", ""))
        tc["Steps"] = clean_text(tc.get("Steps", ""))

        # Step 1: Try to extract 'Ticket ###' from the test case title.
        ticket_id = extract_ticket_id(tc["Title"])
        us_match = None
        match_found = False

        if ticket_id and ticket_id in us_dict:
            us_match = us_dict[ticket_id]
            match_found = True
        else:
            # Step 2: If no 'Ticket ###' match, extract a leading numeric code.
            leading_num = extract_leading_numeric(tc["Title"])
            if leading_num and leading_num in us_dict:
                us_match = us_dict[leading_num]
                match_found = True
            else:
                # Step 3: Fallback - check if any user story title is contained in the test case title.
                tc_title_lower = tc["Title"].lower()
                for us in us_dict.values():
                    us_title_lower = us["Title"].lower()
                    if us_title_lower and us_title_lower in tc_title_lower:
                        us_match = us
                        match_found = True
                        break

        # Step 4: Only merge if a match was found.
        if match_found and us_match:
            merged_record = {
                "UserStoryID": us_match.get("ID", ""),
                "UserStoryTitle": us_match.get("Title", ""),
                "UserStoryDescription": us_match.get("Description", ""),
                "AcceptanceCriteria": us_match.get("Acceptance Criteria", ""),
                "TestCaseID": tc.get("ID", ""),
                "TestCaseTitle": tc.get("Title", ""),
                "AreaPath": tc.get("Area Path", ""),
                "TestSteps": tc.get("Steps", "")
            }
            merged.append(merged_record)

    return merged

def main():
    # Define file paths for raw CSV files in the combined folder.
    base_raw = os.path.join("data", "raw", "ONRAMP TCs", "combined")
    userstories_csv = os.path.join(base_raw, "UserStories.csv")
    testcases_csv = os.path.join(base_raw, "test-cases.csv")
    
    # Load CSV files.
    userstories = load_csv(userstories_csv)
    testcases = load_csv(testcases_csv)
    
    # Merge and preprocess the data.
    merged = merge_data(userstories, testcases)
    
    # Define output path and ensure the directory exists.
    output_path = os.path.join("data", "processed", "merged_testcases_userstories.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the merged data as a JSON file.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    
    print(f"Merged {len(merged)} records saved to {output_path}")

if __name__ == "__main__":
    main()
