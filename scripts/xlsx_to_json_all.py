import os
import json
import pandas as pd

def process_excel_to_json(input_file):
    """
    Reads an Excel file and converts its rows into a nested JSON structure.
    
    The Excel file is expected to have the following columns:
      - ID
      - Work Item Type
      - Title
      - Test Step
      - Step Action
      - Step Expected
      - Area Path
      - Assigned To
      - State
      
    The logic is:
      - A non-empty 'ID' cell signals the start of a new test case.
      - Test step details are added to the current test case.
      - At the end, all test cases are returned as a list.
    """
    # Read the Excel file and fill NaN with empty strings for consistency.
    df = pd.read_excel(input_file)
    df = df.fillna("")
    
    test_cases = []
    current_case = None
    
    for _, row in df.iterrows():
        # New test case detected if 'ID' is non-empty.
        if row["ID"]:
            if current_case:
                test_cases.append(current_case)
            current_case = {
                "ID": int(row["ID"]) if row["ID"] != "" else None,
                "Work Item Type": row["Work Item Type"],
                "Title": row["Title"],
                "Test Steps": [],
                "Area Path": row["Area Path"].replace("SG\\", "SG\\\\"),
                "Assigned To": row["Assigned To"],
                "State": row["State"]
            }
        
        # If the row contains a test step, add it to the current case.
        if row["Test Step"]:
            try:
                step_num = int(row["Test Step"]) if row["Test Step"] != "" else None
            except Exception:
                step_num = row["Test Step"]
            step = {
                "Step": step_num,
                "Step Action": row["Step Action"].replace("\n", "\\n"),
                "Step Expected": row["Step Expected"].replace("\n", "\\n")
            }
            if current_case is not None:
                current_case["Test Steps"].append(step)
    
    # Append the last test case if available.
    if current_case:
        test_cases.append(current_case)
    
    return test_cases

def process_system(system):
    """
    Processes all Excel files within a given system subfolder.
    
    Reads each Excel file from:
      data/raw/ONRAMP TCs/<system>/
    and writes a corresponding JSON file into:
      data/processed/ONRAMP TCs/<system>/
    """
    raw_system_dir = os.path.join("data", "raw", "ONRAMP TCs", system)
    processed_system_dir = os.path.join("data", "processed", "ONRAMP TCs", system)
    os.makedirs(processed_system_dir, exist_ok=True)
    
    for file in os.listdir(raw_system_dir):
        if file.lower().endswith(".xlsx"):
            input_file = os.path.join(raw_system_dir, file)
            base_name = os.path.splitext(file)[0].replace(" ", "_")
            output_file = os.path.join(processed_system_dir, f"{base_name}.json")
            print(f"\nProcessing {input_file} ...")
            try:
                test_cases = process_excel_to_json(input_file)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(test_cases, f, indent=2)
                print(f"Saved aggregated JSON to: {output_file}")
            except Exception as e:
                print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    systems = ["DS", "EC", "SM", "VM"]
    for system in systems:
        print(f"\nProcessing system: {system}")
        process_system(system)
