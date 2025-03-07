import os
import pandas as pd

def extract_excel_data(input_file, output_file, sheet_name=0):
    """
    Reads an Excel file and writes its content to a CSV file.
    
    Parameters:
      - input_file: Path to the Excel file.
      - output_file: Path where the CSV file will be saved.
      - sheet_name: Name or index of the sheet to load. Defaults to 0 (the first sheet).
    """
    try:
        # Read the specified sheet (defaults to the first sheet if sheet_name=0)
        data = pd.read_excel(input_file, sheet_name=sheet_name)
        # If data is a dict, select the first sheet
        if isinstance(data, dict):
            first_sheet = list(data.keys())[0]
            df = data[first_sheet]
        else:
            df = data
        
        print(f"Loaded {input_file} with {len(df)} rows and {len(df.columns)} columns.")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the DataFrame as CSV
        df.to_csv(output_file, index=False)
        print(f"Saved raw CSV to: {output_file}")
        
        return df
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return None

def process_system(system_folder):
    """
    Processes all Excel files in a given system folder.
    """
    raw_system_dir = os.path.join("data", "raw", "ONRAMP TCs", system_folder)
    processed_system_dir = os.path.join("data", "processed", "ONRAMP TCs", system_folder)
    
    for file in os.listdir(raw_system_dir):
        if file.endswith(".xlsx"):
            input_file = os.path.join(raw_system_dir, file)
            base_name = os.path.splitext(file)[0].replace(" ", "_")
            output_file = os.path.join(processed_system_dir, f"{base_name}_raw.csv")
            extract_excel_data(input_file, output_file)

if __name__ == "__main__":
    # List the systems (subfolders) to process: DS, EC, SM, VM
    systems = ["DS", "EC", "SM", "VM"]
    for system in systems:
        print(f"\nProcessing system: {system}")
        process_system(system)
