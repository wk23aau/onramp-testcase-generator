import os
import json

def merge_json_files(root_dir):
    """
    Walks through the given root directory (e.g., data/processed/ONRAMP TCs) and 
    merges all JSON files from its subdirectories (DS, EC, SM, VM) into one list.
    """
    merged_data = []
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith('.json'):
                    file_path = os.path.join(subdir_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            merged_data.extend(data)
                        print(f"Merged {len(data)} test cases from {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    return merged_data

if __name__ == "__main__":
    # Set the root directory to your processed data for ONRAMP TCs
    root_directory = os.path.join("data", "processed", "ONRAMP TCs")
    merged_data = merge_json_files(root_directory)
    
    # Write the merged data to a single JSON file
    output_file = os.path.join("data", "processed", "ONRAMP TCs", "merged_all.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged a total of {len(merged_data)} test cases into {output_file}")
