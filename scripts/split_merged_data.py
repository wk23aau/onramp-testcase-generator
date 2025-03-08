# File: scripts/split_merged_data.py

import json
import os
from sklearn.model_selection import train_test_split

def main():
    input_path = os.path.join("data", "processed", "merged_testcases_userstories.json")
    output_dir = os.path.join("data", "processed")
    
    # Load the merged dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Split into training (80%) and temporary set (20%)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Split the temporary set equally into validation and test sets (10% each of total)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Define output file paths
    train_path = os.path.join(output_dir, "merged_train.json")
    val_path = os.path.join(output_dir, "merged_val.json")
    test_path = os.path.join(output_dir, "merged_test.json")
    
    # Save each split to its respective JSON file
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Training set: {len(train_data)} records saved to {train_path}")
    print(f"Validation set: {len(val_data)} records saved to {val_path}")
    print(f"Test set: {len(test_data)} records saved to {test_path}")

if __name__ == "__main__":
    main()
