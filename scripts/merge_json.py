import os
import json

def merge_json_files(directory):
    merged_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.extend(data)
    return merged_data

if __name__ == "__main__":
    ec_dir = os.path.join("data", "processed", "ONRAMP TCs", "EC")
    merged = merge_json_files(ec_dir)
    output_file = os.path.join(ec_dir, "merged.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(merged)} test cases into {output_file}")
