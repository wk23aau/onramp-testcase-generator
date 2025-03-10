import os
import json

data_dir = "C:/Users/wk23aau/Documents/onramp-testcase-generator/data/processed/"

for split in ["train", "val", "test"]:
    file_path = os.path.join(data_dir, f"merged_{split}.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"{split.capitalize()} set size: {len(data)}")