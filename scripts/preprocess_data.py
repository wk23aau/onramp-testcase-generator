import os
import pandas as pd

def standardize_columns(df):
    """
    Standardize DataFrame column names:
      - Convert to lower case.
      - Replace spaces with underscores.
    """
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

def aggregate_test_steps(group):
    """
    Given a grouped DataFrame corresponding to a single test case,
    aggregate the test steps by combining the 'test_step', 'step_action', and 'step_expected'
    columns into a single multi-line string.
    
    Each step will be formatted as:
        Step {test_step}: {step_action} -> Expected: {step_expected}
    """
    steps = []
    # Sort the group by test_step to preserve order (if numeric order exists)
    group = group.sort_values("test_step", ascending=True)
    for _, row in group.iterrows():
        step_num = str(row.get("test_step", "")).strip()
        action = str(row.get("step_action", "")).strip()
        expected = str(row.get("step_expected", "")).strip()
        step_str = f"Step {step_num}: {action}"
        if expected:
            step_str += f" -> Expected: {expected}"
        steps.append(step_str)
    return "\n".join(steps)

def process_csv_file(input_file, output_file):
    """
    Process the CSV file:
      - Standardize column names.
      - Group rows by unique test case (using 'id' and 'title').
      - Aggregate the test steps for each test case.
      - Save the resulting DataFrame to a new CSV file.
    """
    df = pd.read_csv(input_file)
    df = standardize_columns(df)
    print("Standardized columns:", df.columns.tolist())
    
    # Identify the columns that define a unique test case.
    # Adjust these keys if needed; here we assume that 'id', 'work_item_type', 'title',
    # 'area_path', 'assigned_to', and 'state' are the same for all steps of a test case.
    case_keys = ["id", "work_item_type", "title", "area_path", "assigned_to", "state"]
    
    # Ensure these columns exist
    missing_keys = [key for key in case_keys if key not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing expected columns: {missing_keys}")
    
    # Group the rows by the unique test case identifier and other metadata.
    grouped = df.groupby(case_keys, as_index=False)
    
    # For each group (i.e., test case), aggregate the test steps.
    aggregated_records = []
    for keys, group in grouped:
        record = dict(zip(case_keys, keys)) if isinstance(keys, tuple) else {case_keys[0]: keys}
        # Aggregate the test steps for this test case.
        record["flattened_test_steps"] = aggregate_test_steps(group)
        aggregated_records.append(record)
    
    # Create a DataFrame from the aggregated records.
    df_agg = pd.DataFrame(aggregated_records)
    
    # Save the aggregated DataFrame to CSV.
    df_agg.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")

if __name__ == "__main__":
    # Update these paths as needed:
    input_csv_file = "data/processed/EM/Elections_Test Cases.csv"
    output_csv_file = "data/processed/EM_cleaned/Elections_Test Cases_aggregated.csv"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    process_csv_file(input_csv_file, output_csv_file)
