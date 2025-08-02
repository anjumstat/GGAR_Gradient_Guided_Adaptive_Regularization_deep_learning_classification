import pandas as pd
from pathlib import Path

# Define paths
root_dir = Path("E:/GGAR/results")
output_dir = root_dir / "Combined_CSVs"
output_dir.mkdir(exist_ok=True)

csv_data = {}
processed_files = 0

print("Starting CSV combination process...")

# Process all learning rate folders
for lr_folder in root_dir.iterdir():
    if not lr_folder.is_dir() or not lr_folder.name.startswith('lr_'):
        continue
        
    print(f"\nProcessing learning rate folder: {lr_folder.name}")
    
    # Extract learning rate and batch size from folder name
    try:
        # Example: lr_0_0001_bs_128
        parts = lr_folder.name.split('_')
        learning_rate = float(f"{parts[1]}.{parts[2]}")
        batch_size = int(parts[4])  # The part after _bs_
    except Exception as e:
        print(f"Error parsing folder name {lr_folder.name}: {e}")
        continue

    # Process all method folders directly under lr folder
    for method_folder in lr_folder.iterdir():
        if not method_folder.is_dir() or not method_folder.name.startswith('results_'):
            continue
            
        method = method_folder.name.replace('results_', '')
        print(f"  Processing method: {method}")

        # Process all CSV files in method folder
        for csv_file in method_folder.glob('*.csv'):
            print(f"    Processing file: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                
                # Add metadata columns
                df['learning_rate'] = learning_rate
                df['batch_size'] = batch_size
                df['method'] = method
                
                # Store in dictionary
                if csv_file.name not in csv_data:
                    csv_data[csv_file.name] = []
                csv_data[csv_file.name].append(df)
                processed_files += 1
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

# Combine and save all CSV files
print("\nCombining files...")
for csv_name, dfs in csv_data.items():
    try:
        combined_df = pd.concat(dfs, ignore_index=True)
        output_path = output_dir / csv_name
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined file: {output_path}")
    except Exception as e:
        print(f"Error combining {csv_name}: {e}")

print("\nProcessing completed!")
print(f"Total CSV files processed: {processed_files}")
print(f"Total combined files created: {len(csv_data)}")
print(f"Output directory: {output_dir}")