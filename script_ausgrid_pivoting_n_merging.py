import pandas as pd
import os
from glob import glob
import sys

def merge_and_pivot_usage_data(folder_path):
    """
    Loads, pivots, and merges energy usage data from multiple CSV files
    in a given folder, based on the specific file format provided.

    Args:
        folder_path (str): The path to the folder containing the CSV files.

    Returns:
        pd.DataFrame: A single DataFrame with a 'Timestamp' column and
                      individual columns for each house's usage data.
    """

    # 1. Find all CSV files in the specified folder
    # We use os.path.join for cross-platform compatibility
    all_files = glob(os.path.join(folder_path, "*.csv"))

    if not all_files:
        print(f"Error: No CSV files found in the directory: {os.path.abspath(folder_path)}")
        # Check if the path exists to give a more informative error
        if not os.path.exists(folder_path):
            print("The specified folder path does not exist. Please check the path.")
        return pd.DataFrame() # Return an empty DataFrame

    final_df = None

    print(f"Found {len(all_files)} files in '{os.path.abspath(folder_path)}'. Starting merge process...")

    for filename in all_files:
        try:
            # 2. Load the file. The first row contains the headers.
            df = pd.read_csv(filename)

            metadata_cols = ['Year', 'Zone Substation', 'Date', 'Unit']

            # --- Data Transformation (Pivoting/Melting) ---

            # 3. Melt (unpivot) the time columns into a long format
            df_long = pd.melt(
                df,
                id_vars=metadata_cols,
                var_name='Time',
                value_name='Usage_Value'
            )

            # --- Column Generation and Cleanup ---

            # 4. Combine 'Date' and 'Time' to create the final 'Timestamp'
            # Note: We use errors='coerce' to turn invalid date/time combinations into NaT
            df_long['Timestamp'] = pd.to_datetime(
                df_long['Date'] + ' ' + df_long['Time'],
                format='%d/%m/%Y %H:%M',
                errors='coerce'
            )

            # 5. Create the required house usage column name: [Zone Substation] [Unit]
            # Get the unique identifier from the first row of the 'long' data
            house_name = df_long['Zone Substation'].iloc[0].replace(' ', '_')
            unit_name = df_long['Unit'].iloc[0]
            new_col_name = f"{house_name} [{unit_name}]"

            # Prepare the data for merging
            df_final_house = df_long[['Timestamp', 'Usage_Value']].rename(
                columns={'Usage_Value': new_col_name}
            )

            # Drop rows with invalid timestamps
            df_final_house.dropna(subset=['Timestamp'], inplace=True)

            # --- Merging ---

            if final_df is None:
                final_df = df_final_house
            else:
                # Merge the current file's data into the main DataFrame on 'Timestamp'
                final_df = pd.merge(
                    final_df,
                    df_final_house,
                    on='Timestamp',
                    how='outer'
                )

            print(f"Successfully processed and merged: {os.path.basename(filename)}")

        except Exception as e:
            print(f"Error processing file {os.path.basename(filename)}: {e}")
            continue

    if final_df is not None and not final_df.empty:
        final_df = final_df.sort_values(by='Timestamp').reset_index(drop=True)
        print("\nMerge complete.")
    else:
        print("\nProcess finished, but the resulting DataFrame is empty.")

    return final_df

# ====================================================================
# SCRIPT EXECUTION
# ====================================================================

target_folder = '../ausgrid_2024'

# Run the function
result_df = merge_and_pivot_usage_data(target_folder)

# Display the final result
if not result_df.empty:
    print("\n--- Final Merged DataFrame Head ---")
    print("Columns:", result_df.columns.tolist())
    print(result_df.head())

    # Save the final result to a new CSV file in the script's repo
    output_filename = 'ausgrid_merged_usage_data.csv'
    result_df.to_csv(output_filename, index=False)
    print(f"\nResult saved to '{output_filename}' in the script directory.")