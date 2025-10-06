import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_and_chart_usage(file_path='merged_usage_data.csv'):
    """
    Loads merged usage data, analyzes the first building's data,
    calculates the mean usage, and generates a line chart.

    Args:
        file_path (str): The path to the merged CSV file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure the merging script was run successfully and created this file.")
        return

    try:
        # 1. Load the merged data
        df = pd.read_csv(file_path)

        if df.shape[1] < 2:
            print("Error: The DataFrame must have at least two columns (Timestamp and Usage).")
            return

        # 2. Identify and select columns
        timestamp_col = df.columns[0]
        usage_col = df.columns[1] # Selects the first building's usage column

        # Select the two columns
        df_selected = df[[timestamp_col, usage_col]].copy()

        print(f"--- Analyzing Data for: {usage_col} ---")

        # 3. Convert the Timestamp column to datetime objects for accurate plotting
        df_selected[timestamp_col] = pd.to_datetime(df_selected[timestamp_col])

        # Set Timestamp as index for time series plotting
        df_selected.set_index(timestamp_col, inplace=True)

        # Ensure the usage column is numeric
        df_selected[usage_col] = pd.to_numeric(df_selected[usage_col], errors='coerce')
        df_selected.dropna(subset=[usage_col], inplace=True)

        # 4. Calculate Mean Usage
        mean_usage = df_selected[usage_col].mean()
        print(f"Calculated Mean Usage for '{usage_col}': {mean_usage:,.4f}")
        print("-" * 40)

        # 5. Generate and Save a Chart (Line Plot)
        plt.figure(figsize=(12, 6))
        plt.plot(df_selected.index, df_selected[usage_col], label=usage_col, color='tab:blue', linewidth=0.8)

        # Add a horizontal line for the mean
        plt.axhline(mean_usage, color='r', linestyle='--', label=f'Mean: {mean_usage:,.4f}')

        plt.title(f"Usage Data for {usage_col}", fontsize=14)
        plt.xlabel("Timestamp", fontsize=12)

        # Extract the unit from the column name for the Y-axis label
        unit = usage_col.split('[')[-1].replace(']', '').strip()
        plt.ylabel(f"Usage ({unit})", fontsize=12)

        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Improve X-axis date visibility
        plt.gcf().autofmt_xdate()

        # Save the plot
        chart_filename = f"usage_chart_{usage_col.split(' ')[0]}.png"
        plt.savefig(chart_filename)
        plt.close() # Close the plot figure to free memory

        print(f"Successfully generated line chart and saved it as: {chart_filename}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")


# ====================================================================
# SCRIPT EXECUTION
# ====================================================================

# This is the expected output file from the previous merging script.
merged_data_file = 'ausgrid_merged_usage_data.csv'

# Run the function
analyze_and_chart_usage(merged_data_file)