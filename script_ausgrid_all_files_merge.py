import pandas as pd
import os
from glob import glob
import sys
import datetime
import numpy as np

try:
    import holidays
except ImportError:
    print("Error: The 'holidays' library is not found.")
    print("Please install it by running: pip install holidays")
    sys.exit()

def get_aus_holidays(years):
    """
    Uses the 'holidays' library to get all NSW public holidays
    (national, state, and movable) for a given range of years.
    """
    print("Generating comprehensive NSW holiday list (2000-2030)...")
    nsw_holidays = holidays.AU(state='NSW', years=years)
    return set(nsw_holidays.keys())

def process_all_buildings(base_folder, output_csv_name):
    """
    Crawls 'ausgrid_*' folders, processes CSVs, melts data,
    standardizes all usage to KWh, adds features, and merges.
    """

    # --- 1. Setup Holiday List ---
    all_years_in_data = set(range(2000, 2031))
    holiday_list = get_aus_holidays(all_years_in_data)

    # --- 2. Setup Season Map (Southern Hemisphere) ---
    season_map = {
        12: 'Summer', 1: 'Summer', 2: 'Summer',
        3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
        6: 'Winter', 7: 'Winter', 8: 'Winter',
        9: 'Spring', 10: 'Spring', 11: 'Spring'
    }

    # --- 3. Find All Files ---
    search_pattern = os.path.join(base_folder, 'ausgrid_*', '*.csv')
    all_files = glob(search_pattern)

    if not all_files:
        print(f"Error: No CSV files found matching the pattern: {search_pattern}")
        print("Please check your 'target_base_folder' path.")
        return

    print(f"\nFound {len(all_files)} files to process...")

    all_dataframes = []

    for filename in all_files:
        try:
            df = pd.read_csv(filename)

            if df.empty:
                print(f"Skipping {os.path.basename(filename)}: File is empty.")
                continue

            # Convert all columns to lowercase
            df.columns = df.columns.str.lower()

            # --- 4. Melt (Unpivot) the Data ---
            id_vars = ['year', 'zone substation', 'date', 'unit']
            df_long = pd.melt(
                df,
                id_vars=id_vars,
                var_name='time',
                value_name='usage'
            )
            df_long.dropna(subset=['usage'], inplace=True)

            # --- 5. *** NEW: Standardize Units to KWh *** ---

            # Standardize unit column for reliable matching
            df_long['unit'] = df_long['unit'].str.upper()

            # Define conditions (Power readings)
            conditions = [
                (df_long['unit'] == 'KW'),
                (df_long['unit'] == 'MW')
            ]

            # Define choices (Convert power to energy for 15-min interval)
            # 1. KW * 0.25 hours
            # 2. (MW * 1000) * 0.25 hours
            choices = [
                df_long['usage'] * 0.25,
                df_long['usage'] * 1000 * 0.25
            ]

            # Create new 'usage_kwh' column
            # Default is df_long['usage'] (for KWH units, which are already energy)
            df_long['usage_kwh'] = np.select(conditions, choices, default=df_long['usage'])

            # --- 6. Create Full Timestamp ---
            df_long['dt_date'] = pd.to_datetime(df_long['date'], format='%d%b%Y', errors='coerce')

            is_2400 = (df_long['time'] == '24:00')
            df_long.loc[is_2400, 'time'] = '00:00'

            df_long['full_timestamp'] = pd.to_datetime(
                df_long['dt_date'].dt.strftime('%Y-%m-%d') + ' ' + df_long['time'],
                format='%Y-%m-%d %H:%M',
                errors='coerce'
            )

            df_long.loc[is_2400, 'full_timestamp'] = df_long.loc[is_2400, 'full_timestamp'] + pd.Timedelta(days=1)

            df_long.dropna(subset=['full_timestamp'], inplace=True)

            # --- 7. Feature Engineering ---
            df_long['day_of_week'] = df_long['full_timestamp'].dt.dayofweek
            df_long['is_weekend'] = (df_long['day_of_week'] >= 5).astype(int)
            df_long['is_holiday'] = df_long['full_timestamp'].dt.date.isin(holiday_list).astype(int)
            df_long['season'] = df_long['full_timestamp'].dt.month.map(season_map)

            # --- 8. Final Cleanup and Selection ---
            df_long.rename(columns={'zone substation': 'building_name'}, inplace=True)

            # *** UPDATED: Use 'usage_kwh' and remove 'usage'/'unit' ***
            final_cols = [
                'building_name',
                'full_timestamp',
                'usage_kwh',      # <-- Standardized column
                'season',
                'is_holiday',
                'is_weekend'
            ]

            df_final_file = df_long[final_cols]
            all_dataframes.append(df_final_file)

            # print(f"Successfully processed: {os.path.basename(filename)}")

        except Exception as e:
            print(f"Error processing file {os.path.basename(filename)}: {e}")
            continue

    # --- 9. Concatenate All Data ---
    if not all_dataframes:
        print("No data was successfully processed.")
        return

    print("\nConcatenating all processed data...")
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # --- 10. Sort and Save ---
    print("Sorting data...")
    final_df = final_df.sort_values(by=['building_name', 'full_timestamp'])

    final_df.to_csv(output_csv_name, index=False)

    print("-" * 50)
    print(f"\nSuccess! All data merged and saved to '{output_csv_name}'")
    print(f"All usage data has been standardized to KWh.")
    print(f"Total rows: {len(final_df)}")
    print(f"Unique buildings found: {final_df['building_name'].nunique()}")
    print("--- Final DataFrame Head ---")
    print(final_df.head())

# ====================================================================
# SCRIPT EXECUTION
# ====================================================================

target_base_folder = '../ausgrid' # <--- ADJUST THIS PATH

output_file = 'all_buildings_merged_standardized.csv' # Changed output name

# Run the function
process_all_buildings(target_base_folder, output_file)

