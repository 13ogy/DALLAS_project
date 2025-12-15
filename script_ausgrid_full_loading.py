import pandas as pd
import os
from glob import glob
import sys
from datetime import date

# ====================================================================
# PART 1: MERGE AND PIVOT (Your Original Script)
# ====================================================================

def merge_and_pivot_usage_data(folder_path):
    """
    Loads, pivots, and merges energy usage data from multiple CSV files
    in a given folder.
    """
    all_files = glob(os.path.join(folder_path, "*.csv"))

    if not all_files:
        print(f"Error: No CSV files found in the directory: {os.path.abspath(folder_path)}")
        if not os.path.exists(folder_path):
            print("The specified folder path does not exist.")
        return pd.DataFrame()

    final_df = None
    print(f"Found {len(all_files)} files in '{os.path.abspath(folder_path)}'. Starting merge...")

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            metadata_cols = ['Year', 'Zone Substation', 'Date', 'Unit']

            df_long = pd.melt(
                df,
                id_vars=metadata_cols,
                var_name='Time',
                value_name='Usage_Value'
            )

            # Use errors='coerce' to handle any invalid date/time formats
            df_long['Timestamp'] = pd.to_datetime(
                df_long['Date'] + ' ' + df_long['Time'],
                format='%d/%m/%Y %H:%M',
                errors='coerce'
            )

            house_name = df_long['Zone Substation'].iloc[0].replace(' ', '_')
            unit_name = df_long['Unit'].iloc[0]
            new_col_name = f"{house_name} [{unit_name}]"

            df_final_house = df_long[['Timestamp', 'Usage_Value']].rename(
                columns={'Usage_Value': new_col_name}
            )

            df_final_house.dropna(subset=['Timestamp'], inplace=True)

            if final_df is None:
                final_df = df_final_house
            else:
                final_df = pd.merge(
                    final_df,
                    df_final_house,
                    on='Timestamp',
                    how='outer'
                )
            # print(f"Successfully processed: {os.path.basename(filename)}")

        except Exception as e:
            print(f"Error processing file {os.path.basename(filename)}: {e}")
            continue

    if final_df is not None and not final_df.empty:
        final_df = final_df.sort_values(by='Timestamp').reset_index(drop=True)
        print("Merge complete.")
    else:
        print("Process finished, but the resulting DataFrame is empty.")

    return final_df

# ====================================================================
# PART 2: ENRICH DATA (New Script)
# ====================================================================

def enrich_data(energy_df, weather_file_path):
    """
    Enriches the merged energy dataframe with weather and engineered features.
    """
    print("\n--- Starting Part 2: Enriching Data ---")

    # --- 1. Load Weather Data ---
    if not os.path.exists(weather_file_path):
        print(f"Error: Weather file '{weather_file_path}' not found.")
        return energy_df

    print(f"Loading weather data: {weather_file_path}")
    weather_df = pd.read_csv(weather_file_path)

    # --- 2. Prepare for Merge ---
    print("Preparing data for merging...")
    # 'Timestamp' should already be datetime from Part 1, but we ensure it
    energy_df['Timestamp'] = pd.to_datetime(energy_df['Timestamp'])
    energy_df['merge_date'] = energy_df['Timestamp'].dt.date

    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df['merge_date'] = weather_df['datetime'].dt.date

    # --- 3. Select Relevant Weather Features ---
    selected_weather_cols = [
        'merge_date', 'tempmax', 'tempmin', 'temp', 'feelslike',
        'humidity', 'precip', 'windspeed', 'cloudcover',
        'solarradiation', 'solarenergy', 'uvindex'
    ]
    # Keep only selected cols and remove any duplicate days
    weather_df_selected = weather_df[selected_weather_cols].drop_duplicates(subset=['merge_date'])
    print(f"Selected {len(weather_df_selected.columns) - 1} weather features.")

    # --- 4. Merge Energy and Weather Data ---
    print("Merging energy and weather data...")
    merged_df = pd.merge(
        energy_df,
        weather_df_selected,
        on='merge_date',
        how='left' # Keeps all energy data, adds NaNs for non-2023 dates
    )

    # --- 5. Feature Engineering ---
    print("Starting feature engineering...")

    # a) Temporal Features
    merged_df['hour_of_day'] = merged_df['Timestamp'].dt.hour
    merged_df['day_of_week'] = merged_df['Timestamp'].dt.dayofweek # Mon=0, Sun=6
    merged_df['month'] = merged_df['Timestamp'].dt.month
    merged_df['year'] = merged_df['Timestamp'].dt.year

    # b) Season (Southern Hemisphere: Sydney)
    # 1:Summer, 2:Autumn, 3:Winter, 4:Spring
    season_map = {
        12: 1, 1: 1, 2: 1,  # Summer
        3: 2, 4: 2, 5: 2,   # Autumn
        6: 3, 7: 3, 8: 3,   # Winter
        9: 4, 10: 4, 11: 4  # Spring
    }
    merged_df['season'] = merged_df['month'].map(season_map)

    # c) Public Holidays (NSW, Australia for 2023)
    # We manually define the 2023 list.
    NSW_HOLIDAYS_2023 = [
        date(2023, 1, 1),  # New Year's Day
        date(2023, 1, 2),  # New Year's Day (observed)
        date(2023, 1, 26), # Australia Day
        date(2023, 4, 7),  # Good Friday
        date(2023, 4, 8),  # Easter Saturday
        date(2023, 4, 9),  # Easter Sunday
        date(2023, 4, 10), # Easter Monday
        date(2023, 4, 25), # Anzac Day
        date(2023, 6, 12), # King's Birthday
        date(2023, 10, 2), # Labour Day
        date(2023, 12, 25),# Christmas Day
        date(2023, 12, 26) # Boxing Day
    ]
    # Use a set for much faster lookup
    nsw_holidays_set = set(NSW_HOLIDAYS_2023)

    # This feature will be 0 for all non-2023 dates
    merged_df['is_holiday'] = merged_df['merge_date'].apply(lambda x: x in nsw_holidays_set).astype(int)

    print("Feature engineering complete.")

    # --- 6. Final Cleanup ---
    merged_df = merged_df.drop(columns=['merge_date'])
    print("Enrichment complete.")
    return merged_df

# ====================================================================
# SCRIPT EXECUTION
# ====================================================================

# --- Define Paths ---
target_folder = '../ausgrid_2024' # Folder with your raw CSVs
weather_file = 'Sydney 2023-01-01 to 2023-12-31.csv'
output_filename = 'ausgrid_enriched_data.csv'

# --- Run Part 1 ---
merged_energy_df = merge_and_pivot_usage_data(target_folder)

# --- Run Part 2 ---
if not merged_energy_df.empty:
    enriched_df = enrich_data(merged_energy_df, weather_file)

    # --- Display and Save Final Result ---
    if not enriched_df.empty:
        print("\n--- Final Enriched DataFrame Head ---")
        print(enriched_df.head())

        print("\n--- Final Enriched DataFrame Info ---")
        enriched_df.info()

        print("\n--- Missing Values (Weather-related) ---")
        # This will show how many rows are missing weather data
        weather_cols = ['tempmax', 'solarradiation', 'uvindex']
        print(enriched_df[weather_cols].isnull().sum())

        # Save the final result
        enriched_df.to_csv(output_filename, index=False)
        print(f"\nResult saved to '{output_filename}'")
else:
    print("Skipping Part 2 because Part 1 resulted in an empty DataFrame.")