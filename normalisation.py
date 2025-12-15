import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

# --- Configuration ---
INPUT_FILE = 'all_buildings_pivoted_wide.csv'
OUTPUT_FILE = 'final_ausgrid_scaled.csv'

print(f"Loading {INPUT_FILE}... This may take a moment.")
start_time = time.time()

try:
    # --- 1. Load the data ---
    df = pd.read_csv(INPUT_FILE)
    df['full_timestamp'] = pd.to_datetime(df['full_timestamp'])

    print(f"File loaded in {time.time() - start_time:.2f} seconds.")
    print(f"Data has {len(df)} rows and {len(df.columns)} columns.")

    # --- 2. Identify Metadata vs. Building Columns ---

    # These are the columns we *don't* want to scale
    metadata_cols = ['full_timestamp', 'season', 'is_holiday', 'is_weekend']

    # These are the columns we *do* want to clean and scale
    building_cols = [col for col in df.columns if col not in metadata_cols]

    print(f"Identified {len(building_cols)} building columns to process.")

    # --- 3. Step 1: EDA & Imputation ---

    print("Starting Step 1: Imputation...")

    # a) Interpolate small, intermittent gaps
    # We set a 'limit' of 4 (1 hour) so we only fill small gaps,
    # not massive multi-year gaps.
    print("  Applying linear interpolation (limit=1 hour)...")
    df[building_cols] = df[building_cols].interpolate(
        method='linear',
        limit_direction='both',
        limit=4
    )

    # b) Fill remaining large gaps with 0
    # These are the structural NaNs (e.g., pre-commissioning)
    print("  Filling remaining structural NaNs with 0...")
    df[building_cols] = df[building_cols].fillna(0)

    print("Step 1 complete.")

    # --- 4. Step 3: Data Normalization (Scaling) ---

    print("Starting Step 3: Scaling...")

    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform the building data
    # This scales each column independently
    scaled_data = scaler.fit_transform(df[building_cols])

    # Put the scaled data back into the DataFrame
    df[building_cols] = scaled_data

    print("Step 3 complete.")

    # --- 5. Save the final processed data ---
    print(f"Saving final dataset to {OUTPUT_FILE}...")

    df.to_csv(OUTPUT_FILE, index=False)

    end_time = time.time()

    print("-" * 50)
    print("Success!")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print(f"Final processed & scaled data saved to '{OUTPUT_FILE}'")

    print("\n--- Final DataFrame Head (Showing scaled data) ---")
    print(df.head())

except Exception as e:
    print(f"An error occurred: {e}")