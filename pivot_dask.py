import dask.dataframe as dd
import pandas as pd
import time
from pandas.api.types import CategoricalDtype
import os

# --- Configuration ---
input_file = 'all_buildings_merged_standardized.csv'
output_file = 'all_buildings_pivoted_wide.csv'
batch_size = 50  # Process 50 buildings at a time

print(f"Starting Batched Pivot Process...")
start_time = time.time()

try:
    # --- STEP 1: Find all unique building names ---
    print(f"Step 1: Finding all unique building names in {input_file}...")

    unique_buildings = dd.read_csv(
        input_file,
        usecols=['building_name'],
        dtype={'building_name': 'str'}
    )['building_name'].unique().compute()

    building_names_list = sorted(list(unique_buildings))
    total_buildings = len(building_names_list)
    print(f"Found {total_buildings} unique buildings.")

    # --- STEP 2: Create batches ---
    batches = [building_names_list[i:i + batch_size]
               for i in range(0, total_buildings, batch_size)]
    num_batches = len(batches)
    print(f"Will process in {num_batches} batches of up to {batch_size} buildings each.")

    # --- STEP 3: Get feature table (only once) ---
    print("Step 2: Loading feature columns (season, holiday, weekend)...")

    df_features = dd.read_csv(
        input_file,
        usecols=['full_timestamp', 'season', 'is_holiday', 'is_weekend'],
        parse_dates=['full_timestamp'],
        dtype={
            'season': 'category',
            'is_holiday': 'Int8',
            'is_weekend': 'Int8'
        }
    )

    feature_df = df_features.drop_duplicates(subset=['full_timestamp']).compute()
    feature_df = feature_df.sort_values('full_timestamp').reset_index(drop=True)
    print(f"Feature table loaded: {len(feature_df)} unique timestamps.")

    # --- STEP 4: Process each batch ---
    batch_results = []

    for batch_idx, building_batch in enumerate(batches, 1):
        batch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx}/{num_batches}")
        print(f"Buildings {(batch_idx-1)*batch_size + 1} to {min(batch_idx*batch_size, total_buildings)}")
        print(f"{'='*60}")

        # Load only the rows for buildings in this batch
        print(f"  Loading data for {len(building_batch)} buildings...")

        df_batch = dd.read_csv(
            input_file,
            parse_dates=['full_timestamp'],
            dtype={
                'building_name': 'str',
                'usage_kwh': 'float64'
            },
            usecols=['full_timestamp', 'building_name', 'usage_kwh']
        )

        # Filter to only buildings in this batch
        df_batch = df_batch[df_batch['building_name'].isin(building_batch)]

        # Convert to categorical for efficient pivoting
        cat_dtype = CategoricalDtype(categories=building_batch, ordered=False)
        df_batch['building_name'] = df_batch['building_name'].astype(cat_dtype)

        print(f"  Pivoting batch {batch_idx}...")

        # Pivot this batch
        pivot_batch = dd.pivot_table(
            df_batch,
            index='full_timestamp',
            columns='building_name',
            values='usage_kwh',
            aggfunc='mean'
        ).compute()  # Compute this batch

        pivot_batch = pivot_batch.reset_index()

        batch_results.append(pivot_batch)

        batch_time = time.time() - batch_start
        print(f"  Batch {batch_idx} completed in {batch_time:.2f} seconds.")
        print(f"  Memory freed. Shape: {pivot_batch.shape}")

    # --- STEP 5: Merge all batch results horizontally ---
    print(f"\n{'='*60}")
    print("Step 3: Merging all batches together...")

    # Start with the first batch
    final_pivot = batch_results[0]

    # Merge each subsequent batch
    for i in range(1, len(batch_results)):
        print(f"  Merging batch {i+1}/{num_batches}...")
        final_pivot = pd.merge(
            final_pivot,
            batch_results[i],
            on='full_timestamp',
            how='outer'
        )

    print(f"All building columns merged. Shape: {final_pivot.shape}")

    # --- STEP 6: Add feature columns ---
    print("Step 4: Adding feature columns (season, holiday, weekend)...")

    final_df = pd.merge(
        final_pivot,
        feature_df,
        on='full_timestamp',
        how='left'
    )

    # Sort by timestamp
    final_df = final_df.sort_values('full_timestamp').reset_index(drop=True)

    print(f"Final dataframe shape: {final_df.shape}")

    # --- STEP 7: Save to CSV ---
    print(f"Step 5: Saving to {output_file}...")

    final_df.to_csv(output_file, index=False)

    total_time = time.time() - start_time
    print("-" * 60)
    print("SUCCESS!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output saved to: {output_file}")
    print(f"Final shape: {final_df.shape}")
    print("-" * 60)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()