import pandas as pd
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ml.features.feature_utils import (
    calculate_time_features,
    apply_station_rolling,
    apply_station_lags,
    apply_station_deltas,
    calculate_persistence,
    calculate_neighbor_differences
)

def build_features(input_path: str, output_path: str):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    if df.empty:
        print("Empty dataset.")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort strictly by station and time to guarantee no future leakage during calculations
    df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
    
    # Base columns
    features = ['temperature_c', 'pressure_hpa', 'relative_humidity_pct']
    
    # 1. Missingness indicators
    for col in features:
        df[f'{col}_is_missing'] = df[col].isna().astype(int)
    
    # 2. Time cyclical features
    df = calculate_time_features(df, 'timestamp')
    
    def get_prefix(c: str) -> str:
        if 'humidity' in c: return 'humidity'
        return c.split('_')[0]

    # 3. Rolling & Z-Score features (1 hour, 3 hour windows assuming 1 row = 1 hour)
    for col in features:
        prefix = get_prefix(col)
        # We use pandas time-based rolling. '1h' rolling includes the current timestamp and anything in the past 1 hour.
        # By using on='timestamp', we don't need to set the index.
        
        # 1-hour features
        roll_1h = df.groupby('station_id').rolling('1h', on='timestamp')
        df[f'rolling_{prefix}_mean_1h'] = roll_1h[col].mean().values
        df[f'rolling_{prefix}_std_1h'] = roll_1h[col].std().values
        
        # 3-hour features
        roll_3h = df.groupby('station_id').rolling('3h', on='timestamp')
        df[f'rolling_{prefix}_mean_3h'] = roll_3h[col].mean().values
        df[f'rolling_{prefix}_std_3h'] = roll_3h[col].std().values
        
        # Z-Scores based on 3h rolling (using past context to standardize current)
        mean_3h = df[f'rolling_{prefix}_mean_3h']
        std_3h = df[f'rolling_{prefix}_std_3h']
        df[f'{prefix}_zscore'] = (df[col] - mean_3h) / std_3h.replace(0, pd.NA)

    # 4. Deltas
    for col in features:
        df[f'{get_prefix(col)}_delta'] = apply_station_deltas(df, col)
        
    # 5. Lags
    for col in features:
        lags_df = apply_station_lags(df, col, [1, 2]) # 1-step and 2-step lags
        df = pd.concat([df, lags_df], axis=1)
        
    # 6. Persistence
    for col in features:
        df[f'{get_prefix(col)}_persistence'] = calculate_persistence(df, col)
        
    # 7. Neighbor Differences
    for col in features:
        df[f'neighbor_{get_prefix(col)}_difference'] = calculate_neighbor_differences(df, 'timestamp', col)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete. Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build features for ML.")
    parser.add_argument('--input', type=str, default='data/anomaly/anomaly_injected_aws.csv')
    parser.add_argument('--output', type=str, default='data/processed/features.csv')
    args = parser.parse_args()
    
    build_features(args.input, args.output)
