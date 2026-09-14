import pandas as pd
import numpy as np

def calculate_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Adds cyclical hour and day_of_year features."""
    df['hour_sin'] = np.sin(2 * np.pi * df[time_col].dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df[time_col].dt.hour / 24.0)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df[time_col].dt.dayofyear / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df[time_col].dt.dayofyear / 365.25)
    return df

def apply_station_rolling(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """Calculates rolling mean and std per station without future leakage (uses current and past)."""
    # Assuming dataframe is sorted by time
    roll = df.groupby('station_id')[col].rolling(window=window, min_periods=1)
    
    # We must match the original index. groupby rolling returns MultiIndex (station_id, original_index)
    # We drop the station_id level and sort index to align with original df
    mean_series = roll.mean().reset_index(level=0, drop=True)
    std_series = roll.std().reset_index(level=0, drop=True)
    
    return mean_series, std_series

def apply_station_lags(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    """Calculates lagged values per station."""
    res = {}
    for lag in lags:
        res[f'{col}_lag_{lag}'] = df.groupby('station_id')[col].shift(lag)
    return pd.DataFrame(res)

def apply_station_deltas(df: pd.DataFrame, col: str) -> pd.Series:
    """Calculates difference from previous observation per station."""
    return df.groupby('station_id')[col].diff()

def calculate_persistence(df: pd.DataFrame, col: str) -> pd.Series:
    """Calculates number of consecutive identical observations per station."""
    # True where value changes from previous row in the same station (or is first row of station)
    changes = df[col] != df.groupby('station_id')[col].shift(1)
    
    # Cumulative sum of changes creates unique group IDs for identical consecutive values
    # Because df is sorted by station_id, the group_ids will just monotonically increase
    group_ids = changes.cumsum()
    
    # Cumulative count within these groups gives the persistence length
    return df.groupby(['station_id', group_ids]).cumcount()

def calculate_neighbor_differences(df: pd.DataFrame, time_col: str, col: str) -> pd.Series:
    """Calculates the difference between a station's reading and the mean of all other stations at the same timestamp."""
    # Mean for all stations at that timestamp
    global_mean = df.groupby(time_col)[col].transform('mean')
    global_count = df.groupby(time_col)[col].transform('count')
    
    # If a station is the only one reporting at that time, neighbor mean is undefined (NaN)
    # Mean of OTHERS = (Total Sum - My Value) / (Total Count - 1)
    
    total_sum = global_mean * global_count
    neighbor_mean = (total_sum - df[col]) / (global_count - 1)
    
    # Difference = My value - Neighbor mean
    return df[col] - neighbor_mean
