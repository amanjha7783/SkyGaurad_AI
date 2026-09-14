import pandas as pd

def flag_missing_values(df: pd.DataFrame, required_cols: list) -> pd.Series:
    """Flags rows where any required column is missing."""
    return df[required_cols].isna().any(axis=1)

def flag_physical_range(df: pd.DataFrame, col: str, min_val: float, max_val: float) -> pd.Series:
    """Flags rows where the specified column is out of the physical range."""
    # Mask is True if value is outside range. We ignore NaNs here since they are caught by missing value check.
    mask = (df[col] < min_val) | (df[col] > max_val)
    return mask.fillna(False)

def flag_duplicates(df: pd.DataFrame, subset: list) -> pd.Series:
    """Flags duplicate records based on subset of columns."""
    return df.duplicated(subset=subset, keep=False)

def check_sampling_interval(df: pd.DataFrame, time_col: str, station_col: str) -> dict:
    """Calculates sampling interval statistics per station."""
    if df.empty or time_col not in df.columns or station_col not in df.columns:
        return {}
        
    df_sorted = df.dropna(subset=[time_col]).sort_values(by=[station_col, time_col])
    
    stats = {}
    for station, group in df_sorted.groupby(station_col):
        diffs = group[time_col].diff().dropna()
        if not diffs.empty:
            # Convert timedelta to seconds for easier serialization
            diff_seconds = diffs.dt.total_seconds()
            stats[station] = {
                'mean_interval_sec': float(diff_seconds.mean()),
                'median_interval_sec': float(diff_seconds.median()),
                'min_interval_sec': float(diff_seconds.min()),
                'max_interval_sec': float(diff_seconds.max()),
                'most_common_sec': float(diff_seconds.mode().iloc[0]) if not diff_seconds.mode().empty else None
            }
    return stats
