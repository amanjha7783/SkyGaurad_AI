import pandas as pd
import json
import argparse
from pathlib import Path
from ml.preprocessing.quality_checks import (
    flag_missing_values,
    flag_physical_range,
    flag_duplicates,
    check_sampling_interval
)

def run_cleaning_pipeline(input_path: str, clean_out: str, quarantine_out: str, report_out: str, summary_out: str):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    reasons = pd.Series([[] for _ in range(len(df))], index=df.index)

    # 1. & 3. Timestamp normalization & missing value analysis
    # Ensure vital columns are not missing
    vital_cols = ['timestamp', 'station_id', 'temperature_c']
    missing_mask = flag_missing_values(df, [c for c in vital_cols if c in df.columns])
    for idx in df[missing_mask].index:
        reasons[idx].append('missing_required_value')

    # 4. & 5. & 6. & 7. Physical range and quality checks
    temp_mask = pd.Series(False, index=df.index)
    if 'temperature_c' in df.columns:
        temp_mask = flag_physical_range(df, 'temperature_c', -50, 60)
        for idx in df[temp_mask].index:
            reasons[idx].append('invalid_temperature')

    press_mask = pd.Series(False, index=df.index)
    if 'pressure_hpa' in df.columns:
        press_mask = flag_physical_range(df, 'pressure_hpa', 800, 1100)
        for idx in df[press_mask].index:
            reasons[idx].append('invalid_pressure')

    rh_mask = pd.Series(False, index=df.index)
    if 'relative_humidity_pct' in df.columns:
        rh_mask = flag_physical_range(df, 'relative_humidity_pct', 0, 100)
        for idx in df[rh_mask].index:
            reasons[idx].append('invalid_humidity')

    # 8. Station metadata validation (simple check for now)
    station_mask = pd.Series(False, index=df.index)
    if 'station_id' in df.columns:
        station_mask = df['station_id'].isna() | (df['station_id'] == "")
        for idx in df[station_mask].index:
            reasons[idx].append('invalid_station_metadata')

    # 2. Duplicate detection
    dup_mask = pd.Series(False, index=df.index)
    if 'timestamp' in df.columns and 'station_id' in df.columns:
        dup_mask = flag_duplicates(df, ['timestamp', 'station_id'])
        for idx in df[dup_mask].index:
            reasons[idx].append('duplicate_record')

    # Quarantine classification
    quarantine_mask = reasons.apply(lambda x: len(x) > 0)
    df['quarantine_reasons'] = reasons.apply(lambda x: "|".join(x))

    clean_df = df[~quarantine_mask].drop(columns=['quarantine_reasons'])
    quarantine_df = df[quarantine_mask]

    # 10. Outlier reporting (Row-by-row quarantine reasons)
    report_df = quarantine_df[['timestamp', 'station_id', 'quarantine_reasons']] if not quarantine_df.empty else pd.DataFrame(columns=['timestamp', 'station_id', 'quarantine_reasons'])

    # 9. Sampling interval analysis on clean data
    sampling_freq = check_sampling_interval(clean_df, 'timestamp', 'station_id')

    # Build Summary
    summary = {
        'input_rows': len(df),
        'clean_rows': len(clean_df),
        'quarantined_rows': len(quarantine_df),
        'missing_values_flagged': int(missing_mask.sum()),
        'duplicates_flagged': int(dup_mask.sum()),
        'invalid_temp_flagged': int(temp_mask.sum()),
        'invalid_pressure_flagged': int(press_mask.sum()),
        'invalid_humidity_flagged': int(rh_mask.sum()),
        'invalid_station_flagged': int(station_mask.sum()),
        'sampling_frequency': sampling_freq
    }

    # Save outputs
    Path(clean_out).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_out, index=False)
    quarantine_df.to_csv(quarantine_out, index=False)
    
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_out, index=False)
    
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Cleaning complete. {len(clean_df)} clean rows, {len(quarantine_df)} quarantined.")
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean standardized AWS data.")
    parser.add_argument('--input', type=str, default='data/processed/real_aws_baseline.csv')
    parser.add_argument('--clean-out', type=str, default='data/processed/clean_aws.csv')
    parser.add_argument('--quarantine-out', type=str, default='data/processed/quarantine.csv')
    parser.add_argument('--report-out', type=str, default='reports/data_quality_report.csv')
    parser.add_argument('--summary-out', type=str, default='reports/data_quality_summary.json')
    
    args = parser.parse_args()
    
    run_cleaning_pipeline(
        input_path=args.input,
        clean_out=args.clean_out,
        quarantine_out=args.quarantine_out,
        report_out=args.report_out,
        summary_out=args.summary_out
    )
