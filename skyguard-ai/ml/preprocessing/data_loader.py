import pandas as pd
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.schema = self.config.get('schema', {})
        self.validation_rules = self.config.get('validation', {})

    def detect_columns(self, columns: list) -> Dict[str, str]:
        """Map raw columns to standard schema columns."""
        mapping = {}
        for raw_col in columns:
            raw_col_lower = raw_col.lower().strip()
            mapped = False
            for std_col, rules in self.schema.items():
                if raw_col_lower == std_col or raw_col_lower in rules.get('aliases', []):
                    mapping[raw_col] = std_col
                    mapped = True
                    break
            if not mapped:
                logger.warning(f"Column '{raw_col}' could not be mapped to standard schema and will be ignored.")
        return mapping

    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate dataframe against rules and handle missing/invalid data."""
        report = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "validation_errors": []
        }

        # Timestamp duplicate check
        if 'timestamp' in df.columns and 'station_id' in df.columns:
            duplicates = df.duplicated(subset=['timestamp', 'station_id']).sum()
            if duplicates > 0:
                report["validation_errors"].append(f"Found {duplicates} duplicate timestamp-station entries.")
                df = df.drop_duplicates(subset=['timestamp', 'station_id'], keep='first')

        # Value bounds validation
        for col, rules in self.validation_rules.items():
            if col in df.columns:
                invalid_mask = pd.Series(False, index=df.index)
                if 'min' in rules:
                    invalid_mask |= (df[col] < rules['min'])
                if 'max' in rules:
                    invalid_mask |= (df[col] > rules['max'])
                
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    report["validation_errors"].append(f"Column '{col}' has {invalid_count} out-of-bound values.")
                    df.loc[invalid_mask, col] = pd.NA # Set invalid to NA rather than dropping row

        return df, report

    def ingest(self, source_path: str, source_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main ingestion function."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file {source_path} not found.")

        # Read without modification
        df_raw = pd.read_csv(path)
        
        # Mapping
        mapping = self.detect_columns(df_raw.columns.tolist())
        df_mapped = df_raw[list(mapping.keys())].rename(columns=mapping)

        # Enforce timestamp parsing
        if 'timestamp' in df_mapped.columns:
            df_mapped['timestamp'] = pd.to_datetime(df_mapped['timestamp'], errors='coerce')
        
        # Add source
        df_mapped['source'] = source_name

        # Ensure all schema columns exist
        for std_col in self.schema.keys():
            if std_col not in df_mapped.columns:
                df_mapped[std_col] = pd.NA
        
        # Reorder to match schema
        df_standard = df_mapped[list(self.schema.keys())].copy()
        
        # Validate
        df_validated, validation_report = self.validate_data(df_standard)

        # Build Metadata
        metadata = {
            "source_file": str(path.name),
            "source_name": source_name,
            "original_columns": df_raw.columns.tolist(),
            "mapped_columns": mapping,
            "time_range": {
                "start": df_validated['timestamp'].min().isoformat() if not pd.isna(df_validated['timestamp'].min()) else None,
                "end": df_validated['timestamp'].max().isoformat() if not pd.isna(df_validated['timestamp'].max()) else None
            } if 'timestamp' in df_validated.columns else None,
            "stations": df_validated['station_id'].dropna().unique().tolist() if 'station_id' in df_validated.columns else [],
            "row_count": len(df_validated),
            "validation_report": validation_report
        }

        return df_validated, metadata

