import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.preprocessing.data_loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Ingest raw weather station data.")
    parser.add_argument('--input', type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument('--config', type=str, default='config/data_config.yaml', help="Path to data config YAML.")
    parser.add_argument('--output', type=str, default='data/processed/real_aws_baseline.csv', help="Path for processed output.")
    parser.add_argument('--metadata', type=str, default='data/processed/real_aws_baseline_metadata.json', help="Path for output metadata.")
    parser.add_argument('--source-name', type=str, default='baseline_real_aws', help="Name of the source.")
    args = parser.parse_args()

    print(f"Starting ingestion of {args.input}...")
    
    loader = DataLoader(args.config)
    
    try:
        df_standard, metadata = loader.ingest(args.input, args.source_name)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)

    # Save outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_standard.to_csv(output_path, index=False)
    
    metadata_path = Path(args.metadata)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Data successfully standardized and saved to {args.output}")
    print(f"Metadata saved to {args.metadata}")
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
