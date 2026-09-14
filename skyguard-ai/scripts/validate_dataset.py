import pandas as pd
import json

def validate_datasets():
    print("Validating datasets...")
    
    files = {
        'train': 'data/raw/train.csv',
        'validation': 'data/raw/validation.csv',
        'test': 'data/raw/test.csv'
    }
    
    report = {}
    
    for split, path in files.items():
        try:
            df = pd.read_csv(path)
            report[split] = {
                'exists': True,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'missing_values': int(df.isnull().sum().sum()),
                'duplicates': int(df.duplicated().sum()),
                'dtypes': {k: str(v) for k, v in df.dtypes.items()}
            }
            
            # Label distribution if 'is_anomaly' exists
            if 'is_anomaly' in df.columns:
                report[split]['class_distribution'] = df['is_anomaly'].value_counts().to_dict()
            elif 'label' in df.columns:
                report[split]['class_distribution'] = df['label'].value_counts().to_dict()
                
        except FileNotFoundError:
            report[split] = {'exists': False}
            
    # Check consistency
    is_consistent = True
    train_cols = set(report['train'].get('column_names', []))
    for split in ['validation', 'test']:
        if report[split]['exists']:
            cols = set(report[split].get('column_names', []))
            if cols != train_cols:
                is_consistent = False
                print(f"Warning: Columns in {split} do not match train.")
                
    report['consistency'] = 'passed' if is_consistent else 'failed'
    
    with open('data/processed/dataset_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    print(json.dumps(report, indent=4))
    print("\nDataset validation complete. Output saved to data/processed/dataset_report.json")

if __name__ == "__main__":
    validate_datasets()
