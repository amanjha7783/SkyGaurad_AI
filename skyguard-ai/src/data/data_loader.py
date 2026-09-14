import pandas as pd

def load_data():
    train_df = pd.read_csv('data/raw/train.csv')
    val_df = pd.read_csv('data/raw/validation.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    return train_df, val_df, test_df
