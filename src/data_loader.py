# src/data_loader.py
import pandas as pd
from src.config import VIDGEN_DATASET_PATH, DAVIDSON_DATASET_PATH, HATEMOJI_VALIDATION_PATH, OMG_PATH

def load_vidgen_dataset(path=VIDGEN_DATASET_PATH):
    """Loads the Vidgen dataset and maps labels."""
    vidgen_df = pd.read_csv(path)
    # Ensure the column name is correct, based on your notebook it was missing but used later.
    # Assuming 'label' is the column that needs mapping.
    vidgen_df = vidgen_df[['text', 'label']] # Adjusted based on your usage
    label_map = {'nothate': 0, 'hate': 1}
    vidgen_df['label'] = vidgen_df['label'].map(label_map)
    print(f"Loaded Vidgen dataset from {path}")
    print("Vidgen Label distribution:\n", vidgen_df['label'].value_counts()) #
    return vidgen_df

def load_davidson_dataset(path=DAVIDSON_DATASET_PATH):
    """Loads the Davidson dataset and converts labels to binary."""
    davidson_df = pd.read_csv(path)
    davidson_df['label'] = davidson_df['class'].apply(lambda x: 0 if x == 2 else 1) #
    davidson_df['text'] = davidson_df['tweet'] # Align column name
    davidson_df.drop(columns=['class', 'tweet'], inplace=True, errors='ignore')
    print(f"Loaded Davidson dataset from {path}")
    print("Davidson Label distribution:\n", davidson_df['label'].value_counts()) #
    return davidson_df

def load_omg_dataset(path=OMG_PATH):
    # Load dataset
    df = pd.read_csv(path)
    print(f"Loaded OMG dataset from {path}")
    print("OMG Label distribution:\n", df['label'].value_counts()) #
    return df[['text', 'label']]