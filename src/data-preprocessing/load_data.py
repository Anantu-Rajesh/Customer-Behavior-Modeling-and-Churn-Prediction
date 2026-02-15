import pandas as pd
import numpy as np
import os
import src.config as config

def load_data(filepath):
    if os.path.exists(filepath):
        df=pd.read_excel(filepath)
        print(f"\nDataset loaded succesfully.")
        print(f"\nDataset shape: {df.shape}")
        print(f"\nDataset size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    else:
        print(f"Failed to load the dataset.")
    return df

def normalise_col_names(df):
    df.columns = (
        df.columns
          .str.lower()
          .str.replace(r'\(.*?\)', '', regex=True)  # remove anything inside brackets
          .str.replace(r'[^a-z0-9]+', '_', regex=True)
          .str.replace(r'_+', '_', regex=True)
          .str.strip('_')
    )
    return df

def describe_df(df):
    print("\nDataset info:")
    df.info()
    print(f"\nDataset description:\n{df.describe()}")
    print(f"\nDataset column list:\n {list(df.columns)}")
    print(f"Dataset Overview:\n")
    
def dataset_overview(df):
    print(f"{df.head(10)}")
 
def load_and_describe_data(filepath):
    df = load_data(filepath)
    df = normalise_col_names(df)
    describe_df(df)
    dataset_overview(df)
    return df   

if __name__ == "__main__":
    df=load_and_describe_data(config.org_filepath)