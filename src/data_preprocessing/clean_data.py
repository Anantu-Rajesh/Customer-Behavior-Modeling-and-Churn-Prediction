import os
import numpy as np
import pandas as pd
from . import load_data as ld
import src.config as config

'''
To-Dos:
1. Handle missing vals
2. Handle duplicate vals
3. Handle invalid unit prices
4. Handle invoice number format
5. Handle quantity column(-ve value indicates cancellation)
6. Handle invoice date column
'''

def handle_missing_data(df):
    print(f"\nMissing values in Dataset: {df.isnull().sum()}")
    print(f"\nMissing values found in customerid column thus removing them is the best way to handle them.")
    df=df.dropna(subset=['customerid'])
    print(f"\nAfter handling missing values, Dataset shape: {df.shape}")
    return df

def handle_duplicates(df):
    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of exact duplicate rows in the dataset: {duplicate_count}")
    if duplicate_count > 0: 
        print(f"\nRemoving duplicate rows from the dataset.")
        df = df.drop_duplicates()
        print(f"\nAfter removing duplicates, Dataset shape: {df.shape}")
    return df

def handle_invalid_prices(df):
    print(f"\nInvalid unit prices in the dataset: {(df['unitprice']<=0).sum()}")
    print(f"\nRemoving the invalid unit prices.")
    df=df[df['unitprice']>0]
    print(f"\nAfter handling, invalid unit prices in the dataset: {(df['unitprice']<=0).sum()}")
    return df

def standardize_invoice_number(df):
    df["invoiceno"] = df["invoiceno"].astype(str).str.strip()
    return df

def handle_quantity(df):
    print(f"\nHandling the quantity column")
    print(f"\nIn the given dataset, -ve quantity indicates cancellation cases while +ve quantity indicates actual purchase case")
    df["is_cancellation"] = df["invoiceno"].astype(str).str.startswith("C")
    df["purchase_qty"] = df["quantity"].clip(lower=0)
    df["cancel_qty"] = (-df["quantity"].clip(upper=0))
    df['cancel_amnt']=df['cancel_qty']*df['unitprice']
    df['purchase_amnt']=df['purchase_qty']*df['unitprice']
    #Final check
    assert (
        (df.loc[df["is_cancellation"], "purchase_qty"] == 0).all()
    ), "C invoices contain purchase_qty"

    assert (
        (df.loc[~df["is_cancellation"], "cancel_qty"] == 0).all()
    ), "Non-C invoices contain cancel_qty"

    return df

def handle_invoice_date(df):
    min_date=df['invoicedate'].min()
    max_date=df['invoicedate'].max()
    print(f"Date ranges from {min_date} to {max_date}")
    df['date_months']=df['invoicedate'].dt.to_period('M')
    return df


def clean_data(df):
    df=handle_missing_data(df)
    df=handle_duplicates(df)
    df=handle_invalid_prices(df)
    df=handle_quantity(df)
    df=standardize_invoice_number(df)
    df=handle_invoice_date(df)
    print(f"Overview of transactional dataset after cleaning:\n")
    ld.dataset_overview(df)
    return df
    
    
if __name__ == "__main__":
    df = ld.load_and_describe_data(config.org_filepath)
    df = clean_data(df)
    