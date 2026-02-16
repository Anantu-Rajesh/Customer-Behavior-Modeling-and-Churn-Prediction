import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import clean_data as cd
from . import load_data as ld
import src.config as config

'''
To-Dos:
1. Set reference date (2011-09-30) :DONE BBG

2. Filter transactions to BEFORE reference date for features :DONE BBG

3. Mark transaction types (Purchase vs Cancellation) :DONE IN CLEAN_DATA BBG

4. PURCHASE FEATURES (Customer-level): DONE BBG
   - Total_Purchased (sum of purchase amounts)
   - Num_Purchase_Orders (count of purchase invoices)
   - Avg_Order_Value (Total_Purchased / Num_Orders)
   - Total_Items_Bought (sum of purchased quantities)
   - Avg_Items_Per_Order (Total_Items / Num_Orders)
   - Max_Order_Value (largest single purchase)
   - Std_Order_Value (order amount variability)

5. CANCELLATION FEATURES (Customer-level):
   - Total_Cancelled (sum of cancelled amounts)
   - Num_Cancellations (count of cancelled invoices)
   - Total_Items_Cancelled (sum of cancelled quantities)

6. DERIVED FEATURES:
   - Net_Revenue (Total_Purchased - could also track if you want gross)
   - Cancellation_Rate (Total_Cancelled / (Total_Purchased + Total_Cancelled))
   - Order_Completion_Rate (Num_Purchase_Orders / (Num_Purchase_Orders + Num_Cancellations))

7. RECENCY FEATURES (relative to reference date): DONE BBG
   - First_Purchase_Date (min InvoiceDate from purchases)
   - Last_Purchase_Date (max InvoiceDate from purchases)
   - Days_Since_Last_Purchase (reference_date - Last_Purchase_Date)
   - Days_Since_First_Purchase (reference_date - First_Purchase_Date)
   - Purchase_Span (Last_Purchase_Date - First_Purchase_Date)
   - Avg_Days_Between_Orders (Purchase_Span / (Num_Orders - 1)) [if Num_Orders > 1]

8. PRODUCT DIVERSITY (Optional, if not saving for NLP stage):
   - Num_Unique_Products (count distinct StockCode)
   - Product_Diversity_Ratio (Num_Unique_Products / Total_Items_Bought)

9. HANDLE EDGE CASES:
   - Fill NaN for customers with 1 order (Avg_Days_Between_Orders)
   - Fill NaN for customers with 0 cancellations (Cancellation_Rate = 0)
   - Check for any infinite values

10. CREATE LABELS (using data AFTER reference date):
    - Churned (1 if no purchase after reference date, 0 otherwise)

11. CLEAN UP:
    - Drop intermediate columns (raw dates, keep only derived features)
    - Keep only relevant features for modeling
    - Save to data/processed/customer_features.csv

12. VALIDATE:
    - Check for missing values
    - Check data types
    - Verify no data leakage (all features use data before reference date only)
    - Spot-check a few customers manually
'''

#1,2: finding reference date & dividing dataset wrt reference date
def set_reference_date(df,reference_date=None):
    if reference_date is None:
        reference_date = df['invoicedate'].max() - pd.DateOffset(months=3)
    print(f"Reference date set to: {reference_date.date()}")
    
    df_before=df[df['invoicedate']<=reference_date].copy()
    print(f"Transactions before reference date ({reference_date.date()}): {df_before.shape[0]}")

    df_after=df[df['invoicedate']>reference_date].copy()
    print(f"Transactions after reference date ({reference_date.date()}): {df_after.shape[0]}")
    return df_before,df_after,reference_date


#4,7: handling purchase features and recency features(wrt reference date)
def purchase_features(df_before,reference_date):
    df_temp1=df_before[~df_before['is_cancellation']].copy()
    
    #per invoice totals(to find max vals and std vals)
    order_totals = df_temp1.groupby(['customerid', 'invoiceno']).agg(
        order_total=('purchase_amnt', 'sum')
    ).reset_index()
    
    #customer wise purchase features
    df_purchase=df_temp1.groupby('customerid').agg(
        total_purchase=('purchase_amnt','sum'),
        count_orders=('invoiceno','nunique'),
        tot_items=('purchase_qty','sum'),
        first_purchase_date=('invoicedate','min'),
        last_purchase_date=('invoicedate','max'),
        num_unique_products=('stockcode', 'nunique')
    ).reset_index()
    
    order_features=order_totals.groupby('customerid').agg(
        max_order_val=('order_total', 'max'),
        min_order_val=('order_total','min'),
        std_order_val=('order_total', 'std')
    ).reset_index()
    
    df_purchase['avg_order_val'] = df_purchase['total_purchase'] / df_purchase['count_orders']
    df_purchase['avg_items_per_order'] = df_purchase['tot_items'] / df_purchase['count_orders']
    df_purchase['product_diversity_ratio'] = df_purchase['num_unique_products'] / df_purchase['tot_items']
    
    #merging order lvl feature with purchase lvl ones
    df_purchase=df_purchase.merge(order_features,on='customerid',how='left')
    
    #remaining recency features ko yaha pe kr rhe handle
    df_purchase['days_since_last_purchase'] = (reference_date - df_purchase['last_purchase_date']).dt.days
    df_purchase['days_since_first_purchase'] = (reference_date - df_purchase['first_purchase_date']).dt.days
    df_purchase['purchase_span'] = (df_purchase['last_purchase_date'] - df_purchase['first_purchase_date']).dt.days
    df_purchase['avg_days_between_orders'] = df_purchase.apply(
        lambda row: row['purchase_span'] / (row['count_orders'] - 1) if row['count_orders'] > 1 else 0,
        axis=1
    )
    
    #edge case handling
    df_purchase['std_order_val'] = df_purchase['std_order_val'].fillna(0)  
    df_purchase['min_order_val'] = df_purchase['min_order_val'].fillna(df_purchase['max_order_val'])
    
    return df_purchase
    

def feature_eng(df):
    df_before,df_after,reference_date=set_reference_date(df,reference_date=None)
    df_purchase=purchase_features(df_before,reference_date)
    print(f"df_purchase shape: {df_purchase.shape}")
    print(f"purchase features:\n")
    ld.dataset_overview(df_purchase)
    

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.org_filepath)
    df = cd.clean_data(df)
    df = feature_eng(df)
    