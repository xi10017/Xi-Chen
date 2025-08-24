#!/usr/bin/env python3
"""
Debug script to check data ranges for search terms vs flu data
"""

import pandas as pd
import numpy as np
from time_series_analysis_specific import load_data, create_date_index

def debug_data_ranges():
    """Check the data ranges for search terms vs flu data"""
    
    print("Loading data...")
    df_ili, df_nrevss, df_search = load_data()
    
    # Test term
    test_term = 'h1n1'
    
    print(f"\n=== DATA RANGES FOR TERM: {test_term} ===")
    
    # Check search term data
    if test_term in df_search.columns:
        search_data = df_search[test_term]
        print(f"Search term '{test_term}' data:")
        print(f"  Min: {search_data.min()}")
        print(f"  Max: {search_data.max()}")
        print(f"  Mean: {search_data.mean():.2f}")
        print(f"  Std: {search_data.std():.2f}")
        print(f"  Non-zero values: {(search_data > 0).sum()}")
        print(f"  Total values: {len(search_data)}")
    else:
        print(f"Term '{test_term}' not found in search data")
        print(f"Available columns: {list(df_search.columns)[:10]}...")
    
    # Check ILI data
    df_ili = create_date_index(df_ili, 'YEAR', 'WEEK')
    ili_data = df_ili['% WEIGHTED ILI']
    print(f"\nILI data:")
    print(f"  Min: {ili_data.min()}")
    print(f"  Max: {ili_data.max()}")
    print(f"  Mean: {ili_data.mean():.2f}")
    print(f"  Std: {ili_data.std():.2f}")
    
    # Check NREVSS data
    df_nrevss = create_date_index(df_nrevss, 'YEAR', 'WEEK')
    nrevss_data = df_nrevss['flu_pct_positive']
    print(f"\nNREVSS data:")
    print(f"  Min: {nrevss_data.min()}")
    print(f"  Max: {nrevss_data.max()}")
    print(f"  Mean: {nrevss_data.mean():.2f}")
    print(f"  Std: {nrevss_data.std():.2f}")
    
    # Check merged data
    df_search['date'] = pd.to_datetime(df_search['date'])
    df_ili_merged = pd.merge(df_ili, df_search[['date', test_term]], on='date', how='left')
    
    print(f"\nMerged data sample:")
    print(f"  ILI dates: {df_ili_merged['date'].min()} to {df_ili_merged['date'].max()}")
    print(f"  Search dates: {df_search['date'].min()} to {df_search['date'].max()}")
    print(f"  Merged rows: {len(df_ili_merged)}")
    print(f"  Non-null search values: {df_ili_merged[test_term].notna().sum()}")

if __name__ == "__main__":
    debug_data_ranges()
