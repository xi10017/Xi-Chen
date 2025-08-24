#!/usr/bin/env python3
"""
Test script to verify the plot fixes work for a single term
"""

import sys
import os
sys.path.append('.')

from time_series_analysis_specific import *

def test_single_term():
    """Test plotting for a single term"""
    print("Loading Bonferroni significant terms...")
    
    # Parse significant terms
    significant_combinations = parse_bonferroni_summary("ShiHaoYang/Data/bonferroni_significant_terms_summary.txt")
    
    print(f"Found {len(significant_combinations)} significant combinations")
    
    # Get unique terms
    unique_terms = list(set([s['term'] for s in significant_combinations]))
    print(f"Unique terms: {unique_terms}")
    
    # Test with 'h1n1' term
    test_term = 'h1n1'
    if test_term not in unique_terms:
        print(f"Test term '{test_term}' not found. Using first available term.")
        test_term = unique_terms[0]
    
    print(f"Testing with term: {test_term}")
    
    print("Loading data...")
    
    # Load data
    df_ili, df_nrevss, df_search = load_data()
    
    # Create output directory
    output_dir = "ShiHaoYang/Results/test_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for test term
    try:
        create_term_plots(test_term, df_ili, df_nrevss, df_search, significant_combinations, output_dir)
        print(f"Successfully created plots for term: {test_term}")
        print(f"Check output directory: {output_dir}")
    except Exception as e:
        print(f"Error processing term '{test_term}': {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_term()
