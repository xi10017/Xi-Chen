import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from dataframes import *
import os
from datetime import datetime

def load_significant_terms():
    """Load significant terms from the text file"""
    with open("ShiHaoYang/Data/US_terms_significant.txt", "r") as f:
        terms = [line.strip() for line in f if line.strip()]
    return terms

def create_lagged_plots(term, df_ili, df_nrevss, output_dir):
    """Create lagged plots for a single term comparing with flu data"""
    
    # Check if term exists in the data
    if term not in df_ili.columns:
        print(f"Warning: Term '{term}' not found in ILI dataset")
        return
    
    # Create figure with 5 subplots (one for each lag)
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    fig.suptitle(f'Time Series Analysis: {term} vs Flu Data (Lags 1-5)', fontsize=16, fontweight='bold')
    
    # Create date index for x-axis
    df_ili['date'] = pd.to_datetime(df_ili['YEAR'].astype(str) + '-W' + df_ili['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
    df_nrevss['date'] = pd.to_datetime(df_nrevss['YEAR'].astype(str) + '-W' + df_nrevss['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
    
    # Sort by date
    df_ili = df_ili.sort_values('date')
    df_nrevss = df_nrevss.sort_values('date')
    
    for lag in range(1, 6):
        ax = axes[lag-1]
        
        # Create lagged term data
        df_ili[f'{term}_lag{lag}'] = df_ili[term].shift(lag)
        
        # Plot data
        x = df_ili['date']
        
        # Plot search term (original)
        ax.plot(x, df_ili[term], label=f'{term} (original)', color='blue', alpha=0.7, linewidth=1)
        
        # Plot lagged search term
        ax.plot(x, df_ili[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='red', alpha=0.7, linewidth=1)
        
        # Plot ILI data
        ax2 = ax.twinx()
        ax2.plot(x, df_ili['% WEIGHTED ILI'], label='ILI %', color='green', alpha=0.8, linewidth=2)
        
        # Plot NREVSS data (if available for the same time period)
        nrevss_subset = df_nrevss[df_nrevss['date'].isin(df_ili['date'])]
        if not nrevss_subset.empty:
            ax2.plot(nrevss_subset['date'], nrevss_subset['flu_pct_positive'], 
                    label='NREVSS % Positive', color='orange', alpha=0.8, linewidth=2)
        
        # Customize plot
        ax.set_title(f'Lag {lag}', fontweight='bold')
        ax.set_ylabel(f'{term} Search Volume', color='blue')
        ax2.set_ylabel('Flu Activity (%)', color='green')
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    filename = f"{output_dir}/{term.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_time_series_lags.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

def main():
    """Main function to generate all time series plots"""
    print("Loading data...")
    
    # Load data
    df_ili = get_dataframe_ili()
    df_nrevss = get_dataframe_nrevss()
    
    # Load significant terms
    significant_terms = load_significant_terms()
    
    print(f"Found {len(significant_terms)} significant terms")
    print("Terms:", significant_terms)
    
    # Create output directory
    output_dir = "ShiHaoYang/Results/time_series_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each term
    for i, term in enumerate(significant_terms, 1):
        print(f"\nProcessing term {i}/{len(significant_terms)}: {term}")
        try:
            create_lagged_plots(term, df_ili, df_nrevss, output_dir)
        except Exception as e:
            print(f"Error processing term '{term}': {e}")
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
