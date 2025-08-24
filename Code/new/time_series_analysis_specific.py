import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from dataframes import *
import os
from datetime import datetime
import re

def parse_bonferroni_summary(filepath):
    """Parse the Bonferroni significant terms summary file"""
    significant_combinations = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    current_term = None
    for line in lines:
        line = line.strip()
        if line.endswith(' significant terms:'):
            current_term = line.replace(' significant terms:', '')
        elif line and current_term and not line.startswith('===') and not line.startswith('Total') and not line.startswith('Unique') and not line.startswith('Tests'):
            # Parse: term lag p_value test_identifier
            # Handle terms with spaces by finding the last three parts
            parts = line.split(' ')
            if len(parts) >= 4:
                # The last three parts are lag, p_value, and test_info
                lag = int(parts[-3])
                p_value = float(parts[-2])
                test_info = parts[-1]  # e.g., "nrevss_maxlag_2"
                
                # Everything before the last three parts is the term
                term_name = ' '.join(parts[:-3])
                
                # Extract test type and maxlag
                test_match = re.search(r'(\w+)_maxlag_(\d+)', test_info)
                test_type = test_match.group(1) if test_match else None
                maxlag = int(test_match.group(2)) if test_match else None
                
                if test_type is not None and maxlag is not None:
                    significant_combinations.append({
                        'term': current_term,
                        'lag': lag,
                        'p_value': p_value,
                        'test_type': test_type,
                        'maxlag': maxlag,
                        'test_info': test_info
                    })
    
    return significant_combinations

def load_data():
    """Load ILI and NREVSS data using dataframes module"""
    # Use the dataframes module which already handles merging with search trends
    df_ili = get_dataframe_ili()
    df_nrevss = get_dataframe_nrevss()
    
    return df_ili, df_nrevss

def create_ili_plot(term, df_ili, significant_lags, term_significant, term_dir):
    """Create ILI analysis plot"""
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    
    # Get ILI significant info for title
    ili_info = [s for s in term_significant if s['test_type'] == 'ili']
    if ili_info:
        ili_parts = []
        for s in ili_info:
            ili_parts.append(f"lag {s['lag']}, p={s['p_value']:.6f} in maxlag_{s['maxlag']}")
        ili_title_info = ', '.join(ili_parts)
        fig.suptitle(f'{term} - ILI Analysis (Significant: {ili_title_info})', fontsize=12, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'{term} - ILI Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    for lag in range(1, 6):
        ax = axes[lag-1]
        
        # Create lagged term data
        df_ili[f'{term}_lag{lag}'] = df_ili[term].shift(lag)
        
        # Plot data
        x = df_ili['date']
        
        # Plot search term (original) - left y-axis
        ax.plot(x, df_ili[term], label=f'{term} (original)', color='blue', alpha=0.7, linewidth=1)
        
        # Plot lagged search term (highlight if significant) - left y-axis
        if lag in significant_lags:
            ax.plot(x, df_ili[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='red', alpha=0.8, linewidth=2)
        else:
            ax.plot(x, df_ili[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='blue', alpha=0.7, linewidth=1)
        
        # Plot ILI data - right y-axis
        ax2 = ax.twinx()
        ax2.plot(x, df_ili['% WEIGHTED ILI'], label='ILI %', color='green', alpha=0.8, linewidth=2)
        
        # Customize plot
        ax.set_title(f'Lag {lag}', fontweight='bold')
        ax.set_ylabel(f'{term} Search Volume', color='blue')
        ax2.set_ylabel('ILI Activity (%)', color='green')
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    filename = os.path.join(term_dir, f"{term.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_ili_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def create_nrevss_plot(term, df_nrevss, significant_lags, term_significant, term_dir):
    """Create NREVSS analysis plot"""
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    
    # Get NREVSS significant info for title
    nrevss_info = [s for s in term_significant if s['test_type'] == 'nrevss']
    if nrevss_info:
        nrevss_parts = []
        for s in nrevss_info:
            nrevss_parts.append(f"lag {s['lag']}, p={s['p_value']:.6f} in maxlag_{s['maxlag']}")
        nrevss_title_info = ', '.join(nrevss_parts)
        fig.suptitle(f'{term} - NREVSS Analysis (Significant: {nrevss_title_info})', fontsize=12, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'{term} - NREVSS Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    # Normalize NREVSS data
    nrevss_min = df_nrevss['flu_pct_positive'].min()
    nrevss_max = df_nrevss['flu_pct_positive'].max()
    df_nrevss['flu_pct_positive_normalized'] = (df_nrevss['flu_pct_positive'] - nrevss_min) / (nrevss_max - nrevss_min)
    
    for lag in range(1, 6):
        ax = axes[lag-1]
        
        # Create lagged term data
        df_nrevss[f'{term}_lag{lag}'] = df_nrevss[term].shift(lag)
        
        # Plot data
        x = df_nrevss['date']
        
        # Plot search term (original) - left y-axis
        ax.plot(x, df_nrevss[term], label=f'{term} (original)', color='blue', alpha=0.7, linewidth=1)
        
        # Plot lagged search term (highlight if significant) - left y-axis
        if lag in significant_lags:
            ax.plot(x, df_nrevss[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='red', alpha=0.8, linewidth=2)
        else:
            ax.plot(x, df_nrevss[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='blue', alpha=0.7, linewidth=1)
        
        # Plot normalized NREVSS data - right y-axis
        ax2 = ax.twinx()
        ax2.plot(x, df_nrevss['flu_pct_positive_normalized'], label='NREVSS % Positive (normalized)', color='orange', alpha=0.8, linewidth=2)
        
        # Customize plot
        ax.set_title(f'Lag {lag}', fontweight='bold')
        ax.set_ylabel(f'{term} Search Volume', color='blue')
        ax2.set_ylabel('NREVSS Activity (normalized)', color='orange')
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    filename = os.path.join(term_dir, f"{term.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_nrevss_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def create_combined_plot(term, df_ili, df_nrevss, term_significant, term_dir):
    """Create combined analysis plot"""
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    
    # Create combined title with all significant info
    ili_info = [s for s in term_significant if s['test_type'] == 'ili']
    nrevss_info = [s for s in term_significant if s['test_type'] == 'nrevss']
    
    title_parts = []
    if ili_info:
        ili_parts = []
        for s in ili_info:
            ili_parts.append(f"lag {s['lag']}, p={s['p_value']:.6f} in maxlag_{s['maxlag']}")
        ili_part = f"ILI: {', '.join(ili_parts)}"
        title_parts.append(ili_part)
    if nrevss_info:
        nrevss_parts = []
        for s in nrevss_info:
            nrevss_parts.append(f"lag {s['lag']}, p={s['p_value']:.6f} in maxlag_{s['maxlag']}")
        nrevss_part = f"NREVSS: {', '.join(nrevss_parts)}"
        title_parts.append(nrevss_part)
    
    combined_title = f"{term} - Combined Analysis ({' | '.join(title_parts)})"
    fig.suptitle(combined_title, fontsize=12, fontweight='bold', y=0.98)
    
    # Get significant lags for both test types
    ili_significant_lags = [s['lag'] for s in term_significant if s['test_type'] == 'ili']
    nrevss_significant_lags = [s['lag'] for s in term_significant if s['test_type'] == 'nrevss']
    
    # Normalize both ILI and NREVSS data for combined plot
    ili_min = df_ili['% WEIGHTED ILI'].min()
    ili_max = df_ili['% WEIGHTED ILI'].max()
    df_ili['% WEIGHTED ILI_normalized'] = (df_ili['% WEIGHTED ILI'] - ili_min) / (ili_max - ili_min)
    
    nrevss_min = df_nrevss['flu_pct_positive'].min()
    nrevss_max = df_nrevss['flu_pct_positive'].max()
    df_nrevss['flu_pct_positive_normalized'] = (df_nrevss['flu_pct_positive'] - nrevss_min) / (nrevss_max - nrevss_min)
    
    for lag in range(1, 6):
        ax = axes[lag-1]
        
        # Create lagged term data for both datasets
        df_ili[f'{term}_lag{lag}'] = df_ili[term].shift(lag)
        df_nrevss[f'{term}_lag{lag}'] = df_nrevss[term].shift(lag)
        
        # Plot search term data (use ILI dates as reference) - left y-axis
        x = df_ili['date']
        
        # Plot search term (original) - left y-axis
        ax.plot(x, df_ili[term], label=f'{term} (original)', color='blue', alpha=0.7, linewidth=1)
        
        # Plot lagged search term (highlight if significant in either test) - left y-axis
        is_significant = (lag in ili_significant_lags) or (lag in nrevss_significant_lags)
        if is_significant:
            ax.plot(x, df_ili[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='red', alpha=0.8, linewidth=2)
        else:
            ax.plot(x, df_ili[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='blue', alpha=0.7, linewidth=1)
        
        # Plot both normalized flu datasets - right y-axis
        ax2 = ax.twinx()
        
        # Plot normalized ILI data
        ax2.plot(x, df_ili['% WEIGHTED ILI_normalized'], label='ILI % (normalized)', color='green', alpha=0.8, linewidth=2)
        
        # Plot normalized NREVSS data (align dates)
        # Handle duplicate dates by taking the mean
        nrevss_data = df_nrevss.groupby('date')['flu_pct_positive_normalized'].mean()
        nrevss_aligned = nrevss_data.reindex(x).ffill()
        ax2.plot(x, nrevss_aligned, label='NREVSS % (normalized)', color='orange', alpha=0.8, linewidth=2)
        
        # Customize plot
        ax.set_title(f'Lag {lag}', fontweight='bold')
        ax.set_ylabel(f'{term} Search Volume', color='blue')
        ax2.set_ylabel('Flu Activity (normalized)', color='green')
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    filename = os.path.join(term_dir, f"{term.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_combined_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def create_term_plots(term, df_ili, df_nrevss, significant_combinations, output_dir):
    """Create 3 plots for a single term: ILI, NREVSS, and combined"""
    
    # Filter significant combinations for this term
    term_significant = [s for s in significant_combinations if s['term'] == term]
    
    if not term_significant:
        print(f"No significant combinations found for term: {term}")
        return
    
    # Check if term exists in the data
    if term not in df_ili.columns:
        print(f"Warning: Term '{term}' not found in ILI dataset")
        return
    
    # Create date indices
    df_ili['date'] = pd.to_datetime(df_ili['YEAR'].astype(str) + '-W' + df_ili['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
    df_nrevss['date'] = pd.to_datetime(df_nrevss['YEAR'].astype(str) + '-W' + df_nrevss['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
    
    # Sort by date
    df_ili = df_ili.sort_values('date')
    df_nrevss = df_nrevss.sort_values('date')
    
    # Create output directory for this term
    term_dir = os.path.join(output_dir, term.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', ''))
    os.makedirs(term_dir, exist_ok=True)
    
    # Get significant lags for each test type
    ili_significant_lags = [s['lag'] for s in term_significant if s['test_type'] == 'ili']
    nrevss_significant_lags = [s['lag'] for s in term_significant if s['test_type'] == 'nrevss']
    
    # Create ILI plot
    create_ili_plot(term, df_ili, ili_significant_lags, term_significant, term_dir)
    
    # Create NREVSS plot
    create_nrevss_plot(term, df_nrevss, nrevss_significant_lags, term_significant, term_dir)
    
    # Create combined plot
    create_combined_plot(term, df_ili, df_nrevss, term_significant, term_dir)

def main():
    """Main function to generate all time series plots"""
    print("Loading Bonferroni significant terms...")
    
    # Parse significant terms
    significant_combinations = parse_bonferroni_summary("ShiHaoYang/Data/bonferroni_significant_terms_summary_new.txt")
    
    print(f"Found {len(significant_combinations)} significant combinations")
    
    # Get unique terms
    unique_terms = list(set([s['term'] for s in significant_combinations]))
    print(f"Unique terms: {unique_terms}")
    
    print("Loading data...")
    
    # Load data
    df_ili, df_nrevss = load_data()
    
    # Create output directory
    output_dir = "ShiHaoYang/Results/time_series_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each term
    for i, term in enumerate(unique_terms, 1):
        print(f"\nProcessing term {i}/{len(unique_terms)}: {term}")
        try:
            create_term_plots(term, df_ili, df_nrevss, significant_combinations, output_dir)
        except Exception as e:
            print(f"Error processing term '{term}': {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Total plots created: {len(unique_terms) * 3}")

if __name__ == "__main__":
    main()
