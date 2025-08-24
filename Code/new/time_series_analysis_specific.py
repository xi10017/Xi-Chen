import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
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
        elif line and current_term and '_lag_' in line:
            # Parse: term_lag_X p_value test_identifier
            # Handle terms with spaces by finding the last two parts
            parts = line.split(' ')
            if len(parts) >= 3:
                # The last two parts are p_value and test_info
                p_value = float(parts[-2])
                test_info = parts[-1]  # e.g., "nrevss_maxlag_2"
                
                # Everything before the last two parts is the lag_info
                lag_info = ' '.join(parts[:-2])  # e.g., "(TOPIC)09gh4jl + best flu medicine + best medicine_c82ee129_lag_1"
                
                # Extract lag number from the end
                lag_match = re.search(r'_lag_(\d+)$', lag_info)
                lag = int(lag_match.group(1)) if lag_match else None
                
                # Extract test type and maxlag
                test_match = re.search(r'(\w+)_maxlag_(\d+)', test_info)
                test_type = test_match.group(1) if test_match else None
                maxlag = int(test_match.group(2)) if test_match else None
                
                if lag is not None and test_type is not None and maxlag is not None:
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
    """Load ILI and NREVSS data"""
    # Load ILI data
    df_ili = pd.read_csv("ShiHaoYang/Data/ILINet_all.csv", skiprows=1)
    
    # Load NREVSS data
    df_pub = pd.read_csv("ShiHaoYang/Data/ICL_NREVSS_Public_Health_Labs_all.csv", skiprows=1)
    df_combined = pd.read_csv("ShiHaoYang/Data/ICL_NREVSS_Combined_prior_to_2015_16.csv", skiprows=1)
    
    # Create percent positive columns for NREVSS
    flu_cols_pub = ['A (2009 H1N1)', 'A (H3)', 'A (Subtyping not Performed)', 'B', 'BVic', 'BYam', 'H3N2v', 'A (H5)']
    df_pub['flu_total_positive'] = df_pub[flu_cols_pub].sum(axis=1)
    df_pub['flu_pct_positive'] = df_pub['flu_total_positive'] / df_pub['TOTAL SPECIMENS']

    flu_cols_combined = ['A (2009 H1N1)', 'A (H1)', 'A (H3)', 'A (Subtyping not Performed)', 'A (Unable to Subtype)', 'B', 'H3N2v', 'A (H5)']
    df_combined['flu_total_positive'] = df_combined[flu_cols_combined].sum(axis=1)
    df_combined['flu_pct_positive'] = df_combined['flu_total_positive'] / df_combined['TOTAL SPECIMENS']

    # Standardize columns and concatenate
    common_cols = ['REGION TYPE', 'REGION', 'YEAR', 'WEEK', 'TOTAL SPECIMENS', 'flu_total_positive', 'flu_pct_positive']
    df_pub = df_pub[common_cols]
    df_combined = df_combined[common_cols]
    df_nrevss = pd.concat([df_combined, df_pub], ignore_index=True)
    
    # Load search trends data
    df_search = pd.read_csv("ShiHaoYang/Data/flu_trends_regression_dataset.csv")
    
    return df_ili, df_nrevss, df_search

def create_ili_plot(term, df_ili_merged, significant_lags, term_significant, term_dir):
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
        df_ili_merged[f'{term}_lag{lag}'] = df_ili_merged[term].shift(lag)
        
        # Plot data
        x = df_ili_merged['date']
        
        # Plot search term (original) - left y-axis
        ax.plot(x, df_ili_merged[term], label=f'{term} (original)', color='blue', alpha=0.7, linewidth=1)
        
        # Plot lagged search term (highlight if significant) - left y-axis
        if lag in significant_lags:
            ax.plot(x, df_ili_merged[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='red', alpha=0.8, linewidth=2)
        else:
            ax.plot(x, df_ili_merged[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='blue', alpha=0.7, linewidth=1)
        
        # Plot ILI data - right y-axis
        ax2 = ax.twinx()
        ax2.plot(x, df_ili_merged['% WEIGHTED ILI'], label='ILI %', color='green', alpha=0.8, linewidth=2)
        
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

def create_nrevss_plot(term, df_nrevss_merged, significant_lags, term_significant, term_dir):
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
    
    for lag in range(1, 6):
        ax = axes[lag-1]
        
        # Create lagged term data
        df_nrevss_merged[f'{term}_lag{lag}'] = df_nrevss_merged[term].shift(lag)
        
        # Plot data
        x = df_nrevss_merged['date']
        
        # Plot search term (original) - left y-axis
        ax.plot(x, df_nrevss_merged[term], label=f'{term} (original)', color='blue', alpha=0.7, linewidth=1)
        
        # Plot lagged search term (highlight if significant) - left y-axis
        if lag in significant_lags:
            ax.plot(x, df_nrevss_merged[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='red', alpha=0.8, linewidth=2)
        else:
            ax.plot(x, df_nrevss_merged[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='blue', alpha=0.7, linewidth=1)
        
        # Plot NREVSS data - right y-axis
        ax2 = ax.twinx()
        ax2.plot(x, df_nrevss_merged['flu_pct_positive'], label='NREVSS % Positive', color='orange', alpha=0.8, linewidth=2)
        
        # Customize plot
        ax.set_title(f'Lag {lag}', fontweight='bold')
        ax.set_ylabel(f'{term} Search Volume', color='blue')
        ax2.set_ylabel('NREVSS Activity (%)', color='orange')
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

def create_combined_plot(term, df_ili_merged, df_nrevss_merged, term_significant, term_dir):
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
    
    for lag in range(1, 6):
        ax = axes[lag-1]
        
        # Create lagged term data for both datasets
        df_ili_merged[f'{term}_lag{lag}'] = df_ili_merged[term].shift(lag)
        df_nrevss_merged[f'{term}_lag{lag}'] = df_nrevss_merged[term].shift(lag)
        
        # Plot search term data (use ILI dates as reference) - left y-axis
        x = df_ili_merged['date']
        
        # Plot search term (original) - left y-axis
        ax.plot(x, df_ili_merged[term], label=f'{term} (original)', color='blue', alpha=0.7, linewidth=1)
        
        # Plot lagged search term (highlight if significant in either test) - left y-axis
        is_significant = (lag in ili_significant_lags) or (lag in nrevss_significant_lags)
        if is_significant:
            ax.plot(x, df_ili_merged[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='red', alpha=0.8, linewidth=2)
        else:
            ax.plot(x, df_ili_merged[f'{term}_lag{lag}'], label=f'{term} (lag {lag})', color='blue', alpha=0.7, linewidth=1)
        
        # Plot both flu datasets - right y-axis
        ax2 = ax.twinx()
        
        # Plot ILI data
        ax2.plot(x, df_ili_merged['% WEIGHTED ILI'], label='ILI %', color='green', alpha=0.8, linewidth=2)
        
        # Plot NREVSS data (align dates)
        # Handle duplicate dates by taking the mean
        nrevss_data = df_nrevss_merged.groupby('date')['flu_pct_positive'].mean()
        nrevss_aligned = nrevss_data.reindex(x).ffill()
        ax2.plot(x, nrevss_aligned, label='NREVSS %', color='orange', alpha=0.8, linewidth=2)
        
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
    
    plt.tight_layout()
    filename = os.path.join(term_dir, f"{term.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')}_combined_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def create_term_plots(term, df_ili, df_nrevss, df_search, significant_combinations, output_dir, 
                     start_date=None, end_date=None):
    """Create 3 plots for a single term: ILI, NREVSS, and combined"""
    
    # Filter significant combinations for this term
    term_significant = [s for s in significant_combinations if s['term'] == term]
    
    if not term_significant:
        print(f"No significant combinations found for term: {term}")
        return
    
    # Check if term exists in search data
    if term not in df_search.columns:
        print(f"Warning: Term '{term}' not found in search dataset")
        return
    
    # Create date indices
    df_ili['date'] = pd.to_datetime(df_ili['YEAR'].astype(str) + '-W' + df_ili['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
    df_nrevss['date'] = pd.to_datetime(df_nrevss['YEAR'].astype(str) + '-W' + df_nrevss['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
    df_search['date'] = pd.to_datetime(df_search['date'])
    
    # Sort by date
    df_ili = df_ili.sort_values('date')
    df_nrevss = df_nrevss.sort_values('date')
    df_search = df_search.sort_values('date')
    
    # Ensure we only use overlapping date ranges
    ili_start = df_ili['date'].min()
    ili_end = df_ili['date'].max()
    search_start = df_search['date'].min()
    search_end = df_search['date'].max()
    
    # Use the overlapping range
    start_date_common = max(ili_start, search_start)
    end_date_common = min(ili_end, search_end)
    
    # Filter data to overlapping range
    df_ili_filtered = df_ili[(df_ili['date'] >= start_date_common) & (df_ili['date'] <= end_date_common)]
    df_search_filtered = df_search[(df_search['date'] >= start_date_common) & (df_search['date'] <= end_date_common)]
    
    # Merge search data with flu data
    df_ili_merged = pd.merge(df_ili_filtered, df_search_filtered[['date', term]], on='date', how='left')
    df_nrevss_merged = pd.merge(df_nrevss, df_search_filtered[['date', term]], on='date', how='left')
    
    # Filter by date range if specified
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df_ili_merged = df_ili_merged[df_ili_merged['date'] >= start_dt]
        df_nrevss_merged = df_nrevss_merged[df_nrevss_merged['date'] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df_ili_merged = df_ili_merged[df_ili_merged['date'] <= end_dt]
        df_nrevss_merged = df_nrevss_merged[df_nrevss_merged['date'] <= end_dt]
    
    # Create output directory for this term
    term_dir = os.path.join(output_dir, term.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', ''))
    os.makedirs(term_dir, exist_ok=True)
    
    # Get significant lags for each test type
    ili_significant_lags = [s['lag'] for s in term_significant if s['test_type'] == 'ili']
    nrevss_significant_lags = [s['lag'] for s in term_significant if s['test_type'] == 'nrevss']
    
    # Create ILI plot
    create_ili_plot(term, df_ili_merged, ili_significant_lags, term_significant, term_dir)
    
    # Create NREVSS plot
    create_nrevss_plot(term, df_nrevss_merged, nrevss_significant_lags, term_significant, term_dir)
    
    # Create combined plot
    create_combined_plot(term, df_ili_merged, df_nrevss_merged, term_significant, term_dir)

def main(start_date=None, end_date=None):
    """Main function to generate all time series plots"""
    print("Loading Bonferroni significant terms...")
    
    # Parse significant terms
    significant_combinations = parse_bonferroni_summary("ShiHaoYang/Data/bonferroni_significant_terms_summary.txt")
    
    print(f"Found {len(significant_combinations)} significant combinations")
    
    # Get unique terms
    unique_terms = list(set([s['term'] for s in significant_combinations]))
    print(f"Unique terms: {unique_terms}")
    
    print("Loading data...")
    
    # Load data
    df_ili, df_nrevss, df_search = load_data()
    
    # Create output directory
    output_dir = "ShiHaoYang/Results/time_series_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each term
    for i, term in enumerate(unique_terms, 1):
        print(f"\nProcessing term {i}/{len(unique_terms)}: {term}")
        try:
            create_term_plots(term, df_ili, df_nrevss, df_search, significant_combinations, output_dir, start_date, end_date)
        except Exception as e:
            print(f"Error processing term '{term}': {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Total plots created: {len(unique_terms) * 3}")

if __name__ == "__main__":
    # You can specify custom date ranges here
    # main(start_date="2015-01-01", end_date="2020-12-31")
    
    # Or use full period
    main()
