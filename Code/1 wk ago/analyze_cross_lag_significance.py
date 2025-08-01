import re
import pandas as pd
from collections import defaultdict, Counter

def parse_txt_file(filepath):
    """Parse a TXT summary file and extract significant terms with their p-values"""
    significant_terms = []
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract the dataset type and lag from filename
        if 'ili' in filepath:
            dataset = 'ILI'
        elif 'nrevss' in filepath:
            dataset = 'NREVSS'
        else:
            dataset = 'Unknown'
        
        # Extract lag from filename
        lag_match = re.search(r'max_lag(\d+)', filepath)
        lag = int(lag_match.group(1)) if lag_match else 'Unknown'
        
        # Find the "All significant terms" section
        lines = content.split('\n')
        in_significant_section = False
        
        for line in lines:
            if 'All significant terms (p < 0.05):' in line:
                in_significant_section = True
                continue
            
            if in_significant_section:
                # Stop if we hit an empty line or another section
                if not line.strip() or line.startswith('Results saved to:') or line.startswith('Summary:'):
                    break
                
                # Parse significant term line
                # Format: "  1. term_name: p=0.000123"
                match = re.match(r'\s*\d+\.\s+(.+?):\s+p=([\d.]+)', line)
                if match:
                    term = match.group(1).strip()
                    p_value = float(match.group(2))
                    significant_terms.append({
                        'term': term,
                        'p_value': p_value,
                        'dataset': dataset,
                        'lag': lag
                    })
        
        return significant_terms
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def analyze_cross_lag_significance():
    """Analyze all TXT files to find terms significant across multiple lags and datasets"""
    
    # Define the 8 TXT files
    txt_files = [
        'ShiHaoYang/Results/New/granger_causality_robust_summary_ili_max_lag2.txt',
        'ShiHaoYang/Results/New/granger_causality_robust_summary_ili_max_lag3.txt',
        'ShiHaoYang/Results/New/granger_causality_robust_summary_ili_max_lag4.txt',
        'ShiHaoYang/Results/New/granger_causality_robust_summary_ili_max_lag5.txt',
        'ShiHaoYang/Results/New/granger_causality_robust_summary_nrevss_max_lag2.txt',
        'ShiHaoYang/Results/New/granger_causality_robust_summary_nrevss_max_lag3.txt',
        'ShiHaoYang/Results/New/granger_causality_robust_summary_nrevss_max_lag4.txt',
        'ShiHaoYang/Results/New/granger_causality_robust_summary_nrevss_max_lag5.txt'
    ]
    
    # Parse all files
    all_terms = []
    for filepath in txt_files:
        terms = parse_txt_file(filepath)
        all_terms.extend(terms)
        print(f"Parsed {filepath}: {len(terms)} significant terms")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_terms)
    
    if df.empty:
        print("No significant terms found in any files!")
        return
    
    print(f"\nTotal significant terms found: {len(df)}")
    print(f"Unique terms: {df['term'].nunique()}")
    
    # Count how many times each term appears across all analyses
    term_counts = df['term'].value_counts()
    
    # Calculate average p-value for each term
    term_avg_p = df.groupby('term')['p_value'].mean().sort_values()
    
    # Create a comprehensive summary
    summary_data = []
    for term in term_counts.index:
        count = term_counts[term]
        avg_p = term_avg_p[term]
        min_p = df[df['term'] == term]['p_value'].min()
        max_p = df[df['term'] == term]['p_value'].max()
        
        # Get datasets and lags where this term appears
        term_data = df[df['term'] == term]
        datasets = sorted(term_data['dataset'].unique())
        lags = sorted(term_data['lag'].unique())
        
        summary_data.append({
            'term': term,
            'appearance_count': count,
            'max_possible_appearances': 8,  # 4 lags × 2 datasets
            'appearance_rate': count / 8,
            'avg_p_value': avg_p,
            'min_p_value': min_p,
            'max_p_value': max_p,
            'datasets': ', '.join(datasets),
            'lags': ', '.join(map(str, lags))
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['appearance_count', 'avg_p_value'], ascending=[False, True])
    
    # Save comprehensive results
    summary_df.to_csv('ShiHaoYang/Results/New/cross_lag_significance_analysis.csv', index=False)
    
    # Print top results
    print("\n" + "="*80)
    print("CROSS-LAG SIGNIFICANCE ANALYSIS")
    print("="*80)
    
    print(f"\nTerms appearing in ALL 8 analyses (100% consistency):")
    perfect_terms = summary_df[summary_df['appearance_count'] == 8]
    if not perfect_terms.empty:
        for _, row in perfect_terms.iterrows():
            print(f"  • {row['term']}")
            print(f"    Avg p-value: {row['avg_p_value']:.6f}, Min: {row['min_p_value']:.6f}, Max: {row['max_p_value']:.6f}")
    else:
        print("  None found")
    
    print(f"\nTerms appearing in 7/8 analyses (87.5% consistency):")
    high_consistency = summary_df[summary_df['appearance_count'] == 7]
    if not high_consistency.empty:
        for _, row in high_consistency.iterrows():
            print(f"  • {row['term']}")
            print(f"    Avg p-value: {row['avg_p_value']:.6f}, Min: {row['min_p_value']:.6f}, Max: {row['max_p_value']:.6f}")
    else:
        print("  None found")
    
    print(f"\nTerms appearing in 6/8 analyses (75% consistency):")
    good_consistency = summary_df[summary_df['appearance_count'] == 6]
    if not good_consistency.empty:
        for _, row in good_consistency.iterrows():
            print(f"  • {row['term']}")
            print(f"    Avg p-value: {row['avg_p_value']:.6f}, Min: {row['min_p_value']:.6f}, Max: {row['max_p_value']:.6f}")
    else:
        print("  None found")
    
    print(f"\nTop 10 terms by average p-value (most significant):")
    top_by_p = summary_df[summary_df['appearance_count'] >= 3].head(10)
    for _, row in top_by_p.iterrows():
        print(f"  • {row['term']}")
        print(f"    Appearances: {row['appearance_count']}/8, Avg p-value: {row['avg_p_value']:.6f}")
    
    # Dataset-specific analysis
    print(f"\n" + "="*80)
    print("DATASET-SPECIFIC ANALYSIS")
    print("="*80)
    
    for dataset in ['ILI', 'NREVSS']:
        dataset_df = df[df['dataset'] == dataset]
        dataset_terms = dataset_df['term'].value_counts()
        
        print(f"\n{dataset} Dataset - Terms appearing in all 4 lags:")
        perfect_dataset_terms = dataset_terms[dataset_terms == 4]
        if not perfect_dataset_terms.empty:
            for term, count in perfect_dataset_terms.items():
                avg_p = dataset_df[dataset_df['term'] == term]['p_value'].mean()
                print(f"  • {term} (avg p-value: {avg_p:.6f})")
        else:
            print("  None found")
    
    # Save detailed breakdown
    with open('ShiHaoYang/Results/New/cross_lag_significance_summary.txt', 'w') as f:
        f.write("CROSS-LAG SIGNIFICANCE ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total analyses: 8 (4 lags × 2 datasets)\n")
        f.write(f"Total significant terms found: {len(df)}\n")
        f.write(f"Unique terms: {df['term'].nunique()}\n\n")
        
        f.write("TERMS BY CONSISTENCY LEVEL:\n")
        f.write("-" * 30 + "\n")
        
        for count in range(8, 0, -1):
            terms_at_level = summary_df[summary_df['appearance_count'] == count]
            if not terms_at_level.empty:
                f.write(f"\n{count}/8 analyses ({count/8*100:.1f}% consistency):\n")
                for _, row in terms_at_level.iterrows():
                    f.write(f"  • {row['term']} (avg p-value: {row['avg_p_value']:.6f})\n")
        
        f.write(f"\n\nDetailed results saved to: cross_lag_significance_analysis.csv\n")
    
    print(f"\n" + "="*80)
    print("RESULTS SAVED:")
    print(f"• cross_lag_significance_analysis.csv - Complete analysis")
    print(f"• cross_lag_significance_summary.txt - Summary report")
    print("="*80)

if __name__ == "__main__":
    analyze_cross_lag_significance() 