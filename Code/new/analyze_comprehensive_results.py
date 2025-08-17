#!/usr/bin/env python3
"""
Script to analyze comprehensive significant terms analysis results
and generate 18 separate bar graphs for individual terms:
- 2 datasets (ILI, NREVSS) × 3 categories (Bonferroni, FDR, Uncorrected) × 3 metrics (count, mean p-value, median p-value)
- Each graph shows individual terms on the x-axis
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_comprehensive_analysis(filepath):
    """
    Parse the comprehensive analysis file and extract data for ILI and NREVSS sections.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into sections
    sections = content.split('================================================================================')
    
    # Find ILI and NREVSS specific sections
    ili_section = None
    nrevss_section = None
    
    for section in sections:
        if '=== ILI-SPECIFIC ANALYSIS' in section:
            ili_section = section
        elif '=== NREVSS-SPECIFIC ANALYSIS' in section:
            nrevss_section = section
    
    # Parse each section
    ili_data = parse_dataset_section(ili_section, 'ILI') if ili_section else {}
    nrevss_data = parse_dataset_section(nrevss_section, 'NREVSS') if nrevss_section else {}
    
    return ili_data, nrevss_data

def parse_dataset_section(section, dataset_name):
    """
    Parse a dataset-specific section and extract data for each category.
    """
    data = {
        'Bonferroni': [],
        'FDR': [],
        'Uncorrected': []
    }
    
    lines = section.split('\n')
    current_category = None
    
    for line in lines:
        line = line.strip()
        
        # Detect category headers
        if 'Bonferroni significant terms:' in line:
            current_category = 'Bonferroni'
        elif 'FDR significant terms (not Bonferroni):' in line:
            current_category = 'FDR'
        elif 'Uncorrected significant terms (not Bonferroni or FDR):' in line:
            current_category = 'Uncorrected'
        
        # Parse data lines (tab-separated)
        elif current_category and '\t' in line and not line.startswith('Term\t'):
            parts = line.split('\t')
            if len(parts) >= 4:
                term = parts[0]
                count_str = parts[1]  # e.g., "4/5"
                mean_pval = float(parts[2])
                median_pval = float(parts[3])
                
                # Extract count from "X/5" format
                count_match = re.match(r'(\d+)/5', count_str)
                if count_match:
                    count = int(count_match.group(1))
                    
                    data[current_category].append({
                        'term': term,
                        'count': count,
                        'mean_pval': mean_pval,
                        'median_pval': median_pval
                    })
    
    return data

def create_individual_term_bar_graphs(ili_data, nrevss_data, output_dir="ShiHaoYang/Results/new"):
    """
    Create 18 separate bar graphs for individual terms:
    - 2 datasets × 3 categories × 3 metrics = 18 graphs
    - Each graph shows individual terms on the x-axis
    """
    datasets = {
        'ILI': ili_data,
        'NREVSS': nrevss_data
    }
    
    categories = ['Bonferroni', 'FDR', 'Uncorrected']
    metrics = ['count', 'mean_pval', 'median_pval']
    metric_names = ['Count/5', 'Mean P-value', 'Median P-value']
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create individual bar graphs for each dataset, category, and metric
    for dataset_name, dataset_data in datasets.items():
        for category in categories:
            for metric, metric_name in zip(metrics, metric_names):
                # Get data for this specific combination
                if category in dataset_data and dataset_data[category]:
                    data = dataset_data[category]
                    
                    # Sort data by the metric value (descending for count, ascending for p-values)
                    if metric == 'count':
                        data = sorted(data, key=lambda x: x[metric], reverse=True)
                    else:
                        data = sorted(data, key=lambda x: x[metric])
                    
                    # Extract terms and values
                    terms = [item['term'] for item in data]
                    values = [item[metric] for item in data]
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(max(12, len(terms) * 0.4), 8))
                    
                    # Create bars
                    x_pos = np.arange(len(terms))
                    colors = ['#FF6B6B' if metric == 'count' else '#4ECDC4' if 'mean' in metric else '#45B7D1']
                    
                    bars = ax.bar(x_pos, values, color=colors[0], alpha=0.7, edgecolor='black', linewidth=1)
                    
                    # Customize the plot
                    ax.set_title(f'{dataset_name} - {category} - {metric_name}', 
                               fontweight='bold', fontsize=14, pad=20)
                    ax.set_ylabel(metric_name, fontsize=12)
                    ax.set_xlabel('Terms', fontsize=12)
                    
                    # Set x-axis labels (terms)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(terms, rotation=45, ha='right', fontsize=8)
                    
                    # Add grid
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on bars
                    for i, (bar, value) in enumerate(zip(bars, values)):
                        height = bar.get_height()
                        if metric == 'count':
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{value}', ha='center', va='bottom', fontsize=8)
                        else:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                                   f'{value:.4f}', ha='center', va='bottom', fontsize=8)
                    
                    # Adjust layout to prevent label cutoff
                    plt.tight_layout()
                    
                    # Save the plot
                    filename = f'{dataset_name.lower()}_{category.lower()}_{metric}_bar_graph.png'
                    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Created: {filename} ({len(terms)} terms)")
                else:
                    # Create empty plot for categories with no data
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, f'No {category} significant terms found for {dataset_name}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{dataset_name} - {category} - {metric_name}', 
                               fontweight='bold', fontsize=14)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    
                    filename = f'{dataset_name.lower()}_{category.lower()}_{metric}_bar_graph.png'
                    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Created: {filename} (no data)")

def create_summary_table(ili_data, nrevss_data, output_dir):
    """
    Create a summary table with statistics for each dataset and category.
    """
    datasets = {
        'ILI': ili_data,
        'NREVSS': nrevss_data
    }
    
    categories = ['Bonferroni', 'FDR', 'Uncorrected']
    metrics = ['count', 'mean_pval', 'median_pval']
    
    summary_data = []
    
    for dataset_name, dataset_data in datasets.items():
        for category in categories:
            if category in dataset_data and dataset_data[category]:
                data = dataset_data[category]
                summary_data.append({
                    'Dataset': dataset_name,
                    'Category': category,
                    'N_Terms': len(data),
                    'Mean_Count': np.mean([item['count'] for item in data]),
                    'Std_Count': np.std([item['count'] for item in data]),
                    'Mean_Pvalue': np.mean([item['mean_pval'] for item in data]),
                    'Std_Pvalue': np.std([item['mean_pval'] for item in data]),
                    'Median_Pvalue': np.mean([item['median_pval'] for item in data]),
                    'Std_Median_Pvalue': np.std([item['median_pval'] for item in data])
                })
            else:
                summary_data.append({
                    'Dataset': dataset_name,
                    'Category': category,
                    'N_Terms': 0,
                    'Mean_Count': 0,
                    'Std_Count': 0,
                    'Mean_Pvalue': 0,
                    'Std_Pvalue': 0,
                    'Median_Pvalue': 0,
                    'Std_Median_Pvalue': 0
                })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_file = f'{output_dir}/comprehensive_analysis_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    
    print(f"\nSummary table saved to: {summary_file}")
    print("\nSummary Statistics:")
    print(df_summary.to_string(index=False))

def main():
    """
    Main function to run the analysis.
    """
    # File paths
    input_file = "ShiHaoYang/Results/new/significant_terms_comprehensive_analysis.txt"
    output_dir = "ShiHaoYang/Results/new"
    
    print("Parsing comprehensive analysis file...")
    ili_data, nrevss_data = parse_comprehensive_analysis(input_file)
    
    print(f"ILI data parsed: {sum(len(data) for data in ili_data.values())} total terms")
    print(f"NREVSS data parsed: {sum(len(data) for data in nrevss_data.values())} total terms")
    
    print("\nCreating 18 individual term bar graphs...")
    create_individual_term_bar_graphs(ili_data, nrevss_data, output_dir)
    
    # Create summary statistics table
    create_summary_table(ili_data, nrevss_data, output_dir)
    
    print(f"\nAnalysis complete! 18 graphs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("ILI graphs:")
    print("- ili_bonferroni_count_bar_graph.png")
    print("- ili_bonferroni_mean_pval_bar_graph.png")
    print("- ili_bonferroni_median_pval_bar_graph.png")
    print("- ili_fdr_count_bar_graph.png")
    print("- ili_fdr_mean_pval_bar_graph.png")
    print("- ili_fdr_median_pval_bar_graph.png")
    print("- ili_uncorrected_count_bar_graph.png")
    print("- ili_uncorrected_mean_pval_bar_graph.png")
    print("- ili_uncorrected_median_pval_bar_graph.png")
    print("\nNREVSS graphs:")
    print("- nrevss_bonferroni_count_bar_graph.png")
    print("- nrevss_bonferroni_mean_pval_bar_graph.png")
    print("- nrevss_bonferroni_median_pval_bar_graph.png")
    print("- nrevss_fdr_count_bar_graph.png")
    print("- nrevss_fdr_mean_pval_bar_graph.png")
    print("- nrevss_fdr_median_pval_bar_graph.png")
    print("- nrevss_uncorrected_count_bar_graph.png")
    print("- nrevss_uncorrected_mean_pval_bar_graph.png")
    print("- nrevss_uncorrected_median_pval_bar_graph.png")
    print("\nSummary:")
    print("- comprehensive_analysis_summary.csv")

if __name__ == "__main__":
    main()
