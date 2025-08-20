import os
import re
import numpy as np
from collections import defaultdict, Counter

# Folder containing significant terms txt files
ili_folder = "ShiHaoYang/Results/new/ili"
nrevss_folder = "ShiHaoYang/Results/new/nrevss"
output_file = "ShiHaoYang/Results/new/significant_terms_comprehensive_analysis.txt"

# Regex to match lines with significant terms
term_line_pattern = re.compile(r"^(.+)\t([0-9.]+)\t(.+)$")

# Data structures to store term information
term_data = defaultdict(lambda: {
    'bonferroni_count': 0,
    'fdr_count': 0, 
    'uncorrected_count': 0,
    'bonferroni_pvalues': [],
    'fdr_pvalues': [],
    'uncorrected_pvalues': [],
    'files_found_in': [],
    'ili_bonferroni_count': 0,
    'ili_fdr_count': 0,
    'ili_uncorrected_count': 0,
    'ili_bonferroni_pvalues': [],
    'ili_fdr_pvalues': [],
    'ili_uncorrected_pvalues': [],
    'ili_files_found_in': [],
    'nrevss_bonferroni_count': 0,
    'nrevss_fdr_count': 0,
    'nrevss_uncorrected_count': 0,
    'nrevss_bonferroni_pvalues': [],
    'nrevss_fdr_pvalues': [],
    'nrevss_uncorrected_pvalues': [],
    'nrevss_files_found_in': []
})

def parse_file(filepath, filename, dataset_type):
    """Parse a significant terms file and extract term information"""
    with open(filepath, "r") as f:
        content = f.read()
        
    # Find the section with significant terms
    if "=== ALL SIGNIFICANT TERMS" in content:
        # Extract lines after the header
        lines = content.split("=== ALL SIGNIFICANT TERMS")[1].split("\n")
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith("===") and not line.startswith("Term\t"):
                match = term_line_pattern.match(line)
                if match:
                    term = match.group(1)
                    p_value = float(match.group(2))
                    significance = match.group(3)
                    
                    # Add to term data
                    term_data[term]['files_found_in'].append(filename)
                    
                    # Add to dataset-specific data
                    if dataset_type == 'ili':
                        term_data[term]['ili_files_found_in'].append(filename)
                    else:  # nrevss
                        term_data[term]['nrevss_files_found_in'].append(filename)
                    
                    # Count by significance type (overall)
                    if significance == "Bonferroni":
                        term_data[term]['bonferroni_count'] += 1
                        term_data[term]['bonferroni_pvalues'].append(p_value)
                        # Bonferroni significant terms are also FDR and Uncorrected significant
                        term_data[term]['fdr_count'] += 1
                        term_data[term]['fdr_pvalues'].append(p_value)
                        term_data[term]['uncorrected_count'] += 1
                        term_data[term]['uncorrected_pvalues'].append(p_value)
                    elif significance == "FDR":
                        term_data[term]['fdr_count'] += 1
                        term_data[term]['fdr_pvalues'].append(p_value)
                        # FDR significant terms are also Uncorrected significant
                        term_data[term]['uncorrected_count'] += 1
                        term_data[term]['uncorrected_pvalues'].append(p_value)
                    elif significance == "Uncorrected":
                        term_data[term]['uncorrected_count'] += 1
                        term_data[term]['uncorrected_pvalues'].append(p_value)
                    
                    # Count by significance type (dataset-specific)
                    if dataset_type == 'ili':
                        if significance == "Bonferroni":
                            term_data[term]['ili_bonferroni_count'] += 1
                            term_data[term]['ili_bonferroni_pvalues'].append(p_value)
                            term_data[term]['ili_fdr_count'] += 1
                            term_data[term]['ili_fdr_pvalues'].append(p_value)
                            term_data[term]['ili_uncorrected_count'] += 1
                            term_data[term]['ili_uncorrected_pvalues'].append(p_value)
                        elif significance == "FDR":
                            term_data[term]['ili_fdr_count'] += 1
                            term_data[term]['ili_fdr_pvalues'].append(p_value)
                            term_data[term]['ili_uncorrected_count'] += 1
                            term_data[term]['ili_uncorrected_pvalues'].append(p_value)
                        elif significance == "Uncorrected":
                            term_data[term]['ili_uncorrected_count'] += 1
                            term_data[term]['ili_uncorrected_pvalues'].append(p_value)
                    else:  # nrevss
                        if significance == "Bonferroni":
                            term_data[term]['nrevss_bonferroni_count'] += 1
                            term_data[term]['nrevss_bonferroni_pvalues'].append(p_value)
                            term_data[term]['nrevss_fdr_count'] += 1
                            term_data[term]['nrevss_fdr_pvalues'].append(p_value)
                            term_data[term]['nrevss_uncorrected_count'] += 1
                            term_data[term]['nrevss_uncorrected_pvalues'].append(p_value)
                        elif significance == "FDR":
                            term_data[term]['nrevss_fdr_count'] += 1
                            term_data[term]['nrevss_fdr_pvalues'].append(p_value)
                            term_data[term]['nrevss_uncorrected_count'] += 1
                            term_data[term]['nrevss_uncorrected_pvalues'].append(p_value)
                        elif significance == "Uncorrected":
                            term_data[term]['nrevss_uncorrected_count'] += 1
                            term_data[term]['nrevss_uncorrected_pvalues'].append(p_value)

def calculate_statistics(pvalues):
    """Calculate mean and median for a list of p-values"""
    if not pvalues:
        return None, None
    return np.mean(pvalues), np.median(pvalues)

# Process all files
print("Processing ILI files...")
for filename in os.listdir(ili_folder):
    if filename.startswith("comprehensive_granger_individual_significant_terms_ili_lag") and filename.endswith(".txt"):
        filepath = os.path.join(ili_folder, filename)
        parse_file(filepath, filename, 'ili')

print("Processing NREVSS files...")
for filename in os.listdir(nrevss_folder):
    if filename.startswith("granger_significant_terms_nrevss_lag") and filename.endswith(".txt"):
        filepath = os.path.join(nrevss_folder, filename)
        parse_file(filepath, filename, 'nrevss')

# Write comprehensive analysis
with open(output_file, "w") as f:
    f.write("=== COMPREHENSIVE SIGNIFICANT TERMS ANALYSIS ===\n")
    f.write(f"Total files analyzed: 10 (5 ILI + 5 NREVSS)\n")
    f.write(f"Total unique terms found: {len(term_data)}\n\n")
    
    # Combined analysis (existing)
    f.write("=== COMBINED ANALYSIS (ILI + NREVSS) ===\n")
    
    # Bonferroni significant terms
    bonferroni_terms = {term: data for term, data in term_data.items() if data['bonferroni_count'] > 0}
    f.write(f"Total Bonferroni significant terms: {len(bonferroni_terms)}\n\n")
    
    if bonferroni_terms:
        f.write("Term\tCount/10\tMean_p_value\tMedian_p_value\tFiles\n")
        for term, data in sorted(bonferroni_terms.items(), key=lambda x: x[1]['bonferroni_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['bonferroni_pvalues'])
            f.write(f"{term}\t{data['bonferroni_count']}/10\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['files_found_in'])}\n")
    else:
        f.write("No terms were Bonferroni significant in any file.\n")
    
    f.write("\n" + "="*80 + "\n\n")
    
    # FDR significant terms (including those already in Bonferroni)
    fdr_terms = {term: data for term, data in term_data.items() 
                 if data['fdr_count'] > 0}
    f.write(f"Total FDR significant terms: {len(fdr_terms)}\n\n")
    
    if fdr_terms:
        f.write("Term\tCount/10\tMean_p_value\tMedian_p_value\tFiles\n")
        for term, data in sorted(fdr_terms.items(), key=lambda x: x[1]['fdr_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['fdr_pvalues'])
            f.write(f"{term}\t{data['fdr_count']}/10\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['files_found_in'])}\n")
    else:
        f.write("No terms were FDR significant in any file.\n")
    
    f.write("\n" + "="*80 + "\n\n")
    
    # Uncorrected significant terms (including those already in Bonferroni or FDR)
    uncorrected_terms = {term: data for term, data in term_data.items() 
                        if data['uncorrected_count'] > 0}
    f.write(f"Total Uncorrected significant terms: {len(uncorrected_terms)}\n\n")
    
    if uncorrected_terms:
        f.write("Term\tCount/10\tMean_p_value\tMedian_p_value\tFiles\n")
        for term, data in sorted(uncorrected_terms.items(), key=lambda x: x[1]['uncorrected_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['uncorrected_pvalues'])
            f.write(f"{term}\t{data['uncorrected_count']}/10\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['files_found_in'])}\n")
    else:
        f.write("No terms were Uncorrected significant in any file.\n")
    
    f.write("\n" + "="*80 + "\n\n")
    
    # ILI-specific analysis
    f.write("=== ILI-SPECIFIC ANALYSIS (5 files) ===\n")
    
    # ILI Bonferroni significant terms
    ili_bonferroni_terms = {term: data for term, data in term_data.items() if data['ili_bonferroni_count'] > 0}
    f.write(f"ILI Bonferroni significant terms: {len(ili_bonferroni_terms)}\n\n")
    
    if ili_bonferroni_terms:
        f.write("Term\tCount/5\tMean_p_value\tMedian_p_value\tILI_Files\n")
        for term, data in sorted(ili_bonferroni_terms.items(), key=lambda x: x[1]['ili_bonferroni_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['ili_bonferroni_pvalues'])
            f.write(f"{term}\t{data['ili_bonferroni_count']}/5\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['ili_files_found_in'])}\n")
    else:
        f.write("No terms were Bonferroni significant in ILI files.\n")
    
    f.write("\n")
    
    # ILI FDR significant terms (including those already in Bonferroni)
    ili_fdr_terms = {term: data for term, data in term_data.items() 
                     if data['ili_fdr_count'] > 0}
    f.write(f"ILI FDR significant terms: {len(ili_fdr_terms)}\n\n")
    
    if ili_fdr_terms:
        f.write("Term\tCount/5\tMean_p_value\tMedian_p_value\tILI_Files\n")
        for term, data in sorted(ili_fdr_terms.items(), key=lambda x: x[1]['ili_fdr_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['ili_fdr_pvalues'])
            f.write(f"{term}\t{data['ili_fdr_count']}/5\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['ili_files_found_in'])}\n")
    else:
        f.write("No terms were FDR significant in ILI files.\n")
    
    f.write("\n")
    
    # ILI Uncorrected significant terms (including those already in Bonferroni or FDR)
    ili_uncorrected_terms = {term: data for term, data in term_data.items() 
                            if data['ili_uncorrected_count'] > 0}
    f.write(f"ILI Uncorrected significant terms: {len(ili_uncorrected_terms)}\n\n")
    
    if ili_uncorrected_terms:
        f.write("Term\tCount/5\tMean_p_value\tMedian_p_value\tILI_Files\n")
        for term, data in sorted(ili_uncorrected_terms.items(), key=lambda x: x[1]['ili_uncorrected_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['ili_uncorrected_pvalues'])
            f.write(f"{term}\t{data['ili_uncorrected_count']}/5\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['ili_files_found_in'])}\n")
    else:
        f.write("No terms were Uncorrected significant in ILI files.\n")
    
    f.write("\n" + "="*80 + "\n\n")
    
    # NREVSS-specific analysis
    f.write("=== NREVSS-SPECIFIC ANALYSIS (5 files) ===\n")
    
    # NREVSS Bonferroni significant terms
    nrevss_bonferroni_terms = {term: data for term, data in term_data.items() if data['nrevss_bonferroni_count'] > 0}
    f.write(f"NREVSS Bonferroni significant terms: {len(nrevss_bonferroni_terms)}\n\n")
    
    if nrevss_bonferroni_terms:
        f.write("Term\tCount/5\tMean_p_value\tMedian_p_value\tNREVSS_Files\n")
        for term, data in sorted(nrevss_bonferroni_terms.items(), key=lambda x: x[1]['nrevss_bonferroni_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['nrevss_bonferroni_pvalues'])
            f.write(f"{term}\t{data['nrevss_bonferroni_count']}/5\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['nrevss_files_found_in'])}\n")
    else:
        f.write("No terms were Bonferroni significant in NREVSS files.\n")
    
    f.write("\n")
    
    # NREVSS FDR significant terms (including those already in Bonferroni)
    nrevss_fdr_terms = {term: data for term, data in term_data.items() 
                        if data['nrevss_fdr_count'] > 0}
    f.write(f"NREVSS FDR significant terms: {len(nrevss_fdr_terms)}\n\n")
    
    if nrevss_fdr_terms:
        f.write("Term\tCount/5\tMean_p_value\tMedian_p_value\tNREVSS_Files\n")
        for term, data in sorted(nrevss_fdr_terms.items(), key=lambda x: x[1]['nrevss_fdr_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['nrevss_fdr_pvalues'])
            f.write(f"{term}\t{data['nrevss_fdr_count']}/5\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['nrevss_files_found_in'])}\n")
    else:
        f.write("No terms were FDR significant in NREVSS files.\n")
    
    f.write("\n")
    
    # NREVSS Uncorrected significant terms (including those already in Bonferroni or FDR)
    nrevss_uncorrected_terms = {term: data for term, data in term_data.items() 
                               if data['nrevss_uncorrected_count'] > 0}
    f.write(f"NREVSS Uncorrected significant terms: {len(nrevss_uncorrected_terms)}\n\n")
    
    if nrevss_uncorrected_terms:
        f.write("Term\tCount/5\tMean_p_value\tMedian_p_value\tNREVSS_Files\n")
        for term, data in sorted(nrevss_uncorrected_terms.items(), key=lambda x: x[1]['nrevss_uncorrected_count'], reverse=True):
            mean_p, median_p = calculate_statistics(data['nrevss_uncorrected_pvalues'])
            f.write(f"{term}\t{data['nrevss_uncorrected_count']}/5\t{mean_p:.6f}\t{median_p:.6f}\t{', '.join(data['nrevss_files_found_in'])}\n")
    else:
        f.write("No terms were Uncorrected significant in NREVSS files.\n")
    
    f.write("\n" + "="*80 + "\n\n")
    
    # Summary statistics
    f.write("=== SUMMARY STATISTICS ===\n")
    f.write("COMBINED (ILI + NREVSS):\n")
    f.write(f"  Bonferroni significant terms: {len(bonferroni_terms)}\n")
    f.write(f"  FDR significant terms (total): {len(bonferroni_terms) + len(fdr_terms)}\n")
    f.write(f"  Uncorrected significant terms (total): {len(bonferroni_terms) + len(fdr_terms) + len(uncorrected_terms)}\n\n")
    
    f.write("ILI-SPECIFIC:\n")
    f.write(f"  Bonferroni significant terms: {len(ili_bonferroni_terms)}\n")
    f.write(f"  FDR significant terms (total): {len(ili_bonferroni_terms) + len(ili_fdr_terms)}\n")
    f.write(f"  Uncorrected significant terms (total): {len(ili_bonferroni_terms) + len(ili_fdr_terms) + len(ili_uncorrected_terms)}\n\n")
    
    f.write("NREVSS-SPECIFIC:\n")
    f.write(f"  Bonferroni significant terms: {len(nrevss_bonferroni_terms)}\n")
    f.write(f"  FDR significant terms (total): {len(nrevss_bonferroni_terms) + len(nrevss_fdr_terms)}\n")
    f.write(f"  Uncorrected significant terms (total): {len(nrevss_bonferroni_terms) + len(nrevss_fdr_terms) + len(nrevss_uncorrected_terms)}\n\n")
    
    # Most frequently significant terms across all categories
    f.write(f"=== MOST FREQUENTLY SIGNIFICANT TERMS (ALL CATEGORIES) ===\n")
    f.write("Term\tTotal_Count/10\tBonferroni\tFDR\tUncorrected\tILI_Count\tNREVSS_Count\n")
    
    # Sort by total count across all significance levels
    for term, data in sorted(term_data.items(), 
                           key=lambda x: x[1]['bonferroni_count'] + x[1]['fdr_count'] + x[1]['uncorrected_count'], 
                           reverse=True)[:20]:  # Top 20
        total_count = data['bonferroni_count'] + data['fdr_count'] + data['uncorrected_count']
        ili_count = data['ili_bonferroni_count'] + data['ili_fdr_count'] + data['ili_uncorrected_count']
        nrevss_count = data['nrevss_bonferroni_count'] + data['nrevss_fdr_count'] + data['nrevss_uncorrected_count']
        f.write(f"{term}\t{total_count}/10\t{data['bonferroni_count']}\t{data['fdr_count']}\t{data['uncorrected_count']}\t{ili_count}/5\t{nrevss_count}/5\n")

print(f"Comprehensive analysis written to {output_file}")
print(f"Total unique terms found: {len(term_data)}")
print(f"Combined - Bonferroni: {len(bonferroni_terms)}, FDR: {len(fdr_terms)}, Uncorrected: {len(uncorrected_terms)}")
print(f"ILI - Bonferroni: {len(ili_bonferroni_terms)}, FDR: {len(ili_fdr_terms)}, Uncorrected: {len(ili_uncorrected_terms)}")
print(f"NREVSS - Bonferroni: {len(nrevss_bonferroni_terms)}, FDR: {len(nrevss_fdr_terms)}, Uncorrected: {len(nrevss_uncorrected_terms)}")