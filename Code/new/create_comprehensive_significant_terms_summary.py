import os
import glob
import re

def parse_detailed_file(filepath):
    """Parse a detailed significant terms file and extract Bonferroni-significant terms with their lag information"""
    significant_terms = []
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract test type and max lag from filename or content
        if 'ili' in filepath.lower():
            test_type = 'ili'
        elif 'nrevss' in filepath.lower():
            test_type = 'nrevss'
        else:
            test_type = 'unknown'
        
        # Extract max lag from filename
        lag_match = re.search(r'lag(\d+)', filepath)
        max_lag = lag_match.group(1) if lag_match else 'unknown'
        

        
        # Look for significant terms in the main data section
        lines = content.split('\n')
        in_data_section = False
        found_format_line = False
        
        for line in lines:
            line = line.strip()
            
            # Start of data section - look for Format line
            if line.startswith('Format: Term'):
                found_format_line = True
                continue
            
            # After Format line, skip the === separator and start parsing
            if found_format_line and line.startswith('==='):
                in_data_section = True
                continue
            
            # End of data section when we hit another === or SUMMARY
            if in_data_section and (line.startswith('===') or line.startswith('SUMMARY')):
                in_data_section = False
                continue
            
            # Parse data lines
            if in_data_section and '\t' in line and not line.startswith('Format:'):
                parts = line.split('\t')
                if len(parts) >= 4:
                    term = parts[0].strip()
                    lag = parts[1].strip()
                    try:
                        p_value = float(parts[3].strip())
                        # Only include Bonferroni significant terms (will check threshold later)
                        significant_terms.append({
                            'term': term,
                            'lag': lag,
                            'p_value': p_value,
                            'test_type': test_type,
                            'max_lag': max_lag
                        })
                    except ValueError:
                        continue
        
        return significant_terms
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def create_comprehensive_significant_terms_summary():
    """Create a comprehensive summary of all Bonferroni-significant terms"""
    print("=== CREATING COMPREHENSIVE SIGNIFICANT TERMS SUMMARY ===")
    
    # Find all detailed significant terms files
    detailed_files = glob.glob("ShiHaoYang/Results/new/detailed_significant_terms_*.txt")
    
    if not detailed_files:
        print("No detailed significant terms files found!")
        return
    
    print(f"Found {len(detailed_files)} detailed files to process")
    
    # Parse all files and collect significant terms
    all_significant_terms = []
    for filepath in detailed_files:
        print(f"Processing: {os.path.basename(filepath)}")
        terms = parse_detailed_file(filepath)
        all_significant_terms.extend(terms)
    
    if not all_significant_terms:
        print("No Bonferroni-significant terms found in any files!")
        return
    
    # Group terms by term name
    terms_dict = {}
    for item in all_significant_terms:
        term = item['term']
        if term not in terms_dict:
            terms_dict[term] = []
        terms_dict[term].append(item)
    
    # Create output file
    output_filename = "ShiHaoYang/Data/bonferroni_significant_terms_summary.txt"
    
    with open(output_filename, 'w') as f:
        f.write("=== BONFERRONI SIGNIFICANT TERMS SUMMARY ===\n")
        f.write(f"Total Bonferroni-significant term-lag combinations: {len(all_significant_terms)}\n")
        f.write(f"Unique terms: {len(terms_dict)}\n")
        f.write(f"Tests included: {', '.join(set([item['test_type'] for item in all_significant_terms]))}\n\n")
        
        # Sort terms by minimum p-value
        sorted_terms = sorted(terms_dict.items(), 
                            key=lambda x: min([item['p_value'] for item in x[1]]))
        
        for term, term_data in sorted_terms:
            f.write(f"{term} significant terms:\n")
            
            # Sort by test type and lag
            sorted_data = sorted(term_data, key=lambda x: (x['test_type'], int(x['lag'])))
            
            for item in sorted_data:
                test_identifier = f"{item['test_type']}_maxlag_{item['max_lag']}"
                f.write(f"{term}_lag_{item['lag']} {item['p_value']:.6f} {test_identifier}\n")
            
            f.write("\n")
    
    print(f"Comprehensive summary saved to {output_filename}")
    return output_filename

if __name__ == "__main__":
    create_comprehensive_significant_terms_summary()