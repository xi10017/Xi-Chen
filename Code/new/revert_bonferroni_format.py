import re

def revert_bonferroni_format():
    """Revert the Bonferroni summary file back to the original format"""
    
    input_file = "ShiHaoYang/Data/bonferroni_significant_terms_summary.txt"
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(input_file, 'w') as f:
        for line in lines:
            line = line.strip()
            
            # Keep header lines and empty lines as is
            if line.startswith('===') or line.startswith('Total') or line.startswith('Unique') or line.startswith('Tests') or not line:
                f.write(line + '\n')
                continue
            
            # Keep term header lines as is
            if line.endswith(' significant terms:'):
                f.write(line + '\n')
                continue
            
            # Convert data lines from "term lag p_value test_info" back to "term_lag_X p_value test_info"
            parts = line.split(' ')
            if len(parts) >= 4:
                # Check if this looks like a data line (has a number in the second position)
                try:
                    lag = int(parts[1])
                    # This is a data line, convert it back
                    term_part = parts[0]
                    p_value = parts[2]
                    test_info = parts[3]
                    
                    # Reconstruct the original format
                    original_line = f"{term_part}_lag_{lag} {p_value} {test_info}"
                    f.write(original_line + '\n')
                except ValueError:
                    # Not a data line, keep as is
                    f.write(line + '\n')
            else:
                f.write(line + '\n')
    
    print(f"Reverted file: {input_file}")

if __name__ == "__main__":
    revert_bonferroni_format()
