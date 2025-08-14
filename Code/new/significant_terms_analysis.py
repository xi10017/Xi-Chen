import os
import re
from collections import Counter

# Folder containing significant terms txt files
input_folder = "ShiHaoYang/Data/significant_terms"
output_file = "ShiHaoYang/Results/granger_significant_term_candidates_summary.txt"

# Regex to match lines with significant terms
term_line_pattern = re.compile(r"^(.+): p = ([0-9.]+)$")

term_counter = Counter()

for fname in os.listdir(input_folder):
    if fname.startswith("granger_significant_terms_") and fname.endswith(".txt"):
        with open(os.path.join(input_folder, fname), "r") as f:
            for line in f:
                match = term_line_pattern.match(line.strip())
                if match:
                    term = match.group(1)
                    term_counter[term] += 1

# Write summary to output file
with open(output_file, "w") as f:
    f.write("Significant term candidates across all Granger causality runs\n")
    f.write("Term\tCount\n")
    for term, count in term_counter.most_common():
        f.write(f"{term}\t{count}\n")

print(f"Summary written to {output_file}")