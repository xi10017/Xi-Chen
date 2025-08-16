import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# --- FLU DATA PREPARATION: Combine and create percent positive columns ---
df_pub = pd.read_csv("ShiHaoYang/Data/ICL_NREVSS_Public_Health_Labs_all.csv", skiprows=1)
df_combined = pd.read_csv("ShiHaoYang/Data/ICL_NREVSS_Combined_prior_to_2015_16.csv", skiprows=1)

# Create percent positive columns
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
df_flu = pd.concat([df_combined, df_pub], ignore_index=True)

# --- CONFIGURABLE SECTION ---
max_lag = 1 # Number of lags
max_terms = None  # Maximum number of search terms to use (set to None to use all)
response_var = 'flu_total_positive'  # Or 'flu_pct_positive' for percent positive

# Load the search trends data
df_search = pd.read_csv("ShiHaoYang/Data/flu_trends_regression_dataset.csv")

# --- DIAGNOSTIC SECTION ---
print("=== DIAGNOSTIC ANALYSIS ===")
print(f"Total search terms: {len(df_search.columns) - 1}")  # Subtract date column

# Check for constant columns
constant_columns = []
low_variance_columns = []
for col in df_search.columns[1:]:  # Skip date column
    if df_search[col].nunique() == 1:
        constant_columns.append(col)
    elif df_search[col].std() < 0.1:  # Very low variance
        low_variance_columns.append(col)

print(f"\nConstant columns: {len(constant_columns)}")
if constant_columns:
    print("Examples:", constant_columns[:5])

print(f"Low variance columns (std < 0.1): {len(low_variance_columns)}")
if low_variance_columns:
    print("Examples:", low_variance_columns[:5])

# Check for columns with too many zeros
zero_dominant_columns = []
for col in df_search.columns[1:]:
    zero_ratio = (df_search[col] == 0).sum() / len(df_search)
    if zero_ratio > 0.8:  # More than 80% zeros
        zero_dominant_columns.append((col, zero_ratio))

print(f"\nColumns with >80% zeros: {len(zero_dominant_columns)}")
if zero_dominant_columns:
    print("Examples:", zero_dominant_columns[:5])

# Check correlation matrix for multicollinearity
print("\n=== MULTICOLLINEARITY CHECK ===")
search_terms = list(df_search.columns[1:])
correlation_matrix = df_search[search_terms].corr()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(search_terms)):
    for j in range(i+1, len(search_terms)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.95:  # Very high correlation
            high_corr_pairs.append((search_terms[i], search_terms[j], corr))

print(f"Pairs with correlation > 0.95: {len(high_corr_pairs)}")
if high_corr_pairs:
    print("Examples:", high_corr_pairs[:3])

# Recommend filtering strategy
print("\n=== RECOMMENDATIONS ===")
print("1. Remove constant columns")
print("2. Remove columns with >80% zeros")
print("3. Remove one of each highly correlated pair")
print("4. Consider using fewer terms (max_terms = 20-50)")

# Filter out problematic columns
filtered_columns = []
for col in search_terms:
    # Skip constant columns
    if col in constant_columns:
        continue
    # Skip low variance columns
    if col in low_variance_columns:
        continue
    # Skip zero-dominant columns
    zero_ratio = (df_search[col] == 0).sum() / len(df_search)
    if zero_ratio > 0.8:
        continue
    filtered_columns.append(col)

print(f"\nAfter filtering problematic columns: {len(filtered_columns)} terms remaining")
print("This should significantly reduce NaN p-values!")

# Use filtered columns for the analysis
search_terms = filtered_columns[:max_terms] if max_terms else filtered_columns

print(f"Final number of search terms to use: {len(search_terms)}")
print("Search terms:", search_terms[:10], "...")  # Show first 10 terms

# Parse week and add YEAR/WEEK columns for merging
df_search['Week'] = pd.to_datetime(df_search['date'])
df_search['YEAR'] = df_search['Week'].dt.isocalendar().year
df_search['WEEK'] = df_search['Week'].dt.isocalendar().week

# Merge search data into flu data
df_flu = pd.merge(
    df_flu,
    df_search[['YEAR', 'WEEK'] + search_terms],
    on=['YEAR', 'WEEK'],
    how='left'
)

# Time Horizon - Exclude COVID years (2020-2021)
df_flu = df_flu[(df_flu['YEAR'] < 2019) | (df_flu['YEAR'] > 2022)]  # Exclude 2020 and 2021

# Rename columns for easier access
rename_map = {col: col.split(':')[0] for col in search_terms}
df_flu = df_flu.rename(columns=rename_map)
search_terms_simple = [col.split(':')[0] for col in search_terms]

print(f"Data points after filtering: {len(df_flu)}")

# Create lagged variables
flu_lags = []
all_lags = []

for lag in range(1, max_lag + 1):
    df_flu[f'flu_lag{lag}'] = df_flu[response_var].shift(lag)
    flu_lags.append(f'flu_lag{lag}')
    
    for term in search_terms_simple:
        if term in df_flu.columns:  # Only create lags for terms that exist
            lag_col = f'{term}_lag{lag}'
            df_flu[lag_col] = df_flu[term].shift(lag)
            all_lags.append(lag_col)

# Clean data
df_flu = df_flu.dropna()
df_flu = df_flu.loc[:, (df_flu != 0).any(axis=0)]

print(f"Data points after cleaning: {len(df_flu)}")

# Only use lag columns that exist in df_flu
existing_flu_lags = [col for col in flu_lags if col in df_flu.columns]
existing_all_lags = [col for col in all_lags if col in df_flu.columns]

print(f"Flu lag columns: {len(existing_flu_lags)}")
print(f"Search term lag columns: {len(existing_all_lags)}")

# Check for constant values in response variable
if df_flu[response_var].nunique() <= 1:
    print(f"Error: Response variable '{response_var}' has constant values")
    exit()

# Check for constant values in search terms
constant_terms = []
for term in search_terms_simple:
    if term in df_flu.columns and df_flu[term].nunique() <= 1:
        constant_terms.append(term)

if constant_terms:
    print(f"Warning: The following terms have constant values and will be excluded: {constant_terms}")
    # Remove constant terms from existing_all_lags
    existing_all_lags = [col for col in existing_all_lags 
                        if not any(term in col for term in constant_terms)]

print(f"Final search term lag columns: {len(existing_all_lags)}")

# Perform Multiple Linear Regression Granger Causality Test
try:
    # Restricted model (only flu lags)
    X_restricted = sm.add_constant(df_flu[existing_flu_lags])
    y = df_flu[response_var]
    model_restricted = sm.OLS(y, X_restricted).fit()
    
    # Unrestricted model (flu lags + all search term lags)
    X_unrestricted = sm.add_constant(df_flu[existing_flu_lags + existing_all_lags])
    model_unrestricted = sm.OLS(y, X_unrestricted).fit()
    
    # Calculate F-statistic
    rss_restricted = np.sum(model_restricted.resid ** 2)
    rss_unrestricted = np.sum(model_unrestricted.resid ** 2)
    df1 = len(existing_all_lags)  # Number of additional parameters
    df2 = len(df_flu) - X_unrestricted.shape[1]  # Degrees of freedom for error
    
    if df1 > 0 and df2 > 0 and rss_unrestricted > 0:
        F = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
        p_value = 1 - f.cdf(F, df1, df2)
        
        print("\n" + "="*60)
        print("MULTIPLE LINEAR REGRESSION GRANGER CAUSALITY TEST")
        print("="*60)
        print(f"Testing if ALL search terms together Granger-cause '{response_var}'")
        print(f"Number of search terms included: {len(existing_all_lags) // max_lag}")
        print(f"Number of lags: {max_lag}")
        print(f"Sample size: {len(df_flu)}")
        print(f"F-statistic: {F:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Degrees of freedom (numerator): {df1}")
        print(f"Degrees of freedom (denominator): {df2}")
        
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = ""
        
        print(f"Significance: {significance}")
        
        if p_value < 0.05:
            print("CONCLUSION: Search terms collectively Granger-cause flu activity (p < 0.05)")
        else:
            print("CONCLUSION: No evidence that search terms collectively Granger-cause flu activity")
            
        # Model fit statistics
        print(f"\nModel Fit Statistics:")
        print(f"Restricted model R²: {model_restricted.rsquared:.4f}")
        print(f"Unrestricted model R²: {model_unrestricted.rsquared:.4f}")
        print(f"R² improvement: {model_unrestricted.rsquared - model_restricted.rsquared:.4f}")
        
    else:
        print("Error: Cannot compute F-statistic (check degrees of freedom or RSS)")
        
except Exception as e:
    print(f"Error in Granger causality test: {e}")
    print("This might be due to:")
    print("- Too many variables relative to sample size")
    print("- Multicollinearity between variables")
    print("- Insufficient data after cleaning")

# Individual term significance from the big multiple regression model
print("\n" + "="*60)
print("INDIVIDUAL TERM SIGNIFICANCE FROM MULTIPLE REGRESSION")
print("="*60)

# Extract individual term significance from the unrestricted model
term_significance = []

# Get the summary of the unrestricted model
model_summary = model_unrestricted.summary()

# Extract coefficients and p-values for search term lags
for term in search_terms_simple:
    term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
    term_pvals = []
    
    for lag_col in term_lags:
        if lag_col in model_unrestricted.params.index:
            # Get the coefficient and p-value for this lag
            coef = model_unrestricted.params[lag_col]
            pval = model_unrestricted.pvalues[lag_col]
            term_pvals.append(pval)
    
    if term_pvals:
        # Use the minimum p-value across all lags for this term
        min_p = min(term_pvals)
        term_significance.append((term, min_p))

# Sort by significance
term_significance.sort(key=lambda x: x[1])

# Calculate Bonferroni-corrected significance threshold
num_tests = len(term_significance)
bonferroni_threshold = 0.05 / num_tests

# Perform FDR correction (Benjamini-Hochberg)
search_term_pvalues = [pval for term, pval in term_significance]
fdr_rejected, fdr_pvalues, _, _ = multipletests(search_term_pvalues, method='fdr_bh', alpha=0.05)

# Create mapping of terms to FDR significance
fdr_significant_terms = set()
for i, (term, pval) in enumerate(term_significance):
    if fdr_rejected[i]:
        fdr_significant_terms.add(term)

print(f"Number of search terms tested: {num_tests}")
print(f"Bonferroni-corrected significance threshold: {bonferroni_threshold:.6f} (0.05/{num_tests})")
print(f"FDR correction applied (Benjamini-Hochberg method)")

# Extract terms and p-values for plotting
valid_terms = [term for term, pval in term_significance]
granger_pvals = [pval for term, pval in term_significance]

# Identify significant terms with different thresholds
significant_uncorrected = [term for term, pval in term_significance if pval < 0.05]
significant_bonferroni = [term for term, pval in term_significance if pval < bonferroni_threshold]

print(f"Significant terms (uncorrected p < 0.05): {len(significant_uncorrected)}")
print(f"Significant terms (Bonferroni-corrected p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)}")
print(f"Significant terms (FDR-corrected): {len(fdr_significant_terms)}")

# Create the plot
if valid_terms:
    # Check for valid p-values (not nan)
    valid_pvals = [(term, pval) for term, pval in term_significance if not np.isnan(pval)]
    
    if not valid_pvals:
        print("\nWARNING: All p-values are NaN. This indicates the multiple regression model failed to fit properly.")
        print("Possible causes:")
        print("- Too many variables relative to sample size")
        print("- Perfect multicollinearity between variables")
        print("- Insufficient data after cleaning")
        print("- Singular matrix in regression")
        print("\nConsider:")
        print("- Reducing the number of lags (max_lag)")
        print("- Using fewer search terms")
        print("- Increasing the sample size")
        print("- Checking for duplicate or highly correlated variables")
    else:
        # Use only valid p-values for plotting
        valid_terms_plot = [term for term, pval in valid_pvals]
        granger_pvals_plot = [pval for term, pval in valid_pvals]
        
        # Dynamic figure sizing based on number of terms
        num_terms = len(valid_terms_plot)
        if num_terms <= 20:
            fig_width = 16
            fig_height = 8
            font_size = 8
            value_font_size = 6
        elif num_terms <= 50:
            fig_width = 20
            fig_height = 10
            font_size = 6
            value_font_size = 5
        else:
            fig_width = 24
            fig_height = 12
            font_size = 4
            value_font_size = 4
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Create bars with colors for different significance levels
        colors = []
        for i, term in enumerate(valid_terms_plot):
            pval = granger_pvals_plot[i]
            if term in fdr_significant_terms:
                colors.append('purple')  # FDR significant
            elif pval < bonferroni_threshold:
                colors.append('darkred')  # Bonferroni significant
            elif pval < 0.05:
                colors.append('red')      # Uncorrected significant
            else:
                colors.append('orange')   # Not significant
        
        bars = plt.bar(valid_terms_plot, granger_pvals_plot, color=colors, alpha=0.7)
        
        plt.ylabel('Min p-value (across lags)', fontsize=12)
        plt.title(f'Individual Term Significance from Multiple Regression Model with Max Lag = {max_lag}', fontsize=14, pad=20)
        plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 (uncorrected)', linewidth=2)
        
        # Add custom legend entries for bar colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='purple', alpha=0.7, label='FDR significant'),
            Patch(facecolor='darkred', alpha=0.7, label='Bonferroni significant'),
            Patch(facecolor='red', alpha=0.7, label='Uncorrected significant'),
            Patch(facecolor='orange', alpha=0.7, label='Not significant')
        ]
        
        # Improve x-axis labels with better rotation and positioning
        plt.xticks(rotation=45, fontsize=font_size, ha='right')
        
        # Add value labels on bars with better positioning
        for bar, pval in zip(bars, granger_pvals_plot):
            height = bar.get_height()
            # Position text above bar with small offset
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{pval:.3f}', ha='center', va='bottom', fontsize=value_font_size, rotation=90)
        
        # Set y-axis limits to ensure visibility (only if we have valid values)
        if granger_pvals_plot:
            plt.ylim(0, max(granger_pvals_plot) * 1.1)
        
        plt.legend(handles=legend_elements, fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Better layout with more space
        plt.tight_layout(pad=2.0)
        plt.savefig(f"ShiHaoYang/Results/granger_pvalues_multiple_regression_nrevss_lag{max_lag}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        significant_uncorrected_plot = [term for term, pval in valid_pvals if pval < 0.05]
        significant_bonferroni_plot = [term for term, pval in valid_pvals if pval < bonferroni_threshold]
        significant_fdr_plot = [term for term in valid_terms_plot if term in fdr_significant_terms]
        
        print(f"\nSummary:")
        print(f"Total terms with valid p-values: {len(valid_terms_plot)}")
        print(f"Terms with NaN p-values: {len(valid_terms) - len(valid_terms_plot)}")
        print(f"Uncorrected (p < 0.05): {len(significant_uncorrected_plot)} terms")
        print(f"Bonferroni-corrected (p < {bonferroni_threshold:.6f}): {len(significant_bonferroni_plot)} terms")
        print(f"FDR-corrected: {len(significant_fdr_plot)} terms")
        
        if significant_fdr_plot:
            print(f"\nTop 5 FDR-significant terms:")
            sorted_valid = sorted(valid_pvals, key=lambda x: x[1])
            for i, (term, pval) in enumerate(sorted_valid[:5]):
                significance = "***" if term in fdr_significant_terms else "**" if pval < 0.05 else ""
                print(f"{i+1}. {term}: p = {pval:.4f}{significance}")
        elif significant_bonferroni_plot:
            print(f"\nTop 5 Bonferroni-significant terms:")
            sorted_valid = sorted(valid_pvals, key=lambda x: x[1])
            for i, (term, pval) in enumerate(sorted_valid[:5]):
                significance = "***" if pval < bonferroni_threshold else "**" if pval < 0.05 else ""
                print(f"{i+1}. {term}: p = {pval:.4f}{significance}")
        elif significant_uncorrected_plot:
            print(f"\nTop 5 uncorrected-significant terms:")
            sorted_valid = sorted(valid_pvals, key=lambda x: x[1])
            for i, (term, pval) in enumerate(sorted_valid[:5]):
                print(f"{i+1}. {term}: p = {pval:.4f}")
else:
    print("No valid terms found for plotting")

# Save significant terms to a text file
txt_filename = f"ShiHaoYang/Results/granger_significant_terms_nrevss_lag{max_lag}.txt"
with open(txt_filename, "w") as f:
    f.write(f"Significant terms for NREVSS multiple regression, max_lag={max_lag}\n")
    f.write(f"Number of tests: {num_tests}\n")
    f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
    f.write(f"FDR correction applied (Benjamini-Hochberg method)\n")
    f.write(f"Total significant terms: {len(fdr_significant_terms) if fdr_significant_terms else len(significant_bonferroni)}\n\n")
    
    # Prioritize FDR significant terms, then Bonferroni
    if fdr_significant_terms:
        f.write("FDR-significant terms:\n")
        for term in fdr_significant_terms:
            pval = [p for t, p in term_significance if t == term][0]
            f.write(f"{term}: p = {pval:.6f}\n")
    elif significant_bonferroni:
        f.write("Bonferroni-significant terms:\n")
        for term in significant_bonferroni:
            pval = [p for t, p in term_significance if t == term][0]
            f.write(f"{term}: p = {pval:.6f}\n")
    else:
        f.write("None\n")
        
        # Also save uncorrected results for reference
        if significant_uncorrected:
            f.write(f"\nUncorrected significant terms (p < 0.05): {len(significant_uncorrected)}\n")
            for term in significant_uncorrected:
                pval = [p for t, p in term_significance if t == term][0]
                f.write(f"{term}: p = {pval:.6f}\n")

print(f"Significant terms saved to {txt_filename}")