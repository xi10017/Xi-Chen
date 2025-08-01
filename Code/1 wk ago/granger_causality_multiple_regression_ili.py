import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from scipy.stats import f
import matplotlib.pyplot as plt

# --- CONFIGURABLE SECTION ---
max_lag = 5  # Number of lags
max_terms = 40  # Maximum number of search terms to use (set to None to use all)
# ----------------------------
response_var = 'total_flu_positives'  # The dependent variable in the flu data

# Read the search data and flu data
df_search = pd.read_csv("ShiHaoYang/Data/trends_us_data_grouped.csv")
df_flu = pd.read_csv("ShiHaoYang/Data/ICL_NREVSS_Public_Health_Labs.csv", skiprows=1)

# Create total flu positives column
flu_case_cols = ['A (2009 H1N1)', 'A (H3)', 'A (Subtyping not Performed)', 'B']
df_flu['total_flu_positives'] = df_flu[flu_case_cols].sum(axis=1)

# Filter out zero columns
df_search = df_search.loc[:, (df_search != 0).any(axis=0)]

search_terms = list(df_search.columns[1:])

# Limit the number of search terms if specified
if max_terms is not None and len(search_terms) > max_terms:
    search_terms = search_terms[:max_terms]
    print(f"Limited to first {max_terms} search terms out of {len(df_search.columns)-1} total")

print(f"Number of search terms: {len(search_terms)}")
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

# Time Horizon
df_flu = df_flu[df_flu['YEAR'] >= 2022]  # Filter to only include data from 2022 onwards

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
        print(f"Term: {term}, Min p-value across lags: {min_p:.4f}")

# Sort by significance
term_significance.sort(key=lambda x: x[1])

# Extract terms and p-values for plotting
valid_terms = [term for term, pval in term_significance]
granger_pvals = [pval for term, pval in term_significance]

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
        
        plt.figure(figsize=(16, 8))  # Similar to the original script
        
        # Create bars with better colors for significance
        colors = ['red' if pval < 0.05 else 'orange' for pval in granger_pvals_plot]
        bars = plt.bar(valid_terms_plot, granger_pvals_plot, color=colors, alpha=0.7)
        
        plt.ylabel('Min p-value (across lags)', fontsize=12)
        plt.title(f'Individual Term Significance from Multiple Regression Model with Max Lag = {max_lag}', fontsize=14, pad=20)
        plt.axhline(0.05, color='red', linestyle='--', label='p=0.05', linewidth=2)
        
        # Improve x-axis labels with better rotation and positioning
        plt.xticks(rotation=45, fontsize=5, ha='right')
        
        # Add value labels on bars with better positioning
        for bar, pval in zip(bars, granger_pvals_plot):
            height = bar.get_height()
            # Position text above bar with small offset
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{pval:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)
        
        # Set y-axis limits to ensure visibility (only if we have valid values)
        if granger_pvals_plot:
            plt.ylim(0, max(granger_pvals_plot) * 1.1)
        
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Better layout with more space
        plt.tight_layout()
        plt.savefig(f"ShiHaoYang/Results/granger_pvalues_multiple_regression_ili_lag{max_lag}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        significant_terms = [term for term, pval in valid_pvals if pval < 0.05]
        print(f"\nSummary:")
        print(f"Total terms with valid p-values: {len(valid_terms_plot)}")
        print(f"Terms with NaN p-values: {len(valid_terms) - len(valid_terms_plot)}")
        print(f"Significant terms (p < 0.05): {len(significant_terms)}")
        if valid_terms_plot:
            print(f"Percentage significant: {len(significant_terms)/len(valid_terms_plot)*100:.1f}%")
        
        if significant_terms:
            print(f"\nTop 10 most significant terms in the multiple regression:")
            sorted_valid = sorted(valid_pvals, key=lambda x: x[1])
            for i, (term, pval) in enumerate(sorted_valid[:10]):
                print(f"{i+1}. {term}: p = {pval:.4f}")
else:
    print("No valid terms found for plotting") 