import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
import matplotlib.pyplot as plt

# --- DATA PREPARATION ---
df_ili = pd.read_csv("ShiHaoYang/Data/ILINet_all.csv", skiprows=1)
df_search = pd.read_csv("ShiHaoYang/Data/flu_trends_regression_dataset.csv")

# Create common columns for merging
df_search['Week'] = pd.to_datetime(df_search['date'])
df_search['YEAR'] = df_search['Week'].dt.isocalendar().year
df_search['WEEK'] = df_search['Week'].dt.isocalendar().week

# Filter ILI data to valid rows (optional: National only)
# df_ili = df_ili[df_ili['REGION TYPE'] == 'National']

# --- DIAGNOSTICS ---
print("=== DIAGNOSTIC ANALYSIS ===")
search_terms = [
    col for col in df_search.columns
    if col not in ['date', 'Week', 'YEAR', 'WEEK']
    and np.issubdtype(df_search[col].dtype, np.number)
]  # Exclude date, Week, YEAR, WEEK

constant_columns = [col for col in search_terms if df_search[col].nunique() == 1]
low_variance_columns = [col for col in search_terms if df_search[col].std() < 0.1]
zero_dominant_columns = [col for col in search_terms if (df_search[col] == 0).mean() > 0.8]

print(f"Constant columns: {len(constant_columns)}")
print(f"Low variance columns: {len(low_variance_columns)}")
print(f"Zero dominant columns: {len(zero_dominant_columns)}")

# Filter out problematic columns
filtered_columns = [
    col for col in search_terms
    if col not in constant_columns
    and col not in low_variance_columns
    and col not in zero_dominant_columns
]
print(f"Filtered search terms: {len(filtered_columns)}")

# --- MERGE DATA ---
df_ili['YEAR'] = df_ili['YEAR'].astype(int)
df_ili['WEEK'] = df_ili['WEEK'].astype(int)
df_ili = pd.merge(
    df_ili,
    df_search[['YEAR', 'WEEK'] + filtered_columns],
    on=['YEAR', 'WEEK'],
    how='left'
)

# --- CONFIGURABLE SECTION ---
max_lag = 2
response_var = '% WEIGHTED ILI'

# --- CREATE LAGGED VARIABLES ---
flu_lags = []
all_lags = []
for lag in range(1, max_lag + 1):
    df_ili[f'ili_lag{lag}'] = df_ili[response_var].shift(lag)
    flu_lags.append(f'ili_lag{lag}')
    for term in filtered_columns:
        lag_col = f'{term}_lag{lag}'
        df_ili[lag_col] = df_ili[term].shift(lag)
        all_lags.append(lag_col)

df_ili = df_ili.dropna()
df_ili = df_ili.loc[:, (df_ili != 0).any(axis=0)]

existing_flu_lags = [col for col in flu_lags if col in df_ili.columns]
existing_all_lags = [col for col in all_lags if col in df_ili.columns]

print(f"Sample size after cleaning: {len(df_ili)}")
print(f"ILI lag columns: {len(existing_flu_lags)}")
print(f"Search term lag columns: {len(existing_all_lags)}")

# --- MULTIPLE REGRESSION GRANGER CAUSALITY ---
try:
    X_restricted = sm.add_constant(df_ili[existing_flu_lags])
    y = df_ili[response_var]
    model_restricted = sm.OLS(y, X_restricted).fit()

    X_unrestricted = sm.add_constant(df_ili[existing_flu_lags + existing_all_lags])
    model_unrestricted = sm.OLS(y, X_unrestricted).fit()

    rss_restricted = np.sum(model_restricted.resid ** 2)
    rss_unrestricted = np.sum(model_unrestricted.resid ** 2)
    df1 = len(existing_all_lags)
    df2 = len(df_ili) - X_unrestricted.shape[1]

    if df1 > 0 and df2 > 0 and rss_unrestricted > 0:
        F = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
        p_value = 1 - f.cdf(F, df1, df2)
        print(f"F-statistic: {F:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Degrees of freedom (numerator): {df1}")
        print(f"Degrees of freedom (denominator): {df2}")
        print(f"Restricted model R²: {model_restricted.rsquared:.4f}")
        print(f"Unrestricted model R²: {model_unrestricted.rsquared:.4f}")
        print(f"R² improvement: {model_unrestricted.rsquared - model_restricted.rsquared:.4f}")
    else:
        print("Error: Cannot compute F-statistic (check degrees of freedom or RSS)")
except Exception as e:
    print(f"Error in Granger causality test: {e}")

# --- INDIVIDUAL TERM SIGNIFICANCE ---
term_significance = []
for term in filtered_columns:
    term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
    term_pvals = []
    for lag_col in term_lags:
        if lag_col in model_unrestricted.params.index:
            pval = model_unrestricted.pvalues[lag_col]
            term_pvals.append(pval)
    if term_pvals:
        min_p = min(term_pvals)
        term_significance.append((term, min_p))
        print(f"Term: {term}, Min p-value across lags: {min_p:.4f}")

# --- PLOT ---
term_significance.sort(key=lambda x: x[1])
valid_terms = [term for term, pval in term_significance if not np.isnan(pval)]
granger_pvals = [pval for term, pval in term_significance if not np.isnan(pval)]

if valid_terms:
    # Dynamic figure sizing based on number of terms
    num_terms = len(valid_terms)
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

    # Create bars with better colors for significance
    colors = ['red' if pval < 0.05 else 'orange' for pval in granger_pvals]
    bars = plt.bar(valid_terms, granger_pvals, color=colors, alpha=0.7)

    plt.ylabel('Min p-value (across lags)', fontsize=12)
    plt.title(f'Individual Term Significance from Multiple Regression Model with Max Lag = {max_lag}', fontsize=14, pad=20)
    plt.axhline(0.05, color='red', linestyle='--', label='p=0.05', linewidth=2)

    # Improve x-axis labels with better rotation and positioning
    plt.xticks(rotation=45, fontsize=font_size, ha='right')

    # Add value labels on bars with better positioning
    for bar, pval in zip(bars, granger_pvals):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{pval:.3f}', ha='center', va='bottom', fontsize=value_font_size, rotation=90)

    # Set y-axis limits to ensure visibility (only if we have valid values)
    if granger_pvals:
        plt.ylim(0, max(granger_pvals) * 1.1)

    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Better layout with more space
    plt.tight_layout(pad=2.0)
    plt.savefig(f"ShiHaoYang/Results/granger_pvalues_multiple_regression_ili_lag{max_lag}.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    significant_terms = [term for term, pval in zip(valid_terms, granger_pvals) if pval < 0.05]
    print(f"\nSummary:")
    print(f"Total terms with valid p-values: {len(valid_terms)}")
    print(f"Significant terms (p < 0.05): {len(significant_terms)}")
    if valid_terms:
        print(f"Percentage significant: {len(significant_terms)/len(valid_terms)*100:.1f}%")

    if significant_terms:
        print(f"\nTop 10 most significant terms in the multiple regression:")
        sorted_valid = sorted(zip(valid_terms, granger_pvals), key=lambda x: x[1])
        for i, (term, pval) in enumerate(sorted_valid[:10]):
            print(f"{i+1}. {term}: p = {pval:.4f}")
else:
    print("No valid terms found for plotting")