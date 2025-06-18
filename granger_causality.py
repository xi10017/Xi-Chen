import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import f

# --- CONFIGURABLE SECTION ---
search_terms = [
    'cough: (United States)',
    'vaccine: (United States)',
    'fever: (United States)'
]  # Choose any subset of these columns
max_lag = 4  # Number of lags
# ----------------------------

# Read the search data and flu data
df_search = pd.read_csv("ShiHaoYang/multiTimeline-1.csv", skiprows=2)  # skiprows=2 skips the header lines
df_flu = pd.read_csv("ShiHaoYang/ILINET.csv", skiprows=1)

# Parse week and add YEAR/WEEK columns for merging
df_search['Week'] = pd.to_datetime(df_search['Week'])
df_search['YEAR'] = df_search['Week'].dt.isocalendar().year
df_search['WEEK'] = df_search['Week'].dt.isocalendar().week
print(df_flu.columns)
# Merge search data into flu data
df_flu = pd.merge(
    df_flu,
    df_search[['YEAR', 'WEEK'] + search_terms],
    on=['YEAR', 'WEEK'],
    how='left'
)

# Optionally rename columns for easier access (remove " (United States)" etc.)
rename_map = {col: col.split(':')[0] for col in search_terms}
df_flu = df_flu.rename(columns=rename_map)
search_terms_simple = [col.split(':')[0] for col in search_terms]

# Create lagged variables
flu_lags = []
all_lags = []
for lag in range(1, max_lag + 1):
    df_flu[f'flu_lag{lag}'] = df_flu['% WEIGHTED ILI'].shift(lag)
    flu_lags.append(f'flu_lag{lag}')
    for term in search_terms_simple:
        lag_col = f'{term}_lag{lag}'
        df_flu[lag_col] = df_flu[term].shift(lag)
        all_lags.append(lag_col)

df_flu = df_flu.dropna()

# Restricted model: only flu lags
X_restricted = sm.add_constant(df_flu[flu_lags])
y = df_flu['% WEIGHTED ILI']
model_restricted = sm.OLS(y, X_restricted).fit()

# Unrestricted model: flu lags + all search term lags
X_unrestricted = sm.add_constant(df_flu[flu_lags + all_lags])
model_unrestricted = sm.OLS(y, X_unrestricted).fit()

rss_restricted = np.sum(model_restricted.resid ** 2)
rss_unrestricted = np.sum(model_unrestricted.resid ** 2)
df1 = len(all_lags)
df2 = len(df_flu) - X_unrestricted.shape[1]
F = ((rss_restricted - rss_unrestricted)) / df1 / (rss_unrestricted / df2)
p_value = 1 - f.cdf(F, df1, df2)
print(f"F-statistic: {F}, p-value: {p_value}")

# Granger causality test for each search term
for term in search_terms_simple:
    print(f"\nGranger causality test for {term}:")
    data = df_flu[['% WEIGHTED ILI', term]]
    grangercausalitytests(data, maxlag=max_lag)