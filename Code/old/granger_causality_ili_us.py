import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import f

# --- CONFIGURABLE SECTION ---
max_lag = 5  # Number of lags
# ----------------------------
response_var = '% WEIGHTED ILI'  # The dependent variable in the flu data

# Read the search data and flu data
df_search = pd.read_csv("ShiHaoYang/Data/all_google_trends_us_data.csv")  # skiprows=2 skips the header lines
df_flu = pd.read_csv("ShiHaoYang/Data/ILINet.csv", skiprows=1)
df_search = df_search.loc[:, (df_search != 0).any(axis=0)]

search_terms = list(df_search.columns[1:])

#Filter flu data to only those that are national and for flu
#df_flu = df_flu[df_flu['geo_value'] == 'Ontario']
print(df_search.columns)
# Parse week and add YEAR/WEEK columns for merging
df_search['Week'] = pd.to_datetime(df_search['date'])
df_search['YEAR'] = df_search['Week'].dt.isocalendar().year
df_search['WEEK'] = df_search['Week'].dt.isocalendar().week


#print(df_flu[['YEAR', 'WEEK', 'time']].head())
#print(df_flu.columns)

# Merge search data into flu data
df_flu = pd.merge(
    df_flu,
    df_search[['YEAR', 'WEEK'] + search_terms],
    on=['YEAR', 'WEEK'],
    how='left'
)

#Time Horizon
df_flu = df_flu[df_flu['YEAR'] >= 2022]  # Filter to only include data from 2022 onwards

# Optionally rename columns for easier access (remove " (United States)" etc.)
rename_map = {col: col.split(':')[0] for col in search_terms}
df_flu = df_flu.rename(columns=rename_map)
search_terms_simple = [col.split(':')[0] for col in search_terms]

print(len(df_flu))
# Create lagged variables
flu_lags = []
all_lags = []
for lag in range(1, max_lag + 1):
    df_flu[f'flu_lag{lag}'] = df_flu[response_var].shift(lag)
    flu_lags.append(f'flu_lag{lag}')
    for term in search_terms_simple:
        lag_col = f'{term}_lag{lag}'
        df_flu[lag_col] = df_flu[term].shift(lag)
        all_lags.append(lag_col)
#df_flu = df_flu.drop(['covid', 'rsv'], axis=1)
df_flu = df_flu.dropna()
df_flu = df_flu.loc[:, (df_flu != 0).any(axis=0)]

# Only use lag columns that exist in df_flu
existing_flu_lags = [col for col in flu_lags if col in df_flu.columns]
existing_all_lags = [col for col in all_lags if col in df_flu.columns]

X_restricted = sm.add_constant(df_flu[existing_flu_lags])
y = df_flu[response_var]
model_restricted = sm.OLS(y, X_restricted).fit()

X_unrestricted = sm.add_constant(df_flu[existing_flu_lags + existing_all_lags])
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
    if term not in df_flu.columns:
        print(f"Skipping {term}: not in DataFrame")
        continue
    print(f"\nGranger causality test for {term}:")
    data = df_flu[[response_var, term]]
    grangercausalitytests(data, maxlag=max_lag)

import matplotlib.pyplot as plt
granger_pvals = []
valid_terms = []
for term in search_terms_simple:
    if term not in df_flu.columns:
        continue
    data = df_flu[[response_var, term]]
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    min_p = min([results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])
    granger_pvals.append(min_p)
    valid_terms.append(term)

plt.figure(figsize=(12, 5))
plt.bar(valid_terms, granger_pvals, color='orange')
plt.ylabel('Min p-value (across lags)')
plt.title('Granger Causality Test p-values of ILI Data in the US')
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05')
plt.xticks(rotation=90, fontsize=8)  # Rotate and shrink font
plt.tight_layout()
plt.legend()
plt.savefig("ShiHaoYang/Results/granger_pvalues_ili_us_plot.png", dpi=300)
plt.show()