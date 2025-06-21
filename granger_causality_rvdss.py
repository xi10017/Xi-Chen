import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import f
country = 'Canada'  # Change this to the desired country

# --- CONFIGURABLE SECTION ---
search_terms = [
    f'cough: ({country})',
    f'vaccine: ({country})',
    f'fever: ({country})'
]  # Choose any subset of these columns
max_lag = 3  # Number of lags
# ----------------------------
response_var = 'flu_pct_positive'  # The dependent variable in the flu data

# Read the search data and flu data
df_search = pd.read_csv("ShiHaoYang/multiTimeline-canada.csv", skiprows=2)  # skiprows=2 skips the header lines
df_flu = pd.read_csv("ShiHaoYang/concatenated_rvdss_data.csv")

#Filter flu data to only those that are national and for flu
df_flu = df_flu[df_flu['geo_type'] == 'nation']

# Parse week and add YEAR/WEEK columns for merging
df_search['Week'] = pd.to_datetime(df_search['Week'])
df_search['YEAR'] = df_search['Week'].dt.isocalendar().year
df_search['WEEK'] = df_search['Week'].dt.isocalendar().week
df_flu['Week'] = pd.to_datetime(df_flu['time_value'])
df_flu['YEAR'] = df_flu['Week'].dt.isocalendar().year
df_flu['WEEK'] = df_flu['Week'].dt.isocalendar().week

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
df_flu = df_flu.drop(['rsv_pct_positive', 'sarscov2_pct_positive'], axis=1)
df_flu = df_flu.dropna()

# Restricted model: only flu lags
X_restricted = sm.add_constant(df_flu[flu_lags])
y = df_flu[response_var]
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
    data = df_flu[[response_var, term]]
    grangercausalitytests(data, maxlag=max_lag)
