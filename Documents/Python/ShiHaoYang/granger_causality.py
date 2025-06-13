import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import f
df_cough = pd.read_csv("multiTimeline-1.csv")
df_flu = pd.read_csv("FluView_StackedColumnChart_Data.csv")


df_cough['Week'] = pd.to_datetime(df_cough['Week'])

df_cough['YEAR'] = df_cough['Week'].dt.isocalendar().year
df_cough['WEEK'] = df_cough['Week'].dt.isocalendar().week
print(df_cough.head())
print(df_flu.head())
df_flu = pd.merge(df_flu, df_cough[['YEAR', 'WEEK', 'cough: (United States)']], on=['YEAR', 'WEEK'])
df_flu = df_flu.rename(columns={'cough: (United States)': 'cough'})

#Perform F-test for Granger causality manually
shifts = [1, 2, 3, 4]
flu_lags = []
cough_lags = []
for shift in shifts:
    df_flu[f'flu_lag{shift}'] = df_flu['PERCENT POSITIVE'].shift(shift)
    df_flu[f'cough_lag{shift}'] = df_flu['cough'].shift(shift)
    flu_lags.append(f'flu_lag{shift}')
    cough_lags.append(f'cough_lag{shift}')


df_flu = df_flu.dropna()
X_restricted = sm.add_constant(df_flu[flu_lags])
y = df_flu['PERCENT POSITIVE']
model_restricted = sm.OLS(y, X_restricted).fit()

X_unrestricted = sm.add_constant(df_flu[flu_lags + cough_lags])
y = df_flu['PERCENT POSITIVE']
model_unrestricted = sm.OLS(y, X_unrestricted).fit()

rss_restricted = np.sum(model_restricted.resid ** 2)
rss_unrestricted = np.sum(model_unrestricted.resid ** 2)
df1 = len(shifts)
df2 = len(df_flu) - X_unrestricted.shape[1]
F = ((rss_restricted - rss_unrestricted)) / df1 / (rss_unrestricted / df2)
p_value = 1 - f.cdf(F, df1, df2)
print(f"F-statistic: {F}, p-value: {p_value}")

# Perform Granger causality test automatically
data = df_flu[['PERCENT POSITIVE', 'cough']]
grangercausalitytests(data, maxlag=4)
