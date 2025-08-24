import pandas as pd

def get_dataframe_ili():
    df_ili = pd.read_csv("ShiHaoYang/Data/ILINET_all.csv", skiprows=1)
    df_flu = pd.read_csv("ShiHaoYang/Data/flu_trends_regression_dataset.csv")
    # Convert date columns to allow merging on YEAR and WEEK
    # For df_flu, extract YEAR and WEEK from the 'date' column
    df_flu['Week'] = pd.to_datetime(df_flu['date'])
    df_flu['YEAR'] = df_flu['Week'].dt.isocalendar().year
    df_flu['WEEK'] = df_flu['Week'].dt.isocalendar().week

    # For df_ili, ensure YEAR and WEEK are integers (in case they are strings)
    df_ili['YEAR'] = df_ili['YEAR'].astype(int)
    df_ili['WEEK'] = df_ili['WEEK'].astype(int)

    # Merge on YEAR and WEEK
    merged_df = pd.merge(df_ili, df_flu, on=['YEAR', 'WEEK'], how='left')

    return merged_df


def get_dataframe_nrevss():
    df_pub = pd.read_csv("ShiHaoYang/Data/ICL_NREVSS_Public_Health_Labs_all.csv", skiprows=1)
    df_combined = pd.read_csv("ShiHaoYang/Data/ICL_NREVSS_Combined_prior_to_2015_16.csv", skiprows=1)
    df_flu = pd.read_csv("ShiHaoYang/Data/flu_trends_regression_dataset.csv")
    # Prepare percent positive columns for both public and combined NREVSS datasets
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
    df_nrevss = pd.concat([df_combined, df_pub], ignore_index=True)

    # Optionally, merge with search trends data on YEAR and WEEK
    df_flu['Week'] = pd.to_datetime(df_flu['date'])
    df_flu['YEAR'] = df_flu['Week'].dt.isocalendar().year
    df_flu['WEEK'] = df_flu['Week'].dt.isocalendar().week

    merged_df = pd.merge(df_nrevss, df_flu, on=['YEAR', 'WEEK'], how='left')

    return merged_df