import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from scipy.stats import f, t
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac_simple, cov_hac
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare NREVSS flu data and search trends data"""
    print("=== LOADING AND PREPARING DATA ===")
    
    # Load flu data
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

    # Load search trends data
    df_search = pd.read_csv("ShiHaoYang/Data/flu_trends_regression_dataset.csv")
    
    print(f"Flu data shape: {df_flu.shape}")
    print(f"Search data shape: {df_search.shape}")
    
    return df_flu, df_search

def perform_data_diagnostics(df_search):
    """Perform diagnostic analysis on search terms data"""
    print("\n=== DIAGNOSTIC ANALYSIS ===")
    print(f"Total search terms: {len(df_search.columns) - 1}")

    # Check for constant columns
    constant_columns = []
    low_variance_columns = []
    for col in df_search.columns[1:]:
        if df_search[col].nunique() == 1:
            constant_columns.append(col)
        elif df_search[col].std() < 0.1:
            low_variance_columns.append(col)

    print(f"Constant columns: {len(constant_columns)}")
    print(f"Low variance columns (std < 0.1): {len(low_variance_columns)}")

    # Check for columns with too many zeros
    zero_dominant_columns = []
    for col in df_search.columns[1:]:
        zero_ratio = (df_search[col] == 0).sum() / len(df_search)
        if zero_ratio > 0.8:
            zero_dominant_columns.append((col, zero_ratio))

    print(f"Columns with >80% zeros: {len(zero_dominant_columns)}")

    # Filter out problematic columns
    filtered_columns = []
    for col in df_search.columns[1:]:
        if col in constant_columns or col in low_variance_columns:
            continue
        zero_ratio = (df_search[col] == 0).sum() / len(df_search)
        if zero_ratio > 0.8:
            continue
        filtered_columns.append(col)

    print(f"After filtering: {len(filtered_columns)} terms remaining")
    return filtered_columns

def prepare_merged_data(df_flu, df_search, search_terms, max_lag, response_var):
    """Prepare merged dataset with lagged variables"""
    print(f"\n=== PREPARING MERGED DATA ===")
    
    # Parse week and add YEAR/WEEK columns for merging
    df_search['Week'] = pd.to_datetime(df_search['date'])
    df_search['YEAR'] = df_search['Week'].dt.isocalendar().year
    df_search['WEEK'] = df_search['Week'].dt.isocalendar().week

    # Merge search data into flu data with suffixes to avoid conflicts
    df_flu = pd.merge(
        df_flu,
        df_search[['YEAR', 'WEEK'] + search_terms],
        on=['YEAR', 'WEEK'],
        how='left',
        suffixes=('', '_search')
    )

    # Time Horizon - Exclude COVID years (2020-2021)
    df_flu = df_flu[(df_flu['YEAR'] < 2019) | (df_flu['YEAR'] > 2022)]

    # Rename columns for easier access and ensure uniqueness
    rename_map = {}
    search_terms_simple = []
    seen_terms = set()
    
    for col in search_terms:
        simple_name = col.split(':')[0]
        # Check if the column exists with _search suffix
        search_col = f"{col}_search"
        if search_col in df_flu.columns:
            # Use the search column
            if simple_name in seen_terms:
                # If duplicate, add a suffix to make it unique
                counter = 1
                while f"{simple_name}_{counter}" in seen_terms:
                    counter += 1
                unique_name = f"{simple_name}_{counter}"
                rename_map[search_col] = unique_name
                search_terms_simple.append(unique_name)
                seen_terms.add(unique_name)
            else:
                rename_map[search_col] = simple_name
                search_terms_simple.append(simple_name)
                seen_terms.add(simple_name)
        else:
            # Use the original column
            if simple_name in seen_terms:
                # If duplicate, add a suffix to make it unique
                counter = 1
                while f"{simple_name}_{counter}" in seen_terms:
                    counter += 1
                unique_name = f"{simple_name}_{counter}"
                rename_map[col] = unique_name
                search_terms_simple.append(unique_name)
                seen_terms.add(unique_name)
            else:
                rename_map[col] = simple_name
                search_terms_simple.append(simple_name)
                seen_terms.add(simple_name)
    
    df_flu = df_flu.rename(columns=rename_map)

    print(f"Data points after filtering: {len(df_flu)}")

    # Create lagged variables
    flu_lags = []
    all_lags = []

    for lag in range(1, max_lag + 1):
        df_flu[f'flu_lag{lag}'] = df_flu[response_var].shift(lag)
        flu_lags.append(f'flu_lag{lag}')
        
        for term in search_terms_simple:
            if term in df_flu.columns:
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

    return df_flu, existing_flu_lags, existing_all_lags, search_terms_simple

def perform_granger_causality_test(df_flu, existing_flu_lags, existing_all_lags, response_var):
    """Perform the main Granger causality test"""
    print("\n" + "="*60)
    print("MULTIPLE LINEAR REGRESSION GRANGER CAUSALITY TEST")
    print("="*60)
    
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
        df1 = len(existing_all_lags)
        df2 = len(df_flu) - X_unrestricted.shape[1]
        
        if df1 > 0 and df2 > 0 and rss_unrestricted > 0:
            F = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
            p_value = 1 - f.cdf(F, df1, df2)
            
            print(f"Testing if ALL search terms together Granger-cause '{response_var}'")
            print(f"Number of search terms included: {len(existing_all_lags) // len(existing_flu_lags)}")
            print(f"Number of lags: {len(existing_flu_lags)}")
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
            
            return model_restricted, model_unrestricted, F, p_value, X_unrestricted, y
            
        else:
            print("Error: Cannot compute F-statistic (check degrees of freedom or RSS)")
            return None, None, None, None, None, None
            
    except Exception as e:
        print(f"Error in Granger causality test: {e}")
        return None, None, None, None, None, None

def detect_autocorrelation(model_unrestricted):
    """Detect autocorrelation in residuals using Durbin-Watson test"""
    print(f"\n=== AUTOCORRELATION DETECTION ===")
    
    # Durbin-Watson test
    dw_stat = sm.stats.durbin_watson(model_unrestricted.resid)
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    
    # Interpretation
    if dw_stat < 1.5:
        print("Strong positive autocorrelation detected")
        has_autocorrelation = True
    elif dw_stat > 2.5:
        print("Strong negative autocorrelation detected")
        has_autocorrelation = True
    else:
        print("No significant autocorrelation detected")
        has_autocorrelation = False
    
    return has_autocorrelation

def perform_hac_analysis(df_flu, existing_flu_lags, existing_all_lags, response_var, F, p_value, maxlags=None):
    """Perform HAC analysis using direct covariance matrix computation"""
    print(f"\n=== HAC ANALYSIS ===")
    
    try:
        # Fit unrestricted model
        X_unrestricted = sm.add_constant(df_flu[existing_flu_lags + existing_all_lags])
        y = df_flu[response_var]
        ols_u = sm.OLS(y, X_unrestricted).fit()
        
        # Determine maxlags if not provided
        if maxlags is None:
            T = len(y)
            maxlags = int(np.floor(4 * (T/100)**(2/9)))  # Newey-West rule
            print(f"Using automatic maxlags selection: {maxlags} (T={T})")
        
        # Compute HAC covariance matrix using the correct approach
        try:
            # Use the built-in HAC method from statsmodels (most reliable)
            ols_u_hac = ols_u.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': int(maxlags)})
            hac_cov = ols_u_hac.cov_params()
            print(f"Using statsmodels built-in HAC method with maxlags={maxlags}")
        except Exception as e:
            print(f"Error with statsmodels HAC: {e}")
            # Fallback: try cov_hac function
            try:
                hac_cov = cov_hac(ols_u, nlags=int(maxlags))
                print(f"Using cov_hac function with maxlags={maxlags}")
            except Exception as e2:
                print(f"Error with cov_hac: {e2}")
                # Final fallback: try cov_hac_simple
                try:
                    hac_cov = cov_hac_simple(ols_u.resid, ols_u.model.exog, nlags=int(maxlags))
                    print(f"Using cov_hac_simple with maxlags={maxlags}")
                except Exception as e3:
                    print(f"All HAC methods failed: {e3}")
                    return None, None
        
        # Build restriction matrix R to test that all search-term coefficients = 0
        k_params = len(ols_u.params)
        q = len(existing_all_lags)
        R = np.zeros((q, k_params))
        
        # Identify indices of search-term lag columns
        for j, col in enumerate(X_unrestricted.columns):
            if col in existing_all_lags:
                row_idx = existing_all_lags.index(col)
                R[row_idx, j] = 1
        
        # Compute Wald statistic
        beta = ols_u.params.values.reshape(-1, 1)
        R_beta = R @ beta
        R_V_Rt = R @ hac_cov @ R.T
        
        try:
            R_V_Rt_inv = np.linalg.inv(R_V_Rt)
            wald_stat = (R_beta.T @ R_V_Rt_inv @ R_beta)[0, 0]
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            R_V_Rt_inv = np.linalg.pinv(R_V_Rt)
            wald_stat = (R_beta.T @ R_V_Rt_inv @ R_beta)[0, 0]
        
        # Convert to F-statistic
        f_stat = wald_stat / q
        p_val = 1 - f.cdf(f_stat, q, len(y) - k_params)
        
        print(f"Original OLS F-statistic: {F:.4f}, p-value: {p_value:.6f}")
        print(f"HAC-adjusted F-statistic: {f_stat:.4f}, p-value: {p_val:.6f}")
        print(f"Degrees of freedom: ({q}, {len(y) - k_params})")
        print(f"Max lags for HAC: {maxlags}")
        
        # Create a mock results object for consistency with downstream functions
        class HACResults:
            def __init__(self, original_model, hac_cov, f_stat, p_val):
                self.params = original_model.params
                self.bse = np.sqrt(np.diag(hac_cov))
                self.tvalues = self.params / self.bse
                self.pvalues = 2 * (1 - t.cdf(np.abs(self.tvalues), len(original_model.resid) - len(self.params)))
                self.rsquared = original_model.rsquared
                self.fvalue = f_stat
                self.pvalue = p_val
                self.conf_int = lambda: pd.DataFrame({
                    0: self.params - 1.96 * self.bse,
                    1: self.params + 1.96 * self.bse
                }, index=self.params.index)
                # Add summary method for compatibility
                def summary():
                    return "HAC-adjusted results summary"
                self.summary = summary
        
        hac_results = HACResults(ols_u, hac_cov, f_stat, p_val)
        wald_res = {'fvalue': f_stat, 'pvalue': p_val}
        
        return hac_results, wald_res
        
    except Exception as e:
        print(f"Error in HAC analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def perform_autocorrelation_correction(df_flu, existing_flu_lags, existing_all_lags, response_var):
    """Perform Cochrane-Orcutt transformation for autocorrelation correction"""
    print(f"\n=== COCHRANE-ORCUTT TRANSFORMATION ===")
    
    try:
        # Fit initial model
        X_unrestricted = sm.add_constant(df_flu[existing_flu_lags + existing_all_lags])
        y = df_flu[response_var]
        model_initial = sm.OLS(y, X_unrestricted).fit()
        
        # Estimate rho from residuals
        residuals = model_initial.resid
        rho = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        
        print(f"Estimated rho: {rho:.4f}")
        
        # Transform data
        y_transformed = y[1:] - rho * y[:-1]
        X_transformed = X_unrestricted.iloc[1:] - rho * X_unrestricted.iloc[:-1]
        
        # Fit transformed model
        model_corrected = sm.OLS(y_transformed, X_transformed).fit()
        
        print(f"Original model R²: {model_initial.rsquared:.4f}")
        print(f"Corrected model R²: {model_corrected.rsquared:.4f}")
        
        return model_corrected, rho
        
    except Exception as e:
        print(f"Error in Cochrane-Orcutt transformation: {e}")
        return None, None

def perform_hac_sensitivity_analysis(df_flu, existing_flu_lags, existing_all_lags, response_var):
    """Perform HAC sensitivity analysis with different maxlags values"""
    try:
        print(f"\n=== HAC SENSITIVITY ANALYSIS ===")
        
        # Fit unrestricted model
        X_unrestricted = sm.add_constant(df_flu[existing_flu_lags + existing_all_lags])
        y = df_flu[response_var]
        ols_u = sm.OLS(y, X_unrestricted).fit()
        
        # Test different maxlags values
        maxlags_values = [1, 2, 4, 6, 8]
        results = []
        
        for maxlags in maxlags_values:
            try:
                # Compute HAC covariance matrix using the correct approach
                try:
                    # Use the built-in HAC method from statsmodels (most reliable)
                    ols_u_hac = ols_u.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': int(maxlags)})
                    hac_cov = ols_u_hac.cov_params()
                except Exception as e:
                    print(f"Error with statsmodels HAC: {e}")
                    # Fallback: try cov_hac function
                    try:
                        hac_cov = cov_hac(ols_u, nlags=int(maxlags))
                    except Exception as e2:
                        print(f"Error with cov_hac: {e2}")
                        # Final fallback: try cov_hac_simple
                        try:
                            hac_cov = cov_hac_simple(ols_u.resid, ols_u.model.exog, nlags=int(maxlags))
                        except Exception as e3:
                            print(f"All HAC methods failed for maxlags={maxlags}: {e3}")
                            results.append((maxlags, "Error", "Error"))
                            continue
                
                # Build restriction matrix
                k_params = len(ols_u.params)
                q = len(existing_all_lags)
                R = np.zeros((q, k_params))
                
                for j, col in enumerate(X_unrestricted.columns):
                    if col in existing_all_lags:
                        row_idx = existing_all_lags.index(col)
                        R[row_idx, j] = 1
                
                # Compute Wald statistic
                beta = ols_u.params.values.reshape(-1, 1)
                R_beta = R @ beta
                R_V_Rt = R @ hac_cov @ R.T
                
                try:
                    R_V_Rt_inv = np.linalg.inv(R_V_Rt)
                    wald_stat = (R_beta.T @ R_V_Rt_inv @ R_beta)[0, 0]
                except np.linalg.LinAlgError:
                    R_V_Rt_inv = np.linalg.pinv(R_V_Rt)
                    wald_stat = (R_beta.T @ R_V_Rt_inv @ R_beta)[0, 0]
                
                f_stat = wald_stat / q
                p_val = 1 - f.cdf(f_stat, q, len(y) - k_params)
                
                results.append((maxlags, f_stat, p_val))
                print(f"Maxlags {maxlags}: F = {f_stat:.4f}, p = {p_val:.6f}")
                
            except Exception as e:
                print(f"Error for maxlags {maxlags}: {e}")
                import traceback
                traceback.print_exc()
                results.append((maxlags, "Error", "Error"))
        
        # Summary
        significant_results = [r for r in results if r[1] != "Error" and r[2] < 0.05]
        print(f"\nResults summary: {len(significant_results)}/{len(results)} maxlags values show significant results")
        
        if len(significant_results) == len(results):
            print("Recommendation: Results are robust to maxlags choice")
        else:
            print("Recommendation: Results may be sensitive to maxlags choice")
        
        return results
        
    except Exception as e:
        print(f"Error in HAC sensitivity analysis: {e}")
        return None

def analyze_individual_terms(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value):
    """Analyze individual term significance from the multiple regression model"""
    print("\n" + "="*60)
    print("INDIVIDUAL TERM SIGNIFICANCE FROM MULTIPLE REGRESSION")
    print("="*60)
    
    # Extract individual term significance from the unrestricted model
    term_significance = []
    
    # Get the summary of the unrestricted model
    model_summary = model_unrestricted.summary()
    
    # Extract coefficients and p-values for search term lags
    for term in filtered_columns:
        term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
        term_pvals = []
        
        for lag_col in term_lags:
            if lag_col in model_unrestricted.params.index:
                # Get the coefficient and p-value for this lag
                coef = model_unrestricted.params[lag_col]
                try:
                    pval = model_unrestricted.pvalues[lag_col]
                except (IndexError, TypeError):
                    # For HAC results, pvalues might be a numpy array
                    if hasattr(model_unrestricted.pvalues, 'loc'):
                        pval = model_unrestricted.pvalues.loc[lag_col]
                    else:
                        # Find the index position
                        param_index = list(model_unrestricted.params.index).index(lag_col)
                        pval = model_unrestricted.pvalues[param_index]
                term_pvals.append(pval)
        
        if term_pvals:
            # Use the minimum p-value across all lags for this term
            min_p = min(term_pvals)
            term_significance.append((term, min_p))
    
    # Sort by significance
    term_significance.sort(key=lambda x: x[1])
    
    # Calculate Bonferroni-corrected significance threshold
    num_tests = len(term_significance)
    bonferroni_threshold = 0.05 / num_tests if num_tests > 0 else 0.05
    
    # Perform FDR correction (Benjamini-Hochberg)
    if num_tests > 0:
        search_term_pvalues = [pval for term, pval in term_significance]
        fdr_rejected, fdr_pvalues, _, _ = multipletests(search_term_pvalues, method='fdr_bh', alpha=0.05)
        
        # Create mapping of terms to FDR significance
        fdr_significant_terms = set()
        for i, (term, pval) in enumerate(term_significance):
            if fdr_rejected[i]:
                fdr_significant_terms.add(term)
    else:
        fdr_significant_terms = set()
    
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
    
    return term_significance, significant_uncorrected, significant_bonferroni, fdr_significant_terms, bonferroni_threshold

def create_comprehensive_visualization(model_unrestricted, filtered_columns, max_lag, term_significance, 
                                     significant_uncorrected, significant_bonferroni, fdr_significant_terms, 
                                     bonferroni_threshold, response_var):
    """Create comprehensive visualization of individual term significance"""
    print(f"\n=== CREATING VISUALIZATION ===")
    
    # Check for valid p-values (not nan)
    valid_pvals = [(term, pval) for term, pval in term_significance if not np.isnan(pval)]
    
    if not valid_pvals:
        print("\nWARNING: All p-values are NaN. This indicates the multiple regression model failed to fit properly.")
        return
    
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
    plt.close()  # Close the plot instead of showing it
    
    # Print summary statistics
    significant_uncorrected_plot = [term for term, pval in valid_pvals if pval < 0.05]
    significant_bonferroni_plot = [term for term, pval in valid_pvals if pval < bonferroni_threshold]
    significant_fdr_plot = [term for term in valid_terms_plot if term in fdr_significant_terms]
    
    print(f"\nSummary:")
    print(f"Total terms with valid p-values: {len(valid_terms_plot)}")
    print(f"Terms with NaN p-values: {len(term_significance) - len(valid_terms_plot)}")
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

def save_comprehensive_results(term_significance, significant_uncorrected, significant_bonferroni, 
                             fdr_significant_terms, bonferroni_threshold, F, p_value, 
                             model_unrestricted, max_lag, response_var, has_autocorrelation=None, 
                             dw_statistic=None, hac_results=None):
    """Save comprehensive results to a text file"""
    print(f"\n=== SAVING COMPREHENSIVE RESULTS ===")
    
    txt_filename = f"ShiHaoYang/Results/granger_significant_terms_nrevss_lag{max_lag}.txt"
    with open(txt_filename, "w") as f:
        # Write summary at the top
        f.write(f"=== COMPREHENSIVE GRANGER CAUSALITY ANALYSIS SUMMARY ===\n")
        f.write(f"Response variable: {response_var}\n")
        f.write(f"Max lag: {max_lag}\n")
        f.write(f"Number of tests: {len(term_significance)}\n")
        f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
        f.write(f"FDR correction applied (Benjamini-Hochberg method)\n")
        f.write(f"Overall Granger causality F-statistic: {F:.4f}\n")
        f.write(f"Overall Granger causality p-value: {p_value:.6f}\n")
        f.write(f"Model R-squared: {model_unrestricted.rsquared:.4f}\n\n")
        
        # Add autocorrelation information
        f.write(f"=== AUTOCORRELATION ANALYSIS ===\n")
        if has_autocorrelation is not None:
            f.write(f"Autocorrelation detected: {'Yes' if has_autocorrelation else 'No'}\n")
        if dw_statistic is not None:
            f.write(f"Durbin-Watson statistic: {dw_statistic:.4f}\n")
            if dw_statistic < 1.5:
                f.write(f"Interpretation: Strong positive autocorrelation\n")
            elif dw_statistic > 2.5:
                f.write(f"Interpretation: Strong negative autocorrelation\n")
            else:
                f.write(f"Interpretation: No significant autocorrelation\n")
        if hac_results is not None:
            f.write(f"HAC-adjusted F-statistic: {hac_results['fvalue']:.4f}\n")
            f.write(f"HAC-adjusted p-value: {hac_results['pvalue']:.6f}\n")
        f.write(f"\n")
        
        f.write(f"=== SIGNIFICANCE SUMMARY ===\n")
        f.write(f"Uncorrected significant (p < 0.05): {len(significant_uncorrected)} terms\n")
        f.write(f"Bonferroni significant (p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)} terms\n")
        f.write(f"FDR significant: {len(fdr_significant_terms)} terms\n\n")
        
        # Get all significant terms (any type) and determine most conservative significance
        all_significant_terms = set()
        for term, pval in term_significance:
            if pval < 0.05:  # Any type of significance
                all_significant_terms.add(term)
        
        if all_significant_terms:
            f.write(f"=== ALL SIGNIFICANT TERMS (n={len(all_significant_terms)}) ===\n")
            f.write(f"Term\tMin_p_value\tMost_Conservative_Significance\n")
            
            # Sort by p-value (most significant first)
            significant_terms_sorted = []
            for term in all_significant_terms:
                pval = [p for t, p in term_significance if t == term][0]
                # Determine most conservative significance
                if pval < bonferroni_threshold:
                    most_conservative = "Bonferroni"
                elif term in fdr_significant_terms:
                    most_conservative = "FDR"
                else:
                    most_conservative = "Uncorrected"
                
                significant_terms_sorted.append((term, pval, most_conservative))
            
            # Sort by p-value
            significant_terms_sorted.sort(key=lambda x: x[1])
            
            for term, pval, most_conservative in significant_terms_sorted:
                f.write(f"{term}\t{pval:.6f}\t{most_conservative}\n")
        else:
            f.write("No terms were significant at any level.\n")
    
    print(f"Comprehensive results saved to {txt_filename}")

def save_detailed_significant_terms(model_unrestricted, filtered_columns, max_lag, response_var, test_type="NREVSS"):
    """Save detailed information about Bonferroni-significant terms with their specific lag information"""
    print(f"\n=== SAVING DETAILED SIGNIFICANT TERMS ===")
    
    detailed_filename = f"ShiHaoYang/Results/detailed_significant_terms_{test_type.lower()}_lag{max_lag}.txt"
    
    with open(detailed_filename, "w") as f:
        f.write(f"=== DETAILED SIGNIFICANT TERMS ANALYSIS ===\n")
        f.write(f"Test Type: {test_type}\n")
        f.write(f"Response Variable: {response_var}\n")
        f.write(f"Max Lag: {max_lag}\n")
        f.write(f"Model R-squared: {model_unrestricted.rsquared:.4f}\n\n")
        
        f.write(f"Format: Term\tLag\tCoefficient\tP-value\tSignificance_Level\n")
        f.write(f"{'='*80}\n\n")
        
        # Track all significant terms
        all_significant_terms = []
        
        # Calculate Bonferroni threshold
        num_tests = len(filtered_columns) * max_lag
        bonferroni_threshold = 0.05 / num_tests if num_tests > 0 else 0.05
        
        # Extract individual term significance from the unrestricted model
        for term in filtered_columns:
            term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
            
            for lag_col in term_lags:
                if lag_col in model_unrestricted.params.index:
                    # Get the coefficient and p-value for this lag
                    coef = model_unrestricted.params[lag_col]
                    try:
                        pval = model_unrestricted.pvalues[lag_col]
                    except (IndexError, TypeError):
                        # For HAC results, pvalues might be a numpy array
                        if hasattr(model_unrestricted.pvalues, 'loc'):
                            pval = model_unrestricted.pvalues.loc[lag_col]
                        else:
                            # Find the index position
                            param_index = list(model_unrestricted.params.index).index(lag_col)
                            pval = model_unrestricted.pvalues[param_index]
                    
                    # Determine significance level
                    if pval < 0.001:
                        significance = "***"
                    elif pval < 0.01:
                        significance = "**"
                    elif pval < 0.05:
                        significance = "*"
                    else:
                        significance = ""
                    
                    # Extract lag number from column name
                    lag_num = lag_col.split('_lag')[1]
                    
                    # Only include if Bonferroni significant
                    if pval < bonferroni_threshold:
                        all_significant_terms.append((term, lag_num, coef, pval, significance))
                        f.write(f"{term}\t{lag_num}\t{coef:.6f}\t{pval:.6f}\t{significance}\n")
        
        # Summary statistics
        f.write(f"\n{'='*80}\n")
        f.write(f"SUMMARY STATISTICS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
        f.write(f"Total Bonferroni significant terms: {len(all_significant_terms)}\n")
        f.write(f"Terms with p < 0.001: {len([t for t in all_significant_terms if t[3] < 0.001])}\n")
        f.write(f"Terms with p < 0.01: {len([t for t in all_significant_terms if t[3] < 0.01])}\n")
        f.write(f"Terms with p < Bonferroni threshold: {len([t for t in all_significant_terms if t[3] < bonferroni_threshold])}\n\n")
        
        # Group by term and show all lags
        f.write(f"TERMS BY SIGNIFICANCE (ALL LAGS COMBINED):\n")
        f.write(f"{'='*80}\n")
        
        term_summary = {}
        for term, lag, coef, pval, sig in all_significant_terms:
            if term not in term_summary:
                term_summary[term] = []
            term_summary[term].append((lag, coef, pval, sig))
        
        # Sort terms by their minimum p-value
        sorted_terms = sorted(term_summary.items(), 
                            key=lambda x: min([lag_info[2] for lag_info in x[1]]))
        
        for term, lag_info in sorted_terms:
            min_pval = min([lag_info[2] for lag_info in lag_info])
            f.write(f"\n{term} (min p-value: {min_pval:.6f}):\n")
            for lag, coef, pval, sig in sorted(lag_info, key=lambda x: int(x[0])):
                f.write(f"  Lag {lag}: coef={coef:.6f}, p={pval:.6f} {sig}\n")
    
    print(f"Detailed significant terms saved to {detailed_filename}")
    return detailed_filename

def main():
    """Main function to run the complete Granger causality analysis"""
    print("=== NREVSS GRANGER CAUSALITY ANALYSIS ===")
    
    # Configuration
    max_terms = None  # Maximum number of search terms to use (set to None to use all)
    response_var = 'flu_total_positive'  # Or 'flu_pct_positive' for percent positive
    
    # Load and prepare data
    df_flu, df_search = load_and_prepare_data()
    
    # Perform data diagnostics
    filtered_columns = perform_data_diagnostics(df_search)
    
    # Use filtered columns for the analysis
    search_terms = filtered_columns[:max_terms] if max_terms else filtered_columns
    print(f"Final number of search terms to use: {len(search_terms)}")
    
    # Run analysis for max lags 1-5
    for max_lag in range(1, 6):
        print(f"\n{'='*80}")
        print(f"RUNNING ANALYSIS FOR MAX LAG = {max_lag}")
        print(f"{'='*80}")
        
        # Create a fresh copy of the data for each iteration
        df_flu_fresh = df_flu.copy()
        
        # Prepare merged data
        df_flu_fresh, existing_flu_lags, existing_all_lags, search_terms_simple = prepare_merged_data(
            df_flu_fresh, df_search, search_terms, max_lag, response_var
        )
    
        # Check for constant values in response variable
        if df_flu_fresh[response_var].nunique() <= 1:
            print(f"Error: Response variable '{response_var}' has constant values")
            continue
        
        # Check for constant values in search terms
        constant_terms = []
        for term in search_terms_simple:
            if term in df_flu_fresh.columns and df_flu_fresh[term].nunique() <= 1:
                constant_terms.append(term)
        
        if constant_terms:
            print(f"Warning: The following terms have constant values and will be excluded: {constant_terms}")
            # Remove constant terms from existing_all_lags
            existing_all_lags = [col for col in existing_all_lags 
                                if not any(term in col for term in constant_terms)]
        
        print(f"Final search term lag columns: {len(existing_all_lags)}")
        
        # Perform Granger causality test
        model_restricted, model_unrestricted, F, p_value, X_unrestricted, y = perform_granger_causality_test(
            df_flu_fresh, existing_flu_lags, existing_all_lags, response_var
        )
        
        if model_unrestricted is None:
            print("Granger causality test failed. Continuing to next lag.")
            continue
        
        # Detect autocorrelation
        has_autocorrelation = detect_autocorrelation(model_unrestricted)
        dw_statistic = sm.stats.durbin_watson(model_unrestricted.resid)
        
        # If autocorrelation is detected, perform corrections
        if has_autocorrelation:
            print(f"\n⚠️  Autocorrelation detected! Performing corrections...")
            
            # Method 1: HAC standard errors (corrected approach)
            ols_u_hac, wald_res = perform_hac_analysis(
                df_flu_fresh, existing_flu_lags, existing_all_lags, response_var, F, p_value, maxlags=None
            )
            
            # Method 2: Cochrane-Orcutt transformation
            model_corrected, rho = perform_autocorrelation_correction(
                df_flu_fresh, existing_flu_lags, existing_all_lags, response_var
            )
            
            # Method 3: HAC sensitivity analysis
            hac_sensitivity_results = perform_hac_sensitivity_analysis(
                df_flu_fresh, existing_flu_lags, existing_all_lags, response_var
            )
            
            # Use HAC results for individual term analysis if available
            if ols_u_hac is not None:
                print(f"\nUsing HAC-adjusted results for individual term analysis...")
                analyze_individual_terms(ols_u_hac, search_terms_simple, max_lag, response_var, 
                                       wald_res['fvalue'], wald_res['pvalue'])
                # Save detailed significant terms with HAC results
                save_detailed_significant_terms(ols_u_hac, search_terms_simple, max_lag, response_var, "NREVSS")
            else:
                print(f"\nHAC analysis failed, using original results...")
                analyze_individual_terms(model_unrestricted, search_terms_simple, max_lag, response_var, F, p_value)
                # Save detailed significant terms with original results
                save_detailed_significant_terms(model_unrestricted, search_terms_simple, max_lag, response_var, "NREVSS")
        else:
            # No autocorrelation detected, use original results
            print(f"\n✅ No autocorrelation detected. Using standard results.")
            analyze_individual_terms(model_unrestricted, search_terms_simple, max_lag, response_var, F, p_value)
            # Save detailed significant terms with original results
            save_detailed_significant_terms(model_unrestricted, search_terms_simple, max_lag, response_var, "NREVSS")
        
        # Perform individual term analysis
        term_significance, significant_uncorrected, significant_bonferroni, fdr_significant_terms, bonferroni_threshold = analyze_individual_terms(
            model_unrestricted, search_terms_simple, max_lag, response_var, F, p_value
        )
        
        # Create visualization
        create_comprehensive_visualization(
            model_unrestricted, search_terms_simple, max_lag, term_significance,
            significant_uncorrected, significant_bonferroni, fdr_significant_terms,
            bonferroni_threshold, response_var
        )
        
        # Save comprehensive results
        save_comprehensive_results(
            term_significance, significant_uncorrected, significant_bonferroni,
            fdr_significant_terms, bonferroni_threshold, F, p_value,
            model_unrestricted, max_lag, response_var, has_autocorrelation, 
            dw_statistic=dw_statistic, hac_results=wald_res if ols_u_hac is not None else None
        )
        
        print(f"\n=== ANALYSIS COMPLETE FOR MAX LAG = {max_lag} ===")
    
    print(f"\n=== ALL ANALYSES COMPLETE ===")

if __name__ == "__main__":
    main()