import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac_simple, cov_hac


def load_and_prepare_data():
    """Load and prepare the ILI and search trends data"""
    # Load data
    df_ili = pd.read_csv("ShiHaoYang/Data/ILINet_all.csv", skiprows=1)
    df_search = pd.read_csv("ShiHaoYang/Data/flu_trends_regression_dataset.csv")
    
    # Create common columns for merging
    df_search['Week'] = pd.to_datetime(df_search['date'])
    df_search['YEAR'] = df_search['Week'].dt.isocalendar().year
    df_search['WEEK'] = df_search['Week'].dt.isocalendar().week
    
    return df_ili, df_search


def perform_diagnostic_analysis(df_search):
    """Perform diagnostic analysis on search terms data"""
    search_terms = [
        col for col in df_search.columns
        if col not in ['date', 'Week', 'YEAR', 'WEEK']
        and np.issubdtype(df_search[col].dtype, np.number)
    ]
    
    # Identify problematic columns
    constant_columns = [col for col in search_terms if df_search[col].nunique() == 1]
    low_variance_columns = [col for col in search_terms if df_search[col].std() < 0.1]
    zero_dominant_columns = [col for col in search_terms if (df_search[col] == 0).mean() > 0.8]
    
    # Filter out problematic columns
    filtered_columns = [
        col for col in search_terms
        if col not in constant_columns
        and col not in low_variance_columns
        and col not in zero_dominant_columns
    ]
    
    print(f"Data diagnostics: {len(constant_columns)} constant, {len(low_variance_columns)} low variance, {len(zero_dominant_columns)} zero-dominant columns removed")
    print(f"Final search terms: {len(filtered_columns)}")
    
    return filtered_columns


def merge_and_prepare_data(df_ili, df_search, filtered_columns):
    """Merge data and create lagged variables"""
    # Merge data
    df_ili['YEAR'] = df_ili['YEAR'].astype(int)
    df_ili['WEEK'] = df_ili['WEEK'].astype(int)
    df_ili = pd.merge(
        df_ili,
        df_search[['YEAR', 'WEEK'] + filtered_columns],
        on=['YEAR', 'WEEK'],
        how='left'
    )
    
    # Create lagged variables
    max_lag = 1
    response_var = '% WEIGHTED ILI'
    
    flu_lags = []
    all_lags = []
    for lag in range(1, max_lag + 1):
        df_ili[f'ili_lag{lag}'] = df_ili[response_var].shift(lag)
        flu_lags.append(f'ili_lag{lag}')
        for term in filtered_columns:
            lag_col = f'{term}_lag{lag}'
            df_ili[lag_col] = df_ili[term].shift(lag)
            all_lags.append(lag_col)
    
    # Clean data
    df_ili = df_ili.dropna()
    df_ili = df_ili.loc[:, (df_ili != 0).any(axis=0)]
    
    existing_flu_lags = [col for col in flu_lags if col in df_ili.columns]
    existing_all_lags = [col for col in all_lags if col in df_ili.columns]
    
    print(f"Final dataset: {len(df_ili)} observations, {len(existing_all_lags)} search term lags")
    
    return df_ili, existing_flu_lags, existing_all_lags, response_var, max_lag


def perform_granger_causality_test(df_ili, existing_flu_lags, existing_all_lags, response_var):
    """Perform multiple regression Granger causality test"""
    try:
        # Fit restricted model (only ILI lags)
        X_restricted = sm.add_constant(df_ili[existing_flu_lags])
        y = df_ili[response_var]
        model_restricted = sm.OLS(y, X_restricted).fit()
        
        # Fit unrestricted model (ILI lags + search term lags)
        X_unrestricted = sm.add_constant(df_ili[existing_flu_lags + existing_all_lags])
        model_unrestricted = sm.OLS(y, X_unrestricted).fit()
        
        # Calculate F-statistic
        rss_restricted = np.sum(model_restricted.resid ** 2)
        rss_unrestricted = np.sum(model_unrestricted.resid ** 2)
        df1 = len(existing_all_lags)
        df2 = len(df_ili) - X_unrestricted.shape[1]
        
        if df1 > 0 and df2 > 0 and rss_unrestricted > 0:
            F = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
            p_value = 1 - f.cdf(F, df1, df2)
            
            print(f"\n=== GRANGER CAUSALITY RESULTS ===")
            print(f"F-statistic: {F:.4f}, p-value: {p_value:.6f}")
            print(f"R² improvement: {model_unrestricted.rsquared - model_restricted.rsquared:.4f}")
            
            return model_restricted, model_unrestricted, F, p_value, df1, df2
        else:
            print("Error: Cannot compute F-statistic")
            return None, None, None, None, None, None
            
    except Exception as e:
        print(f"Error in Granger causality test: {e}")
        return None, None, None, None, None, None


def perform_autocorrelation_analysis(model_restricted, model_unrestricted):
    """Perform Durbin-Watson test for autocorrelation"""
    try:
        # Calculate Durbin-Watson statistics
        dw_restricted = durbin_watson(model_restricted.resid)
        dw_unrestricted = durbin_watson(model_unrestricted.resid)
        
        print(f"\n=== AUTOCORRELATION TEST ===")
        print(f"Durbin-Watson (restricted): {dw_restricted:.3f}, (unrestricted): {dw_unrestricted:.3f}")
        
        # Assess autocorrelation
        if 1.5 <= dw_unrestricted <= 2.5:
            print("✓ No significant autocorrelation detected")
            return dw_restricted, dw_unrestricted, False
        else:
            if dw_unrestricted < 1.5:
                print("⚠️  Positive autocorrelation detected - may inflate F-statistics")
            else:
                print("⚠️  Negative autocorrelation detected - may deflate F-statistics")
            return dw_restricted, dw_unrestricted, True
        
    except Exception as e:
        print(f"Error in Durbin-Watson test: {e}")
        return None, None, None


def analyze_individual_terms(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value):
    """Analyze individual term significance within the comprehensive model"""
    # Get coefficient results for all terms in the unrestricted model
    coef_results = pd.DataFrame({
        'Coefficient': model_unrestricted.params,
        'Std Error': model_unrestricted.bse,
        't-value': model_unrestricted.tvalues,
        'p-value': model_unrestricted.pvalues,
        'CI Lower': model_unrestricted.conf_int()[0],
        'CI Upper': model_unrestricted.conf_int()[1]
    })
    
    # Sort by p-value (most significant first)
    coef_results = coef_results.sort_values('p-value')
    
    # Calculate Bonferroni-corrected significance threshold
    # Count search term lags (excluding ILI lags and intercept)
    search_term_lags = [col for col in coef_results.index if not col.startswith('ili_lag') and col != 'const']
    num_tests = len(search_term_lags)
    bonferroni_threshold = 0.05 / num_tests if num_tests > 0 else 0.05
    
    # Perform FDR correction (Benjamini-Hochberg)
    if num_tests > 0:
        search_term_pvalues = coef_results.loc[search_term_lags, 'p-value'].values
        fdr_rejected, fdr_pvalues, _, _ = multipletests(search_term_pvalues, method='fdr_bh', alpha=0.05)
        
        # Create mapping of terms to FDR significance
        fdr_significant_terms = set()
        for i, term in enumerate(search_term_lags):
            if fdr_rejected[i]:
                fdr_significant_terms.add(term)
    else:
        fdr_significant_terms = set()
    
    print(f"\n=== INDIVIDUAL TERM ANALYSIS ===")
    print(f"Number of search term lags tested: {num_tests}")
    print(f"Bonferroni-corrected significance threshold: {bonferroni_threshold:.6f} (0.05/{num_tests})")
    print(f"FDR correction applied (Benjamini-Hochberg method)")
    
    # Identify significant terms with different thresholds
    significant_uncorrected = coef_results[coef_results['p-value'] < 0.05]
    significant_uncorrected = significant_uncorrected[significant_uncorrected.index != 'const']
    
    significant_bonferroni = coef_results[coef_results['p-value'] < bonferroni_threshold]
    significant_bonferroni = significant_bonferroni[significant_bonferroni.index != 'const']
    
    significant_fdr = coef_results[coef_results.index.isin(fdr_significant_terms)]
    
    print(f"Significant terms (uncorrected p < 0.05): {len(significant_uncorrected)}")
    print(f"Significant terms (Bonferroni-corrected p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)}")
    print(f"Significant terms (FDR-corrected): {len(significant_fdr)}")
    
    # Use FDR-corrected results as primary (less conservative)
    significant_in_model = significant_fdr if len(significant_fdr) > 0 else significant_bonferroni
    
    if len(significant_in_model) > 0:
        # Categorize significant terms
        significant_ili_lags = [term for term in significant_in_model.index if term.startswith('ili_lag')]
        significant_search_lags = [term for term in significant_in_model.index if not term.startswith('ili_lag')]
        
        print(f"Breakdown: {len(significant_ili_lags)} ILI lags, {len(significant_search_lags)} search term lags")
        
        # Save comprehensive results to file
        comprehensive_txt_filename = f"ShiHaoYang/Results/comprehensive_granger_individual_significant_terms_ili_lag{max_lag}.txt"
        with open(comprehensive_txt_filename, "w") as f:
            # Write summary at the top
            f.write(f"=== COMPREHENSIVE GRANGER CAUSALITY ANALYSIS SUMMARY ===\n")
            f.write(f"Response variable: {response_var}\n")
            f.write(f"Max lag: {max_lag}\n")
            f.write(f"Number of tests: {num_tests}\n")
            f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
            f.write(f"Overall Granger causality F-statistic: {F:.4f}\n")
            f.write(f"Overall Granger causality p-value: {p_value:.6f}\n")
            f.write(f"Model R-squared: {model_unrestricted.rsquared:.4f}\n\n")
            
            f.write(f"=== SIGNIFICANCE SUMMARY ===\n")
            f.write(f"Uncorrected significant (p < 0.05): {len(significant_uncorrected)} terms\n")
            f.write(f"Bonferroni significant (p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)} terms\n")
            f.write(f"FDR significant: {len(significant_fdr)} terms\n\n")
            
            # Get all significant terms (any type) and determine most conservative significance
            all_significant_terms = set()
            for term in coef_results.index:
                if term != 'const' and not term.startswith('ili_lag'):
                    pval = coef_results.loc[term, 'p-value']
                    if pval < 0.05:  # Any type of significance
                        all_significant_terms.add(term)
            
            if all_significant_terms:
                f.write(f"=== ALL SIGNIFICANT TERMS (n={len(all_significant_terms)}) ===\n")
                f.write(f"Term\tMin_p_value\tMost_Conservative_Significance\n")
                
                # Sort by p-value (most significant first)
                significant_terms_sorted = []
                for term in all_significant_terms:
                    row = coef_results.loc[term]
                    # Determine most conservative significance
                    if term in fdr_significant_terms:
                        most_conservative = "FDR"
                    elif row['p-value'] < bonferroni_threshold:
                        most_conservative = "Bonferroni"
                    else:
                        most_conservative = "Uncorrected"
                    
                    significant_terms_sorted.append((term, row['p-value'], most_conservative))
                
                # Sort by p-value
                significant_terms_sorted.sort(key=lambda x: x[1])
                
                for term, pval, most_conservative in significant_terms_sorted:
                    f.write(f"{term}\t{pval:.6f}\t{most_conservative}\n")
            else:
                f.write("No terms were significant at any level.\n")
        
        print(f"Detailed results saved to {comprehensive_txt_filename}")
        
        # Create visualization
        create_comprehensive_visualization(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value, bonferroni_threshold, fdr_significant_terms)
        
    else:
        print("No terms were significant after multiple testing correction.")
        
        # Still save comprehensive results even if no FDR/Bonferroni significant terms
        if len(significant_uncorrected) > 0:
            print(f"Note: {len(significant_uncorrected)} terms were significant at uncorrected p < 0.05")
            comprehensive_txt_filename = f"ShiHaoYang/Results/comprehensive_granger_individual_significant_terms_ili_lag{max_lag}.txt"
            with open(comprehensive_txt_filename, "w") as f:
                # Write summary at the top
                f.write(f"=== COMPREHENSIVE GRANGER CAUSALITY ANALYSIS SUMMARY ===\n")
                f.write(f"Response variable: {response_var}\n")
                f.write(f"Max lag: {max_lag}\n")
                f.write(f"Number of tests: {num_tests}\n")
                f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
                f.write(f"Overall Granger causality F-statistic: {F:.4f}\n")
                f.write(f"Overall Granger causality p-value: {p_value:.6f}\n")
                f.write(f"Model R-squared: {model_unrestricted.rsquared:.4f}\n\n")
                
                f.write(f"=== SIGNIFICANCE SUMMARY ===\n")
                f.write(f"Uncorrected significant (p < 0.05): {len(significant_uncorrected)} terms\n")
                f.write(f"Bonferroni significant (p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)} terms\n")
                f.write(f"FDR significant: {len(significant_fdr)} terms\n\n")
                
                f.write(f"=== ALL SIGNIFICANT TERMS (n={len(significant_uncorrected)}) ===\n")
                f.write(f"Term\tMin_p_value\tMost_Conservative_Significance\n")
                
                # Sort by p-value (most significant first)
                significant_terms_sorted = []
                for term, row in significant_uncorrected.iterrows():
                    # Determine most conservative significance
                    if term in fdr_significant_terms:
                        most_conservative = "FDR"
                    elif row['p-value'] < bonferroni_threshold:
                        most_conservative = "Bonferroni"
                    else:
                        most_conservative = "Uncorrected"
                    
                    significant_terms_sorted.append((term, row['p-value'], most_conservative))
                
                # Sort by p-value
                significant_terms_sorted.sort(key=lambda x: x[1])
                
                for term, pval, most_conservative in significant_terms_sorted:
                    f.write(f"{term}\t{pval:.6f}\t{most_conservative}\n")
            
            print(f"Comprehensive results saved to {comprehensive_txt_filename}")


def create_comprehensive_visualization(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value, bonferroni_threshold, fdr_significant_terms):
    """Create visualization for comprehensive model results"""
    # Get minimum p-values for each search term
    term_min_pvalues = []
    for term in filtered_columns:
        term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
        term_pvals = []
        for lag_col in term_lags:
            if lag_col in model_unrestricted.params.index:
                # Get p-value for this lag
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
            min_p = min(term_pvals)
            term_min_pvalues.append((term, min_p))
    
    # Sort by p-value (most significant first)
    term_min_pvalues.sort(key=lambda x: x[1])
    valid_terms = [term for term, pval in term_min_pvalues if not np.isnan(pval)]
    granger_pvals = [pval for term, pval in term_min_pvalues if not np.isnan(pval)]
    
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

        # Create bars with colors for different significance levels
        colors = []
        for i, term in enumerate(valid_terms):
            pval = granger_pvals[i]
            # Check if any lag of this term is FDR significant
            term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
            is_fdr_sig = any(lag in fdr_significant_terms for lag in term_lags)
            
            if is_fdr_sig:
                colors.append('purple')  # FDR significant
            elif pval < bonferroni_threshold:
                colors.append('darkred')  # Bonferroni significant
            elif pval < 0.05:
                colors.append('red')      # Uncorrected significant
            else:
                colors.append('orange')   # Not significant
        
        bars = plt.bar(valid_terms, granger_pvals, color=colors, alpha=0.7)

        plt.ylabel('Min p-value (across lags)', fontsize=12)
        plt.title(f'Individual Term Significance from Comprehensive Granger Causality Model\nResponse: {response_var}, Max Lag: {max_lag}', fontsize=14, pad=20)
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
        for bar, pval in zip(bars, granger_pvals):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{pval:.3f}', ha='center', va='bottom', fontsize=value_font_size, rotation=90)

        # Set y-axis limits to ensure visibility
        if granger_pvals:
            plt.ylim(0, max(granger_pvals) * 1.1)

        plt.legend(handles=legend_elements, fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # Better layout with more space
        plt.tight_layout(pad=2.0)
        plt.savefig(f"ShiHaoYang/Results/granger_pvalues_comprehensive_model_ili_lag{max_lag}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary statistics
        significant_uncorrected = [term for term, pval in zip(valid_terms, granger_pvals) if pval < 0.05]
        significant_bonferroni = [term for term, pval in zip(valid_terms, granger_pvals) if pval < bonferroni_threshold]
        
        # Count FDR significant terms
        significant_fdr = []
        for term in valid_terms:
            term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
            if any(lag in fdr_significant_terms for lag in term_lags):
                significant_fdr.append(term)
        
        print(f"Visualization summary:")
        print(f"  Uncorrected (p < 0.05): {len(significant_uncorrected)}/{len(valid_terms)} terms")
        print(f"  Bonferroni-corrected (p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)}/{len(valid_terms)} terms")
        print(f"  FDR-corrected: {len(significant_fdr)}/{len(valid_terms)} terms")

        if significant_fdr:
            print(f"Top 5 FDR-significant terms:")
            sorted_valid = sorted(zip(valid_terms, granger_pvals), key=lambda x: x[1])
            for i, (term, pval) in enumerate(sorted_valid[:5]):
                term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
                is_fdr_sig = any(lag in fdr_significant_terms for lag in term_lags)
                significance = "***" if is_fdr_sig else "**" if pval < 0.05 else ""
                print(f"  {i+1}. {term}: p = {pval:.4f}{significance}")
        elif significant_bonferroni:
            print(f"Top 5 Bonferroni-significant terms:")
            sorted_valid = sorted(zip(valid_terms, granger_pvals), key=lambda x: x[1])
            for i, (term, pval) in enumerate(sorted_valid[:5]):
                significance = "***" if pval < bonferroni_threshold else "**" if pval < 0.05 else ""
                print(f"  {i+1}. {term}: p = {pval:.4f}{significance}")
        elif significant_uncorrected:
            print(f"Top 5 uncorrected-significant terms:")
            sorted_valid = sorted(zip(valid_terms, granger_pvals), key=lambda x: x[1])
            for i, (term, pval) in enumerate(sorted_valid[:5]):
                print(f"  {i+1}. {term}: p = {pval:.4f}")
    else:
        print("No valid terms found for plotting")


def perform_hac_analysis(df_ili, existing_flu_lags, existing_all_lags, response_var, F, p_value, maxlags=None):
    """Perform HAC analysis using direct covariance matrix computation"""
    try:
        print(f"\n=== HAC ANALYSIS (Autocorrelation-Adjusted) ===")
        
        # Fit unrestricted model with OLS (get point estimates)
        X_unrestricted = sm.add_constant(df_ili[existing_flu_lags + existing_all_lags])
        y = df_ili[response_var]
        
        ols_u = sm.OLS(y, X_unrestricted).fit()
        
        # Determine optimal maxlags if not provided
        if maxlags is None:
            # Use Newey-West automatic lag selection rule
            # Rule: maxlags = floor(4*(T/100)^(2/9)) where T is sample size
            T = len(y)
            maxlags = int(np.floor(4 * (T/100)**(2/9)))
            print(f"Automatic maxlags selection: {maxlags} (based on sample size T={T})")
        
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
                    print(f"Error with cov_hac_simple: {e3}")
                    return None, None
        
        # Build restriction matrix R to test that all search-term coefficients = 0
        k_params = len(ols_u.params)
        q = len(existing_all_lags)
        R = np.zeros((q, k_params))
        
        # Identify indices of search-term lag columns
        search_indices = []
        for j, col in enumerate(X_unrestricted.columns):
            if col in existing_all_lags:
                row_idx = existing_all_lags.index(col)
                R[row_idx, j] = 1
                search_indices.append(j)
        
        # Compute Wald statistic manually using HAC covariance
        beta = ols_u.params.values
        Rbeta = R @ beta
        RVR = R @ hac_cov @ R.T
        
        try:
            wald_stat = Rbeta.T @ np.linalg.solve(RVR, Rbeta)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in Wald test computation, using pseudo-inverse")
            wald_stat = Rbeta.T @ np.linalg.pinv(RVR) @ Rbeta
        
        # Convert to F-statistic
        f_stat_hac = wald_stat / q
        df_num = q
        df_denom = len(y) - k_params
        
        # Compute p-value
        p_val_hac = 1 - f.cdf(f_stat_hac, df_num, df_denom)
        
        print(f"Original OLS F-statistic: {F:.4f}, p-value: {p_value:.6f}")
        print(f"HAC-adjusted F-statistic: {f_stat_hac:.4f}, p-value: {p_val_hac:.6f}")
        print(f"Degrees of freedom: ({df_num}, {df_denom})")
        print(f"Max lags for HAC: {maxlags}")
        
        # Create a mock results object for consistency with downstream functions
        from scipy.stats import t
        class HACResults:
            def __init__(self, original_model, hac_cov, f_stat, p_val):
                self.params = original_model.params
                self.bse = np.sqrt(np.diag(hac_cov))
                self.tvalues = self.params / self.bse
                df = len(original_model.resid) - len(self.params)
                self.pvalues = 2 * (1 - t.cdf(np.abs(self.tvalues), df))
                self.rsquared = original_model.rsquared
                self.fvalue = f_stat
                self.pvalue = p_val
                self.conf_int = lambda: pd.DataFrame({
                    0: self.params - 1.96 * self.bse,
                    1: self.params + 1.96 * self.bse
                }, index=self.params.index)
        
        hac_results = HACResults(ols_u, hac_cov, f_stat_hac, p_val_hac)
        
        return hac_results, {'fvalue': f_stat_hac, 'pvalue': p_val_hac, 'df_num': df_num, 'df_denom': df_denom}
        
    except Exception as e:
        print(f"Error in HAC analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def perform_autocorrelation_correction(df_ili, existing_flu_lags, existing_all_lags, response_var):
    """Perform autocorrelation correction using Cochrane-Orcutt or HAC"""
    try:
        print(f"\n=== AUTOCORRELATION CORRECTION ===")
        
        # Method 1: Cochrane-Orcutt transformation
        print("Method 1: Cochrane-Orcutt Transformation")
        X_unrestricted = sm.add_constant(df_ili[existing_flu_lags + existing_all_lags])
        y = df_ili[response_var]
        
        # Fit initial model
        model_initial = sm.OLS(y, X_unrestricted).fit()
        
        # Estimate AR(1) coefficient from residuals
        residuals = model_initial.resid
        rho = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        
        print(f"Estimated AR(1) coefficient (rho): {rho:.4f}")
        
        # Apply Cochrane-Orcutt transformation
        y_transformed = y[1:] - rho * y[:-1]
        X_transformed = X_unrestricted.iloc[1:] - rho * X_unrestricted.iloc[:-1]
        
        # Check for infinite or NaN values in transformed data
        if np.any(np.isinf(X_transformed.values)) or np.any(np.isnan(X_transformed.values)):
            print("Warning: Infinite or NaN values in transformed X matrix")
            mask = ~(np.isinf(X_transformed.values).any(axis=1) | np.isnan(X_transformed.values).any(axis=1))
            X_transformed = X_transformed[mask]
            y_transformed = y_transformed[mask]
            print(f"Removed {np.sum(~mask)} rows with infinite/NaN values from transformed X")
        
        # Fit transformed model
        model_corrected = sm.OLS(y_transformed, X_transformed).fit()
        
        print(f"Original R²: {model_initial.rsquared:.4f}")
        print(f"Corrected R²: {model_corrected.rsquared:.4f}")
        
        return model_corrected, rho
        
    except Exception as e:
        print(f"Error in autocorrelation correction: {e}")
        return None, None


def perform_hac_sensitivity_analysis(df_ili, existing_flu_lags, existing_all_lags, response_var):
    """Perform HAC sensitivity analysis with different maxlags values"""
    try:
        print(f"\n=== HAC SENSITIVITY ANALYSIS ===")
        
        # Fit unrestricted model
        X_unrestricted = sm.add_constant(df_ili[existing_flu_lags + existing_all_lags])
        y = df_ili[response_var]
        
        ols_u = sm.OLS(y, X_unrestricted).fit()
        
        # Build restriction matrix
        k_params = len(ols_u.params)
        q = len(existing_all_lags)
        R = np.zeros((q, k_params))
        
        for j, col in enumerate(X_unrestricted.columns):
            if col in existing_all_lags:
                row_idx = existing_all_lags.index(col)
                R[row_idx, j] = 1
        
        # Test different maxlags values
        maxlags_values = [1, 2, 4, 6, 8]
        results = []
        
        print(f"{'Maxlags':<8} {'F-statistic':<12} {'p-value':<12} {'Significant':<12}")
        print("-" * 50)
        
        for maxlags in maxlags_values:
            try:
                # Compute HAC covariance matrix using the correct approach
                try:
                    # Use the built-in HAC method from statsmodels (most reliable)
                    ols_u_hac = ols_u.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': int(maxlags)})
                    hac_cov = ols_u_hac.cov_params()
                except Exception as e:
                    # Fallback: try cov_hac function
                    try:
                        hac_cov = cov_hac(ols_u, nlags=int(maxlags))
                    except:
                        # Final fallback: try cov_hac_simple
                        hac_cov = cov_hac_simple(ols_u.resid, ols_u.model.exog, nlags=int(maxlags))
                
                # Compute Wald statistic manually
                beta = ols_u.params.values
                Rbeta = R @ beta
                RVR = R @ hac_cov @ R.T
                
                try:
                    wald_stat = Rbeta.T @ np.linalg.solve(RVR, Rbeta)
                except np.linalg.LinAlgError:
                    wald_stat = Rbeta.T @ np.linalg.pinv(RVR) @ Rbeta
                
                # Convert to F-statistic
                f_stat_hac = wald_stat / q
                df_num = q
                df_denom = len(y) - k_params
                p_val_hac = 1 - f.cdf(f_stat_hac, df_num, df_denom)
                
                significant = "Yes" if p_val_hac < 0.05 else "No"
                print(f"{maxlags:<8} {f_stat_hac:<12.4f} {p_val_hac:<12.6f} {significant:<12}")
                
                results.append({
                    'maxlags': maxlags,
                    'f_statistic': f_stat_hac,
                    'p_value': p_val_hac,
                    'significant': p_val_hac < 0.05
                })
                
            except Exception as e:
                print(f"{maxlags:<8} {'Error':<12} {'Error':<12} {'Error':<12}")
                print(f"        Error details: {e}")
        
        # Summary
        significant_count = sum(1 for r in results if r['significant'])
        print(f"\nSummary: {significant_count}/{len(results)} maxlags values show significant results")
        
        if significant_count > 0:
            print("Recommendation: Results are robust to maxlags choice")
        else:
            print("Recommendation: Results are sensitive to maxlags choice - interpret with caution")
        
        return results
        
    except Exception as e:
        print(f"Error in HAC sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run the complete analysis"""
    print("=== COMPREHENSIVE GRANGER CAUSALITY ANALYSIS: ILI vs SEARCH TRENDS ===")
    
    # Load and prepare data
    df_ili, df_search = load_and_prepare_data()
    
    # Perform diagnostic analysis
    filtered_columns = perform_diagnostic_analysis(df_search)
    
    # Merge and prepare data with lags
    df_ili, existing_flu_lags, existing_all_lags, response_var, max_lag = merge_and_prepare_data(
        df_ili, df_search, filtered_columns
    )
    
    # Perform Granger causality test
    model_restricted, model_unrestricted, F, p_value, df1, df2 = perform_granger_causality_test(
        df_ili, existing_flu_lags, existing_all_lags, response_var
    )
    
    if model_restricted is not None and model_unrestricted is not None:
        # Perform autocorrelation analysis
        dw_restricted, dw_unrestricted, has_autocorrelation = perform_autocorrelation_analysis(model_restricted, model_unrestricted)
        
        # If autocorrelation is detected, perform corrections
        if has_autocorrelation:
            print(f"\n⚠️  Autocorrelation detected! Performing corrections...")
            
            # Method 1: HAC standard errors (corrected approach)
            ols_u_hac, wald_res = perform_hac_analysis(
                df_ili, existing_flu_lags, existing_all_lags, response_var, F, p_value, maxlags=None
            )
            
            # Method 2: Cochrane-Orcutt transformation
            model_corrected, rho = perform_autocorrelation_correction(
                df_ili, existing_flu_lags, existing_all_lags, response_var
            )
            
            # Use HAC-adjusted results for individual term analysis
            if ols_u_hac is not None:
                print(f"\nUsing HAC-adjusted results for individual term analysis...")
                # Use the HAC-adjusted model for individual term analysis
                analyze_individual_terms(ols_u_hac, filtered_columns, max_lag, response_var, wald_res['fvalue'], wald_res['pvalue'])
            else:
                print(f"\nUsing original results for individual term analysis...")
                analyze_individual_terms(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value)
            
            # Perform HAC sensitivity analysis
            perform_hac_sensitivity_analysis(df_ili, existing_flu_lags, existing_all_lags, response_var)
        else:
            # No autocorrelation detected, use original results
            print(f"\n✓ No autocorrelation detected. Using standard results...")
            analyze_individual_terms(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
    else:
        print("Analysis could not be completed due to errors in model fitting.")


if __name__ == "__main__":
    main()