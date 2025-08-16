import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.multitest import multipletests


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
    max_lag = 2
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
    bonferroni_threshold = 0.05 / num_tests
    
    # Perform FDR correction (Benjamini-Hochberg)
    search_term_pvalues = coef_results.loc[search_term_lags, 'p-value'].values
    fdr_rejected, fdr_pvalues, _, _ = multipletests(search_term_pvalues, method='fdr_bh', alpha=0.05)
    
    # Create mapping of terms to FDR significance
    fdr_significant_terms = set()
    for i, term in enumerate(search_term_lags):
        if fdr_rejected[i]:
            fdr_significant_terms.add(term)
    
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
        
        # Save significant terms to file
        comprehensive_txt_filename = f"ShiHaoYang/Results/comprehensive_granger_individual_significant_terms_ili_lag{max_lag}.txt"
        with open(comprehensive_txt_filename, "w") as f:
            f.write(f"Significant terms from comprehensive Granger causality model\n")
            f.write(f"Response variable: {response_var}\n")
            f.write(f"Max lag: {max_lag}\n")
            f.write(f"Number of tests: {num_tests}\n")
            f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
            f.write(f"Overall Granger causality F-statistic: {F:.4f}\n")
            f.write(f"Overall Granger causality p-value: {p_value:.6f}\n")
            f.write(f"Total significant terms: {len(significant_in_model)}\n")
            f.write(f"Model R-squared: {model_unrestricted.rsquared:.4f}\n\n")
            
            f.write("Term\tCoefficient\tStd Error\tt-value\tp-value\tCI Lower\tCI Upper\tSignificance\n")
            for term, row in significant_in_model.iterrows():
                if term in fdr_significant_terms:
                    significance = "FDR"
                elif row['p-value'] < bonferroni_threshold:
                    significance = "Bonferroni"
                else:
                    significance = "Uncorrected"
                f.write(f"{term}\t{row['Coefficient']:.6f}\t{row['Std Error']:.6f}\t{row['t-value']:.4f}\t{row['p-value']:.6f}\t{row['CI Lower']:.6f}\t{row['CI Upper']:.6f}\t{significance}\n")
        
        print(f"Detailed results saved to {comprehensive_txt_filename}")
        
        # Create visualization
        create_comprehensive_visualization(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value, bonferroni_threshold, fdr_significant_terms)
        
    else:
        print("No terms were significant after multiple testing correction.")
        
        # Still save uncorrected results for reference
        if len(significant_uncorrected) > 0:
            print(f"Note: {len(significant_uncorrected)} terms were significant at uncorrected p < 0.05")
            comprehensive_txt_filename = f"ShiHaoYang/Results/comprehensive_granger_individual_significant_terms_ili_lag{max_lag}.txt"
            with open(comprehensive_txt_filename, "w") as f:
                f.write(f"Uncorrected significant terms (p < 0.05) from comprehensive Granger causality model\n")
                f.write(f"Response variable: {response_var}\n")
                f.write(f"Max lag: {max_lag}\n")
                f.write(f"Number of tests: {num_tests}\n")
                f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
                f.write(f"Overall Granger causality F-statistic: {F:.4f}\n")
                f.write(f"Overall Granger causality p-value: {p_value:.6f}\n")
                f.write(f"Total significant terms (uncorrected): {len(significant_uncorrected)}\n")
                f.write(f"Model R-squared: {model_unrestricted.rsquared:.4f}\n\n")
                
                f.write("Term\tCoefficient\tStd Error\tt-value\tp-value\tCI Lower\tCI Upper\n")
                for term, row in significant_uncorrected.iterrows():
                    f.write(f"{term}\t{row['Coefficient']:.6f}\t{row['Std Error']:.6f}\t{row['t-value']:.4f}\t{row['p-value']:.6f}\t{row['CI Lower']:.6f}\t{row['CI Upper']:.6f}\n")
            
            print(f"Uncorrected results saved to {comprehensive_txt_filename}")


def create_comprehensive_visualization(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value, bonferroni_threshold, fdr_significant_terms):
    """Create visualization for comprehensive model results"""
    # Get minimum p-values for each search term
    term_min_pvalues = []
    for term in filtered_columns:
        term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
        term_pvals = []
        for lag_col in term_lags:
            if lag_col in model_unrestricted.params.index:
                pval = model_unrestricted.pvalues[lag_col]
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
        
        # Analyze individual terms
        analyze_individual_terms(model_unrestricted, filtered_columns, max_lag, response_var, F, p_value)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
    else:
        print("Analysis could not be completed due to errors in model fitting.")


if __name__ == "__main__":
    main()

