import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def week_to_date(row):
    """Convert YEAR and WEEK to date"""
    year_start = pd.Timestamp(f"{row['YEAR']}-01-01")
    week_start = year_start + pd.Timedelta(weeks=row['WEEK']-1)
    return week_start + pd.Timedelta(days=6)  # Adjust to Sunday

def create_lagged_variables(df, terms, max_lag, response_var):
    """Create lagged variables for terms and response variable"""
    df_lagged = df.copy()
    
    # Create lagged variables for each term
    for term in terms:
        for lag in range(1, max_lag + 1):
            df_lagged[f'{term}_lag{lag}'] = df_lagged[term].shift(lag)
    
    # Create lagged variables for response variable
    for lag in range(1, max_lag + 1):
        df_lagged[f'{response_var}_lag{lag}'] = df_lagged[response_var].shift(lag)
    
    return df_lagged.dropna()

def perform_individual_granger_test(df_lagged, term, max_lag, response_var):
    """Perform Granger causality test for a single term"""
    try:
        # Prepare variables
        X_vars = [f'{term}_lag{i}' for i in range(1, max_lag + 1)]
        y_vars = [f'{response_var}_lag{i}' for i in range(1, max_lag + 1)]
        
        # Model 1: Response variable regressed on its own lags
        X1 = df_lagged[y_vars]
        y1 = df_lagged[response_var]
        
        # Model 2: Response variable regressed on its own lags + term lags
        X2 = df_lagged[y_vars + X_vars]
        y2 = df_lagged[response_var]
        
        # Fit models
        model1 = LinearRegression().fit(X1, y1)
        model2 = LinearRegression().fit(X2, y2)
        
        # Calculate F-statistic
        rss1 = np.sum((y1 - model1.predict(X1))**2)
        rss2 = np.sum((y2 - model2.predict(X2))**2)
        
        if rss2 == 0:
            return 1.0  # No improvement, p-value = 1
        
        f_stat = ((rss1 - rss2) / max_lag) / (rss2 / (len(y2) - len(X_vars) - max_lag))
        p_value = 1 - stats.f.cdf(f_stat, max_lag, len(y2) - len(X_vars) - max_lag)
        
        return p_value
        
    except Exception as e:
        print(f"Error testing {term}: {e}")
        return 1.0

def test_term_in_multiple_groups(df, term, all_terms, max_lag, response_var, num_tests=10):
    """Test a term in multiple different group combinations"""
    p_values = []
    
    # Test the term individually first
    df_lagged = create_lagged_variables(df, [term], max_lag, response_var)
    individual_p = perform_individual_granger_test(df_lagged, term, max_lag, response_var)
    p_values.append(individual_p)
    
    # Test the term in multiple random groups
    other_terms = [t for t in all_terms if t != term]
    
    for _ in range(num_tests - 1):
        # Randomly select other terms to group with
        group_size = min(17, len(other_terms))  # Max 18 terms per group (including our term)
        if group_size > 0:
            random_terms = np.random.choice(other_terms, size=group_size, replace=False)
            group_terms = [term] + list(random_terms)
            
            df_lagged = create_lagged_variables(df, group_terms, max_lag, response_var)
            group_p = perform_individual_granger_test(df_lagged, term, max_lag, response_var)
            p_values.append(group_p)
    
    return p_values

def main():
    print("Loading data...")
    
    # Load search terms data
    df_search = pd.read_csv('ShiHaoYang/Data/trends_us_data_grouped.csv')
    df_search['date'] = pd.to_datetime(df_search['date'])
    
    # Load flu data
    df_flu = pd.read_csv('ShiHaoYang/Data/ICL_NREVSS_Public_Health_Labs.csv', skiprows=1)
    df_flu['date'] = df_flu.apply(week_to_date, axis=1)
    
    # Calculate percent positive
    flu_columns = ['A (2009 H1N1)', 'A (H3)', 'A (Subtyping not Performed)', 'B', 'BVic', 'BYam']
    df_flu['PERCENT POSITIVE'] = df_flu[flu_columns].sum(axis=1) / df_flu['TOTAL SPECIMENS'] * 100
    
    # Merge datasets
    df = pd.merge(df_search, df_flu, on='date', how='inner')
    print(f"Data loaded: {len(df)} observations")
    
    # Parameters
    max_lag = 5
    response_var = 'PERCENT POSITIVE'
    
    # Get all search terms (exclude flu data columns)
    flu_data_columns = ['REGION TYPE', 'REGION', 'YEAR', 'WEEK', 'TOTAL SPECIMENS', 
                       'A (2009 H1N1)', 'A (H3)', 'A (Subtyping not Performed)', 
                       'B', 'BVic', 'BYam', 'H3N2v', 'A (H5)', response_var]
    search_terms = [col for col in df.columns if col not in flu_data_columns + ['date']]
    print(f"Total search terms: {len(search_terms)}")
    
    print("\n=== TESTING EACH TERM IN MULTIPLE GROUP COMBINATIONS ===")
    print("This will test each term individually and in multiple random groups...")
    
    # Test each term
    term_results = {}
    
    for i, term in enumerate(search_terms):
        if i % 5 == 0:  # Show progress every 5 terms instead of 10
            print(f"Testing term {i+1}/{len(search_terms)}: {term}")
        
        p_values = test_term_in_multiple_groups(df, term, search_terms, max_lag, response_var, num_tests=20)
        term_results[term] = {
            'individual_p': p_values[0],
            'group_p_values': p_values[1:],
            'min_group_p': min(p_values[1:]) if len(p_values) > 1 else 1.0,
            'max_group_p': max(p_values[1:]) if len(p_values) > 1 else 1.0,
            'mean_group_p': np.mean(p_values[1:]) if len(p_values) > 1 else 1.0,
            'std_group_p': np.std(p_values[1:]) if len(p_values) > 1 else 0.0
        }
    
    # Analyze results
    print("\n=== RESULTS ANALYSIS ===")
    
    # Find terms that are consistently significant
    consistently_sig = []
    sometimes_sig = []
    never_sig = []
    
    for term, results in term_results.items():
        individual_sig = results['individual_p'] < 0.05
        group_sig_count = sum(1 for p in results['group_p_values'] if p < 0.05)
        group_sig_rate = group_sig_count / len(results['group_p_values'])
        
        if individual_sig and group_sig_rate > 0.5:
            consistently_sig.append(term)
        elif individual_sig or group_sig_rate > 0.2:
            sometimes_sig.append(term)
        else:
            never_sig.append(term)
    
    print(f"Consistently significant terms: {len(consistently_sig)}")
    print(f"Sometimes significant terms: {len(sometimes_sig)}")
    print(f"Never significant terms: {len(never_sig)}")
    
    # Show all consistently significant terms
    if consistently_sig:
        print("\nAll consistently significant terms:")
        sig_terms = [(term, term_results[term]['individual_p']) for term in consistently_sig]
        sig_terms.sort(key=lambda x: x[1])
        for i, (term, p_val) in enumerate(sig_terms, 1):
            print(f"  {i:2d}. {term}: p={p_val:.6f}")
    
    # Show all sometimes significant terms
    if sometimes_sig:
        print("\nAll sometimes significant terms:")
        sig_terms = [(term, term_results[term]['individual_p']) for term in sometimes_sig]
        sig_terms.sort(key=lambda x: x[1])
        for i, (term, p_val) in enumerate(sig_terms, 1):
            print(f"  {i:2d}. {term}: p={p_val:.6f}")
    
    # Create visualization
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Individual vs Group p-values
    plt.subplot(2, 2, 1)
    individual_ps = [term_results[term]['individual_p'] for term in search_terms]
    mean_group_ps = [term_results[term]['mean_group_p'] for term in search_terms]
    
    plt.scatter(individual_ps, mean_group_ps, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    plt.xlabel('Individual p-value')
    plt.ylabel('Mean Group p-value')
    plt.title('Individual vs Group p-values')
    plt.xscale('log')
    plt.yscale('log')
    
    # Plot 2: P-value stability (std of group p-values)
    plt.subplot(2, 2, 2)
    p_stability = [term_results[term]['std_group_p'] for term in search_terms]
    plt.hist(p_stability, bins=20, alpha=0.7)
    plt.xlabel('Standard deviation of group p-values')
    plt.ylabel('Number of terms')
    plt.title('P-value Stability Across Groups')
    
    # Plot 3: All significant terms
    plt.subplot(2, 2, 3)
    significant_terms = [(term, results) for term, results in term_results.items() if results['individual_p'] < 0.05]
    significant_terms.sort(key=lambda x: x[1]['individual_p'])
    
    if len(significant_terms) > 0:
        terms = [item[0][:25] + '...' if len(item[0]) > 25 else item[0] for item in significant_terms]
        p_vals = [item[1]['individual_p'] for item in significant_terms]
        
        plt.barh(range(len(terms)), p_vals)
        plt.yticks(range(len(terms)), terms)
        plt.xlabel('p-value')
        plt.title(f'All Significant Terms (p<0.05): {len(significant_terms)} terms')
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, 'No significant terms found', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('All Significant Terms')
    
    # Plot 4: Consistency analysis
    plt.subplot(2, 2, 4)
    categories = ['Consistently\nSignificant', 'Sometimes\nSignificant', 'Never\nSignificant']
    counts = [len(consistently_sig), len(sometimes_sig), len(never_sig)]
    colors = ['green', 'orange', 'red']
    
    plt.bar(categories, counts, color=colors, alpha=0.7)
    plt.ylabel('Number of terms')
    plt.title('Term Significance Consistency')
    
    plt.tight_layout()
    plt.savefig(f'ShiHaoYang/Results/New/granger_causality_robust_analysis_ili_max_lag{max_lag}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    results_df = pd.DataFrame([
        {
            'term': term,
            'individual_p': results['individual_p'],
            'mean_group_p': results['mean_group_p'],
            'min_group_p': results['min_group_p'],
            'max_group_p': results['max_group_p'],
            'std_group_p': results['std_group_p'],
            'consistency': 'Consistent' if term in consistently_sig else ('Sometimes' if term in sometimes_sig else 'Never')
        }
        for term, results in term_results.items()
    ])
    
    results_df.to_csv(f'ShiHaoYang/Results/New/granger_causality_robust_results_ili_max_lag{max_lag}.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"- ShiHaoYang/Results/New/granger_causality_robust_analysis_ili_max_lag{max_lag}.png")
    print(f"- ShiHaoYang/Results/New/granger_causality_robust_results_ili_max_lag{max_lag}.csv")
    
    print(f"\nSummary:")
    print(f"- Total terms tested: {len(search_terms)}")
    print(f"- Consistently significant: {len(consistently_sig)} ({len(consistently_sig)/len(search_terms)*100:.1f}%)")
    print(f"- Sometimes significant: {len(sometimes_sig)} ({len(sometimes_sig)/len(search_terms)*100:.1f}%)")
    print(f"- Never significant: {len(never_sig)} ({len(never_sig)/len(search_terms)*100:.1f}%)")

if __name__ == "__main__":
    main() 