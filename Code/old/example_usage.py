#!/usr/bin/env python3
"""
Example usage of the Google Trends Extractor

This script demonstrates how to use the trends_extractor.py to extract
Google Trends data and create a comprehensive dataset.
"""

from trends_extractor import GoogleTrendsExtractor, create_dataset, read_keywords_from_file
import pandas as pd

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Read keywords from file
    keywords = read_keywords_from_file('ShiHaoYang/Data/US_terms.txt')
    print(f"Loaded {len(keywords)} keywords")
    
    # Create dataset
    dataset = create_dataset(
        keywords=keywords[:10],  # Use first 10 keywords for testing
        output_file='ShiHaoYang/Data/example_dataset.csv',
        start_date='2023-01-01',  # Shorter timeframe for testing
        end_date='2024-12-31',
        geo='US'
    )
    
    if not dataset.empty:
        print(f"Dataset created successfully!")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {list(dataset.columns)}")
        print(f"Date range: {dataset.index.min()} to {dataset.index.max()}")
    else:
        print("Dataset creation failed")

def example_advanced_usage():
    """Advanced usage with custom settings"""
    print("\n=== Advanced Usage Example ===")
    
    # Initialize extractor with custom settings
    extractor = GoogleTrendsExtractor()
    
    # Custom keywords
    custom_keywords = [
        "flu symptoms",
        "covid symptoms", 
        "fever",
        "cough",
        "headache"
    ]
    
    # Extract data with custom parameters
    trends_data = extractor.extract_trends_data(
        keywords=custom_keywords,
        start_date='2022-01-01',
        end_date='2024-12-31',
        geo='US',
        batch_size=2,  # Small batches
        delay_range=(10, 15)  # Longer delays
    )
    
    if not trends_data.empty:
        # Save to custom file
        output_file = 'ShiHaoYang/Data/custom_trends.csv'
        trends_data.to_csv(output_file)
        print(f"Custom dataset saved to {output_file}")
        print(f"Data shape: {trends_data.shape}")
        
        # Show some statistics
        print("\nData Statistics:")
        print(trends_data.describe())
        
        # Show recent trends
        print("\nRecent Trends (last 5 dates):")
        print(trends_data.tail())
    else:
        print("Custom extraction failed")

def example_related_queries():
    """Example of getting related queries"""
    print("\n=== Related Queries Example ===")
    
    extractor = GoogleTrendsExtractor()
    
    # Get related queries for a keyword
    keyword = "flu symptoms"
    related_data = extractor.get_related_queries(keyword, geo='US')
    
    if related_data:
        print(f"Related queries for '{keyword}':")
        
        if 'top_queries' in related_data:
            print("\nTop Related Queries:")
            for i, query in enumerate(related_data['top_queries'][:5], 1):
                print(f"{i}. {query['query']} (value: {query['value']})")
        
        if 'rising_queries' in related_data:
            print("\nRising Related Queries:")
            for i, query in enumerate(related_data['rising_queries'][:5], 1):
                print(f"{i}. {query['query']} (value: {query['value']})")
    else:
        print("Failed to get related queries")

def example_data_analysis():
    """Example of basic data analysis"""
    print("\n=== Data Analysis Example ===")
    
    # Load existing dataset
    try:
        dataset = pd.read_csv('ShiHaoYang/Data/google_trends_dataset.csv', index_col=0, parse_dates=True)
        
        print(f"Dataset loaded: {dataset.shape}")
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(dataset.describe())
        
        # Find most popular keywords
        print("\nMost Popular Keywords (average interest):")
        avg_interest = dataset.mean().sort_values(ascending=False)
        for keyword, interest in avg_interest.head(10).items():
            print(f"{keyword}: {interest:.2f}")
        
        # Find trending keywords (highest recent interest)
        print("\nTrending Keywords (recent interest):")
        recent_interest = dataset.tail(30).mean().sort_values(ascending=False)
        for keyword, interest in recent_interest.head(10).items():
            print(f"{keyword}: {interest:.2f}")
        
        # Correlation analysis
        print("\nCorrelation Matrix (top 5 keywords):")
        top_keywords = avg_interest.head(5).index
        correlation = dataset[top_keywords].corr()
        print(correlation)
        
    except FileNotFoundError:
        print("Dataset file not found. Run the main extraction first.")

if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_advanced_usage()
    example_related_queries()
    example_data_analysis() 