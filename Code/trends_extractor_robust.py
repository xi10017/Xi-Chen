import pandas as pd
import time
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from pytrends.request import TrendReq
    PYTTRENDS_AVAILABLE = True
except ImportError:
    logger.warning("pytrends not available. Install with: pip install pytrends")
    PYTTRENDS_AVAILABLE = False

class RobustTrendsExtractor:
    def __init__(self):
        self.pytrends = None
        self.setup_pytrends()
        self.successful_batches = 0
        self.failed_batches = 0
        
    def setup_pytrends(self):
        """Setup pytrends with conservative settings"""
        if not PYTTRENDS_AVAILABLE:
            logger.error("pytrends not available")
            return
            
        try:
            # Initialize with very conservative settings
            self.pytrends = TrendReq(
                hl='en-US',
                tz=360,
                retries=3,
                backoff_factor=2,
                timeout=(10, 30)  # (connect timeout, read timeout)
            )
            logger.info("pytrends initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pytrends: {e}")
    
    def extract_trends_data(self, keywords: List[str], 
                           start_date: str = '2020-01-01', 
                           end_date: str = '2025-01-01',
                           geo: str = 'US',
                           batch_size: int = 1,  # Single keyword per batch
                           delay_range: tuple = (15, 25)) -> pd.DataFrame:
        """
        Extract Google Trends data with very conservative settings
        
        Args:
            keywords: List of keywords to extract
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            geo: Geographic location
            batch_size: Number of keywords per batch (recommended: 1)
            delay_range: Range of seconds to delay between requests (min, max)
        
        Returns:
            DataFrame with trends data
        """
        if not self.pytrends:
            logger.error("pytrends not available")
            return pd.DataFrame()
        
        logger.info(f"Starting extraction for {len(keywords)} keywords")
        logger.info(f"Using batch size: {batch_size}, delay range: {delay_range}")
        
        all_data = pd.DataFrame()
        batches = [keywords[i:i + batch_size] for i in range(0, len(keywords), batch_size)]
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {batch}")
            
            try:
                # Build payload with conservative settings
                self.pytrends.build_payload(
                    batch, 
                    cat=0, 
                    timeframe=f'{start_date} {end_date}', 
                    geo=geo
                )
                
                # Get interest over time
                df = self.pytrends.interest_over_time()
                
                if df is not None and not df.empty:
                    # Remove the 'isPartial' column if it exists
                    if 'isPartial' in df.columns:
                        df = df.drop(columns=['isPartial'])
                    
                    # Join with existing data
                    if all_data.empty:
                        all_data = df
                    else:
                        all_data = all_data.join(df, how='outer')
                    
                    self.successful_batches += 1
                    logger.info(f"✓ Batch {i+1} successful")
                else:
                    self.failed_batches += 1
                    logger.warning(f"✗ Batch {i+1} returned empty data")
                
                # Very long random delay
                delay = random.uniform(*delay_range)
                logger.info(f"Waiting {delay:.1f} seconds before next batch...")
                time.sleep(delay)
                
            except Exception as e:
                self.failed_batches += 1
                logger.error(f"Error processing batch {i+1}: {e}")
                
                # Even longer delay on error
                error_delay = delay_range[1] + 10
                logger.info(f"Waiting {error_delay} seconds after error...")
                time.sleep(error_delay)
                continue
        
        # Log final statistics
        logger.info(f"Extraction completed. Successful: {self.successful_batches}, Failed: {self.failed_batches}")
        
        return all_data
    
    def get_related_queries(self, keyword: str, geo: str = 'US') -> Dict:
        """Get related queries for a keyword"""
        if not self.pytrends:
            return {}
        
        try:
            # Build payload for single keyword
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo=geo)
            
            # Get related queries
            related_queries = self.pytrends.related_queries()
            
            if keyword in related_queries:
                return related_queries[keyword]
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting related queries for '{keyword}': {e}")
            return {}
    
    def save_progress(self, data: pd.DataFrame, filename: str):
        """Save data with progress tracking"""
        try:
            data.to_csv(filename)
            logger.info(f"Data saved to {filename}")
            logger.info(f"Shape: {data.shape}")
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'successful_batches': self.successful_batches,
                'failed_batches': self.failed_batches,
                'total_keywords': len(data.columns) if not data.empty else 0,
                'date_range': f"{data.index.min()} to {data.index.max()}" if not data.empty else "N/A"
            }
            
            metadata_file = filename.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")

def read_keywords_from_file(file_path: str) -> List[str]:
    """Read keywords from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip() and not line.startswith('/m/')]
        return keywords
    except Exception as e:
        logger.error(f"Error reading keywords from file: {e}")
        return []

def create_robust_dataset(keywords: List[str], 
                         output_file: str = 'ShiHaoYang/Data/google_trends_robust.csv',
                         start_date: str = '2020-01-01',
                         end_date: str = '2025-01-01',
                         geo: str = 'US',
                         save_progress: bool = True) -> pd.DataFrame:
    """
    Create a robust dataset with conservative settings
    
    Args:
        keywords: List of keywords to extract
        output_file: Path to save the dataset
        start_date: Start date for trends data
        end_date: End date for trends data
        geo: Geographic location
        save_progress: Whether to save progress during extraction
    
    Returns:
        DataFrame with the complete dataset
    """
    logger.info("Creating robust Google Trends dataset")
    
    # Initialize extractor
    extractor = RobustTrendsExtractor()
    
    if not extractor.pytrends:
        logger.error("Cannot proceed without pytrends")
        return pd.DataFrame()
    
    # Extract data with very conservative settings
    trends_data = extractor.extract_trends_data(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        geo=geo,
        batch_size=1,  # Single keyword per batch
        delay_range=(15, 25)  # 15-25 second delays
    )
    
    if trends_data is not None and not trends_data.empty:
        # Save the dataset
        extractor.save_progress(trends_data, output_file)
        
        # Get related queries for a few keywords (to avoid rate limiting)
        sample_keywords = keywords[:3]  # Only first 3 keywords
        related_data = {}
        
        for keyword in sample_keywords:
            logger.info(f"Getting related queries for: {keyword}")
            related_data[keyword] = extractor.get_related_queries(keyword, geo)
            time.sleep(10)  # 10 second delay between related queries
        
        # Save related queries data
        related_file = output_file.replace('.csv', '_related_queries.json')
        with open(related_file, 'w') as f:
            json.dump(related_data, f, indent=2)
        logger.info(f"Related queries saved to {related_file}")
        
        return trends_data
    else:
        logger.error("No data retrieved")
        return pd.DataFrame()

def main():
    """Main function to run the robust trends extraction"""
    
    # Check if pytrends is available
    if not PYTTRENDS_AVAILABLE:
        logger.error("Please install pytrends: pip install pytrends")
        return
    
    # Read keywords from file
    keywords = read_keywords_from_file('ShiHaoYang/Data/US_terms.txt')
    
    if not keywords:
        logger.error("No keywords found in file")
        return
    
    logger.info(f"Loaded {len(keywords)} keywords from file")
    
    # Create robust dataset
    dataset = create_robust_dataset(
        keywords=keywords,
        output_file='ShiHaoYang/Data/google_trends_robust.csv',
        start_date='2020-01-01',
        end_date='2025-01-01',
        geo='US'
    )
    
    if not dataset.empty:
        logger.info("Robust dataset creation completed successfully!")
        logger.info(f"Final dataset contains {len(dataset.columns)} keywords")
        logger.info(f"Time range: {dataset.index.min()} to {dataset.index.max()}")
    else:
        logger.error("Robust dataset creation failed")

if __name__ == "__main__":
    main() 