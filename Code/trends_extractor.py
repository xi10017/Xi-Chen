import pandas as pd
import requests
import time
import json
from datetime import datetime, timedelta
import random
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleTrendsExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.setup_session()
        self.data_cache = {}
        
    def setup_session(self):
        """Setup session with realistic headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
        })
    
    def extract_trends_data(self, keywords: List[str], 
                           start_date: str = '2020-01-01', 
                           end_date: str = '2025-01-01',
                           geo: str = 'US',
                           batch_size: int = 2,
                           delay_range: tuple = (8, 12)) -> pd.DataFrame:
        """
        Extract Google Trends data for a list of keywords
        
        Args:
            keywords: List of keywords to extract
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            geo: Geographic location (e.g., 'US', 'US-CA')
            batch_size: Number of keywords per batch
            delay_range: Range of seconds to delay between requests (min, max)
        
        Returns:
            DataFrame with trends data
        """
        logger.info(f"Starting extraction for {len(keywords)} keywords")
        
        all_data = pd.DataFrame()
        batches = [keywords[i:i + batch_size] for i in range(0, len(keywords), batch_size)]
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {batch}")
            
            try:
                # Try multiple endpoints
                batch_data = self._get_batch_data(batch, start_date, end_date, geo)
                
                if batch_data is not None and not batch_data.empty:
                    if all_data.empty:
                        all_data = batch_data
                    else:
                        all_data = all_data.join(batch_data, how='outer')
                    logger.info(f"✓ Batch {i+1} successful")
                else:
                    logger.warning(f"✗ Batch {i+1} failed")
                
                # Random delay
                delay = random.uniform(*delay_range)
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                time.sleep(delay_range[1] + 5)  # Longer delay on error
                continue
        
        return all_data
    
    def _get_batch_data(self, keywords: List[str], start_date: str, end_date: str, geo: str) -> Optional[pd.DataFrame]:
        """Get data for a batch of keywords using multiple endpoints"""
        
        # Try different endpoints
        endpoints = [
            self._try_interest_over_time,
            self._try_multiline_endpoint,
            self._try_compared_geo
        ]
        
        for endpoint_func in endpoints:
            try:
                data = endpoint_func(keywords, start_date, end_date, geo)
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                logger.debug(f"Endpoint {endpoint_func.__name__} failed: {e}")
                continue
        
        return None
    
    def _try_interest_over_time(self, keywords: List[str], start_date: str, end_date: str, geo: str) -> Optional[pd.DataFrame]:
        """Try the interest over time endpoint"""
        url = "https://trends.google.com/trends/api/widgetdata/interestoverime"
        
        req_data = {
            "time": f"{start_date} {end_date}",
            "keyword": keywords,
            "cat": 0,
            "geo": geo,
            "hl": "en-US",
            "tz": "-120"
        }
        
        params = {
            "hl": "en-US",
            "tz": "-120",
            "req": json.dumps(req_data),
            "token": self._generate_token(keywords)
        }
        
        response = self.session.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            return self._parse_response(response.text, keywords)
        else:
            logger.debug(f"HTTP {response.status_code} for interest over time")
            return None
    
    def _try_multiline_endpoint(self, keywords: List[str], start_date: str, end_date: str, geo: str) -> Optional[pd.DataFrame]:
        """Try the multiline endpoint"""
        url = "https://trends.google.com/trends/api/widgetdata/multiline"
        
        req_data = {
            "time": f"{start_date} {end_date}",
            "keyword": keywords,
            "cat": 0,
            "geo": geo,
            "hl": "en-US",
            "tz": "-120"
        }
        
        params = {
            "hl": "en-US",
            "tz": "-120",
            "req": json.dumps(req_data),
            "token": self._generate_token(keywords)
        }
        
        response = self.session.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            return self._parse_response(response.text, keywords)
        else:
            logger.debug(f"HTTP {response.status_code} for multiline")
            return None
    
    def _try_compared_geo(self, keywords: List[str], start_date: str, end_date: str, geo: str) -> Optional[pd.DataFrame]:
        """Try the compared geo endpoint"""
        url = "https://trends.google.com/trends/api/widgetdata/comparedgeo"
        
        req_data = {
            "time": f"{start_date} {end_date}",
            "keyword": keywords,
            "cat": 0,
            "geo": geo,
            "hl": "en-US",
            "tz": "-120"
        }
        
        params = {
            "hl": "en-US",
            "tz": "-120",
            "req": json.dumps(req_data),
            "token": self._generate_token(keywords)
        }
        
        response = self.session.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            return self._parse_response(response.text, keywords)
        else:
            logger.debug(f"HTTP {response.status_code} for compared geo")
            return None
    
    def _generate_token(self, keywords: List[str]) -> str:
        """Generate a token for the request"""
        timestamp = int(time.time())
        keyword_hash = abs(hash(str(keywords))) % 1000000
        return f"APP6_UEAAAAAY{timestamp}_{keyword_hash}"
    
    def _parse_response(self, response_text: str, keywords: List[str]) -> Optional[pd.DataFrame]:
        """Parse the API response"""
        try:
            # Remove the leading ")]}'" that Google adds
            if response_text.startswith(")]}'"):
                response_text = response_text[5:]
            
            data = json.loads(response_text)
            
            # Handle different response formats
            if 'timelineData' in data:
                return self._parse_timeline_data(data['timelineData'], keywords)
            elif 'default' in data and 'timelineData' in data['default']:
                return self._parse_timeline_data(data['default']['timelineData'], keywords)
            else:
                logger.debug(f"Unexpected response format: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None
    
    def _parse_timeline_data(self, timeline_data: List[Dict], keywords: List[str]) -> pd.DataFrame:
        """Parse timeline data into a pandas DataFrame"""
        if not timeline_data:
            return pd.DataFrame()
        
        dates = []
        values_dict = {keyword: [] for keyword in keywords}
        
        for entry in timeline_data:
            # Convert timestamp to date
            timestamp = entry.get('time', 0)
            date = datetime.fromtimestamp(timestamp)
            dates.append(date)
            
            # Extract values for each keyword
            values = entry.get('value', [])
            for i, keyword in enumerate(keywords):
                if i < len(values):
                    values_dict[keyword].append(values[i])
                else:
                    values_dict[keyword].append(0)
        
        # Create DataFrame
        df = pd.DataFrame(values_dict, index=dates)
        df.index.name = 'date'
        
        return df
    
    def get_related_queries(self, keyword: str, geo: str = 'US') -> Dict:
        """Get related queries for a keyword"""
        try:
            url = "https://trends.google.com/trends/api/widgetdata/relatedsearches"
            
            req_data = {
                "restriction": {
                    "geo": {"country": geo},
                    "time": "today 12-m",
                    "originalTimeRangeForExploreUrl": "today 12-m"
                },
                "keywordType": "QUERY",
                "metric": ["TOP", "RISING"],
                "trendinessSettings": {
                    "compareTime": "today 12-m"
                },
                "requestOptions": {
                    "property": "",
                    "backend": "IZG",
                    "category": 0
                },
                "language": "en"
            }
            
            params = {
                "hl": "en-US",
                "tz": "-120",
                "req": json.dumps(req_data),
                "token": self._generate_token([keyword])
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return self._parse_related_queries(response.text)
            else:
                logger.warning(f"HTTP {response.status_code} for related queries")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting related queries: {e}")
            return {}
    
    def _parse_related_queries(self, response_text: str) -> Dict:
        """Parse related queries response"""
        try:
            if response_text.startswith(")]}'"):
                response_text = response_text[5:]
            
            data = json.loads(response_text)
            
            if 'default' in data and 'rankedList' in data['default']:
                ranked_list = data['default']['rankedList']
                
                top_queries = []
                rising_queries = []
                
                for ranked_item in ranked_list:
                    if 'rankedKeyword' in ranked_item:
                        for item in ranked_item['rankedKeyword']:
                            query_data = {
                                'query': item.get('query', ''),
                                'value': item.get('value', 0)
                            }
                            
                            if ranked_item.get('rank') == 'TOP':
                                top_queries.append(query_data)
                            elif ranked_item.get('rank') == 'RISING':
                                rising_queries.append(query_data)
                
                return {
                    'top_queries': top_queries,
                    'rising_queries': rising_queries
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error parsing related queries: {e}")
            return {}

def read_keywords_from_file(file_path: str) -> List[str]:
    """Read keywords from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip() and not line.startswith('/m/')]
        return keywords
    except Exception as e:
        logger.error(f"Error reading keywords from file: {e}")
        return []

def create_dataset(keywords: List[str], 
                  output_file: str = 'ShiHaoYang/Data/google_trends_dataset.csv',
                  start_date: str = '2020-01-01',
                  end_date: str = '2025-01-01',
                  geo: str = 'US') -> pd.DataFrame:
    """
    Create a comprehensive dataset from Google Trends data
    
    Args:
        keywords: List of keywords to extract
        output_file: Path to save the dataset
        start_date: Start date for trends data
        end_date: End date for trends data
        geo: Geographic location
    
    Returns:
        DataFrame with the complete dataset
    """
    logger.info("Creating Google Trends dataset")
    
    # Initialize extractor
    extractor = GoogleTrendsExtractor()
    
    # Extract trends data
    trends_data = extractor.extract_trends_data(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        geo=geo,
        batch_size=2,
        delay_range=(8, 12)
    )
    
    if trends_data is not None and not trends_data.empty:
        # Save the dataset
        trends_data.to_csv(output_file)
        logger.info(f"Dataset saved to {output_file}")
        logger.info(f"Dataset shape: {trends_data.shape}")
        logger.info(f"Date range: {trends_data.index.min()} to {trends_data.index.max()}")
        
        # Get related queries for a sample of keywords (to avoid rate limiting)
        sample_keywords = keywords[:5]  # First 5 keywords
        related_data = {}
        
        for keyword in sample_keywords:
            logger.info(f"Getting related queries for: {keyword}")
            related_data[keyword] = extractor.get_related_queries(keyword, geo)
            time.sleep(5)  # Delay between related queries requests
        
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
    """Main function to run the trends extraction"""
    
    # Read keywords from file
    keywords = read_keywords_from_file('ShiHaoYang/Data/US_terms.txt')
    
    if not keywords:
        logger.error("No keywords found in file")
        return
    
    logger.info(f"Loaded {len(keywords)} keywords from file")
    
    # Create dataset
    dataset = create_dataset(
        keywords=keywords,
        output_file='ShiHaoYang/Data/google_trends_dataset.csv',
        start_date='2020-01-01',
        end_date='2025-01-01',
        geo='US'
    )
    
    if not dataset.empty:
        logger.info("Dataset creation completed successfully!")
        logger.info(f"Final dataset contains {len(dataset.columns)} keywords")
        logger.info(f"Time range: {dataset.index.min()} to {dataset.index.max()}")
    else:
        logger.error("Dataset creation failed")

if __name__ == "__main__":
    main() 