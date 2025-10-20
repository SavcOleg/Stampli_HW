"""
Query parser to extract intents and metadata filters from natural language queries.
Handles geographic, temporal, and rating-based filtering.
"""

import re
from typing import Dict, List, Optional
import yaml
from pathlib import Path


class QueryParser:
    """Parse queries to extract filtering intents."""
    
    # Season keywords
    SEASONS = {
        'spring': ['spring', 'march', 'april', 'may'],
        'summer': ['summer', 'june', 'july', 'august'],
        'fall': ['fall', 'autumn', 'september', 'october', 'november'],
        'winter': ['winter', 'december', 'january', 'february']
    }
    
    # Month mapping
    MONTHS = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    # Park name variants
    PARKS = {
        'hong kong': ['hong kong', 'hk', 'hongkong'],
        'california': ['california', 'anaheim', 'disneyland california'],
        'paris': ['paris', 'disneyland paris']
    }
    
    def __init__(self, lookup_dir: str = "data/lookup"):
        """Initialize with lookup directory for country mappings."""
        self.lookup_dir = Path(lookup_dir)
        self.country_map = self._load_countries()
    
    def _load_countries(self) -> Dict[str, str]:
        """Load country mappings from YAML."""
        country_file = self.lookup_dir / "countries.yaml"
        if not country_file.exists():
            return {}
        
        with open(country_file, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('mappings', {})
    
    def extract_park(self, query: str) -> Optional[str]:
        """
        Extract park name from query.
        
        Args:
            query: Natural language query
        
        Returns:
            Normalized park name or None
        """
        query_lower = query.lower()
        
        for park, variants in self.PARKS.items():
            for variant in variants:
                if variant in query_lower:
                    return park.title()
        
        return None
    
    def extract_country(self, query: str) -> Optional[str]:
        """
        Extract country from query.
        
        Args:
            query: Natural language query
        
        Returns:
            Normalized country name or None
        """
        query_lower = query.lower()
        
        # Check against known countries
        for country_variant, normalized in self.country_map.items():
            if country_variant.lower() in query_lower:
                return normalized
        
        return None
    
    def extract_season(self, query: str) -> Optional[str]:
        """
        Extract season from query.
        
        Args:
            query: Natural language query
        
        Returns:
            Season name or None
        """
        query_lower = query.lower()
        
        for season, keywords in self.SEASONS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return season.title()
        
        return None
    
    def extract_month(self, query: str) -> Optional[int]:
        """
        Extract month number from query.
        
        Args:
            query: Natural language query
        
        Returns:
            Month number (1-12) or None
        """
        query_lower = query.lower()
        
        for month_name, month_num in self.MONTHS.items():
            if month_name in query_lower:
                return month_num
        
        return None
    
    def extract_rating_intent(self, query: str) -> Optional[str]:
        """
        Extract rating tier intent (positive/negative/neutral).
        
        Args:
            query: Natural language query
        
        Returns:
            Rating tier or None
        """
        query_lower = query.lower()
        
        positive_keywords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                           'fantastic', 'love', 'best', 'highly recommend']
        negative_keywords = ['bad', 'terrible', 'horrible', 'awful', 'worst', 
                           'disappointing', 'not recommend', 'waste']
        
        has_positive = any(kw in query_lower for kw in positive_keywords)
        has_negative = any(kw in query_lower for kw in negative_keywords)
        
        if has_positive and not has_negative:
            return 'positive'
        elif has_negative and not has_positive:
            return 'negative'
        
        return None
    
    def parse(self, query: str) -> Dict:
        """
        Parse query and extract all intents.
        
        Args:
            query: Natural language query
        
        Returns:
            Dictionary with extracted filters and metadata
        """
        filters = {}
        
        # Extract geographic filters
        park = self.extract_park(query)
        if park:
            filters['park'] = park
        
        country = self.extract_country(query)
        if country:
            filters['country'] = country
        
        # Extract temporal filters
        season = self.extract_season(query)
        if season:
            filters['season'] = season
        
        month = self.extract_month(query)
        if month:
            filters['month'] = month
        
        # Extract rating intent
        rating_tier = self.extract_rating_intent(query)
        if rating_tier:
            filters['rating_tier'] = rating_tier
        
        return {
            'original_query': query,
            'filters': filters,
            'has_filters': len(filters) > 0
        }


if __name__ == "__main__":
    # Test the query parser
    parser = QueryParser()
    
    test_queries = [
        "What do visitors from Australia say about Disneyland in Hong Kong?",
        "Is spring a good time to visit Disneyland?",
        "Is Disneyland California crowded in June?",
        "Is the staff in Paris friendly?"
    ]
    
    print("Testing Query Parser:")
    print("=" * 80)
    
    for query in test_queries:
        result = parser.parse(query)
        print(f"\nQuery: {query}")
        print(f"Filters: {result['filters']}")

