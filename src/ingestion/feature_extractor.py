"""
Feature extraction from Disney reviews.
Extracts metadata: park, country, month, season, rating tier, topics.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
import re


class FeatureExtractor:
    """Extract metadata features from review data."""
    
    SEASON_MAP = {
        1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
        12: "Winter"
    }
    
    MONTH_NAMES = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    def __init__(self, lookup_dir: str = "data/lookup"):
        """Initialize with lookup directories for normalization."""
        self.lookup_dir = Path(lookup_dir)
        self.country_map = self._load_yaml("countries.yaml")
        self.park_map = self._load_yaml("parks.yaml")
        self.topic_keywords = self._load_yaml("topics.yaml")
        
    def _load_yaml(self, filename: str) -> Dict:
        """Load YAML lookup file."""
        filepath = self.lookup_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('mappings', {}) if 'mappings' in data else data
    
    def extract_park(self, branch: str) -> str:
        """Normalize park name from Branch column."""
        if not branch or branch.strip() == "":
            return "Unknown"
        return self.park_map.get(branch, "Unknown")
    
    def extract_country(self, reviewer_location: str) -> str:
        """Normalize country from Reviewer_Location."""
        if not reviewer_location or reviewer_location.strip() == "":
            return "Unknown"
        
        # Try exact match first
        if reviewer_location in self.country_map:
            return self.country_map[reviewer_location]
        
        # Try case-insensitive match
        for key, value in self.country_map.items():
            if key.lower() == reviewer_location.lower():
                return value
                
        return "Unknown"
    
    def extract_temporal_features(self, year_month: str) -> Dict[str, any]:
        """
        Extract month, year, and season from Year_Month column.
        Format: YYYY-M (e.g., "2019-4")
        """
        if not year_month or '-' not in str(year_month):
            return {
                "year": None,
                "month": None,
                "month_name": "Unknown",
                "season": "Unknown"
            }
        
        try:
            parts = str(year_month).split('-')
            year = int(parts[0])
            month = int(parts[1])
            
            return {
                "year": year,
                "month": month,
                "month_name": self.MONTH_NAMES.get(month, "Unknown"),
                "season": self.SEASON_MAP.get(month, "Unknown")
            }
        except (ValueError, IndexError):
            return {
                "year": None,
                "month": None,
                "month_name": "Unknown",
                "season": "Unknown"
            }
    
    def extract_rating_tier(self, rating: int) -> str:
        """
        Categorize rating into tiers:
        1-2: negative, 3: neutral, 4-5: positive
        """
        try:
            rating = int(rating)
            if rating <= 2:
                return "negative"
            elif rating == 3:
                return "neutral"
            else:
                return "positive"
        except (ValueError, TypeError):
            return "unknown"
    
    def extract_topics(self, review_text: str) -> List[str]:
        """
        Extract topics from review text based on keyword matching.
        Returns list of detected topics.
        """
        if not review_text:
            return []
        
        text_lower = review_text.lower()
        detected_topics = []
        
        for topic, keywords in self.topic_keywords.get('topics', {}).items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected_topics.append(topic)
                    break  # Found this topic, move to next
        
        return sorted(list(set(detected_topics)))  # Unique and sorted
    
    def extract_all_features(self, row: Dict) -> Dict:
        """
        Extract all features from a review row.
        
        Args:
            row: Dictionary with keys: Review_ID, Rating, Year_Month, 
                 Reviewer_Location, Review_Text, Branch
        
        Returns:
            Dictionary with all extracted features
        """
        temporal = self.extract_temporal_features(row.get('Year_Month'))
        
        return {
            'review_id': row.get('Review_ID'),
            'rating': row.get('Rating'),
            'rating_tier': self.extract_rating_tier(row.get('Rating')),
            'park': self.extract_park(row.get('Branch')),
            'country': self.extract_country(row.get('Reviewer_Location')),
            'year': temporal['year'],
            'month': temporal['month'],
            'month_name': temporal['month_name'],
            'season': temporal['season'],
            'year_month': row.get('Year_Month'),
            'review_text': row.get('Review_Text', ''),
            'topics': self.extract_topics(row.get('Review_Text', ''))
        }


if __name__ == "__main__":
    # Test the feature extractor
    extractor = FeatureExtractor()
    
    test_row = {
        'Review_ID': '670772142',
        'Rating': 4,
        'Year_Month': '2019-4',
        'Reviewer_Location': 'Australia',
        'Review_Text': 'Great park! The staff was friendly but the food was expensive.',
        'Branch': 'Disneyland_HongKong'
    }
    
    features = extractor.extract_all_features(test_row)
    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

