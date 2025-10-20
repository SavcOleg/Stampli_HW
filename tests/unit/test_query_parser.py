"""Unit tests for query parser."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.query_parser import QueryParser


class TestQueryParser:
    """Test query parser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return QueryParser()
    
    def test_extract_park(self, parser):
        """Test park extraction."""
        assert parser.extract_park("Visit Hong Kong Disneyland") == "Hong Kong"
        assert parser.extract_park("California park") == "California"
        assert parser.extract_park("Disneyland Paris") == "Paris"
        assert parser.extract_park("No park mentioned") is None
    
    def test_extract_season(self, parser):
        """Test season extraction."""
        assert parser.extract_season("Visit in spring") == "Spring"
        assert parser.extract_season("Summer vacation") == "Summer"
        assert parser.extract_season("Fall colors") == "Fall"
        assert parser.extract_season("Winter wonderland") == "Winter"
        assert parser.extract_season("No season") is None
    
    def test_extract_month(self, parser):
        """Test month extraction."""
        assert parser.extract_month("Visit in June") == 6
        assert parser.extract_month("January trip") == 1
        assert parser.extract_month("December holidays") == 12
        assert parser.extract_month("No month") is None
    
    def test_extract_rating_intent(self, parser):
        """Test rating intent extraction."""
        assert parser.extract_rating_intent("Great experience") == "positive"
        assert parser.extract_rating_intent("Terrible visit") == "negative"
        assert parser.extract_rating_intent("It was okay") is None
    
    def test_parse_complex_query(self, parser):
        """Test parsing complex query."""
        result = parser.parse("What do Australian visitors say about Hong Kong in spring?")
        
        assert result['filters']['country'] == "Australia"
        assert result['filters']['park'] == "Hong Kong"
        assert result['filters']['season'] == "Spring"
        assert result['has_filters'] is True

