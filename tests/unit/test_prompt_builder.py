"""Unit tests for prompt builder."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generation.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """Test prompt builder functionality."""
    
    def test_build_context_single_result(self):
        """Test context building with single result."""
        results = [{
            'review_id': 'R123',
            'chunk_text': 'Great experience!',
            'park': 'Hong Kong',
            'country': 'Australia',
            'season': 'Summer',
            'rating': 5
        }]
        
        context = PromptBuilder.build_context(results)
        
        assert 'R123' in context
        assert 'Hong Kong' in context
        assert 'Australia' in context
        assert 'Great experience!' in context
    
    def test_build_context_multiple_results(self):
        """Test context building with multiple results."""
        results = [
            {
                'review_id': 'R1',
                'chunk_text': 'Text 1',
                'park': 'Paris',
                'country': 'France',
                'season': 'Winter',
                'rating': 4
            },
            {
                'review_id': 'R2',
                'chunk_text': 'Text 2',
                'park': 'California',
                'country': 'USA',
                'season': 'Summer',
                'rating': 5
            }
        ]
        
        context = PromptBuilder.build_context(results)
        
        assert 'R1' in context
        assert 'R2' in context
        assert context.count('[Review ID:') == 2
    
    def test_build_prompt(self):
        """Test full prompt building."""
        results = [{
            'review_id': 'R123',
            'chunk_text': 'Great!',
            'park': 'Hong Kong',
            'country': 'Australia',
            'season': 'Summer',
            'rating': 5
        }]
        
        system_prompt, user_prompt = PromptBuilder.build_prompt(
            query="Is it good?",
            results=results
        )
        
        assert 'helpful assistant' in system_prompt.lower()
        assert 'Is it good?' in user_prompt
        assert 'R123' in user_prompt
        assert 'Great!' in user_prompt

