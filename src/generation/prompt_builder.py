"""
Prompt builder for RAG system.
Constructs prompts with context, citations, and instructions.
"""

from typing import List, Dict


class PromptBuilder:
    """Build prompts for LLM synthesis."""
    
    SYSTEM_PROMPT = """You are a helpful assistant analyzing Disney park reviews.
    
Your role:
- Answer questions based ONLY on the provided review excerpts
- Cite specific review IDs for every claim using [Review ID: xxx]
- Be concise and factual
- If the context doesn't contain enough information, say so
- Summarize sentiment patterns when multiple reviews express similar views

Guidelines:
- Use direct quotes when highlighting specific experiences
- Mention geographic/temporal patterns if relevant (e.g., "Australian visitors in summer...")
- Balance positive and negative feedback
- Never make up information not in the reviews"""
    
    USER_PROMPT_TEMPLATE = """Question: {query}

Review Context:
{context}

Instructions:
1. Answer based ONLY on the reviews above
2. Cite review IDs for all claims: [Review ID: xxx]
3. If multiple reviews say similar things, summarize the pattern
4. Be specific about which parks, time periods, or visitor types if relevant
5. If you cannot answer from the context, say "The available reviews don't provide enough information about..."

Answer:"""
    
    @staticmethod
    def build_context(results: List[Dict]) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            results: List of retrieval results with metadata
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            review_id = result.get('review_id', 'unknown')
            chunk_text = result.get('chunk_text', '')
            park = result.get('park', 'Unknown')
            country = result.get('country', 'Unknown')
            season = result.get('season', 'Unknown')
            rating = result.get('rating', 'N/A')
            
            # Format: [Review ID] (Park, Country, Season, Rating★): text
            context_part = f"""[Review ID: {review_id}] ({park}, {country}, {season}, {rating}★):
{chunk_text}"""
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    @classmethod
    def build_prompt(cls, query: str, results: List[Dict]) -> tuple[str, str]:
        """
        Build full prompt with system and user messages.
        
        Args:
            query: User query
            results: Retrieval results
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        context = cls.build_context(results)
        user_prompt = cls.USER_PROMPT_TEMPLATE.format(
            query=query,
            context=context
        )
        
        return cls.SYSTEM_PROMPT, user_prompt


if __name__ == "__main__":
    # Test prompt builder
    test_results = [
        {
            'review_id': 'R123',
            'chunk_text': 'The staff was incredibly friendly and helpful throughout our visit.',
            'park': 'Hong Kong',
            'country': 'Australia',
            'season': 'Summer',
            'rating': 5
        },
        {
            'review_id': 'R456',
            'chunk_text': 'Staff seemed overwhelmed and not very attentive.',
            'park': 'Paris',
            'country': 'France',
            'season': 'Winter',
            'rating': 2
        }
    ]
    
    system_prompt, user_prompt = PromptBuilder.build_prompt(
        query="Is the staff friendly?",
        results=test_results
    )
    
    print("System Prompt:")
    print("=" * 80)
    print(system_prompt)
    print("\n" + "=" * 80)
    print("User Prompt:")
    print("=" * 80)
    print(user_prompt[:500] + "...")

