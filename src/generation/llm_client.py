"""
LLM client for OpenAI GPT-4o-mini with token counting and circuit breaker.
"""

import os
import time
from typing import Dict, Optional
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv


class LLMClient:
    """OpenAI client with guardrails."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000,
        temperature: float = 0.3,
        max_context_tokens: int = 6000,
        max_retries: int = 3
    ):
        """
        Initialize LLM client.
        
        Args:
            model: OpenAI model name
            max_tokens: Max tokens in response
            temperature: Sampling temperature (0-1)
            max_context_tokens: Max tokens in prompt (circuit breaker)
            max_retries: Max retry attempts on failure
        """
        # Load API key from .env
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.max_retries = max_retries
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for newer models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print(f"✅ LLM Client initialized: {model}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def check_token_limit(self, system_prompt: str, user_prompt: str) -> bool:
        """
        Check if prompt exceeds token limit.
        
        Args:
            system_prompt: System message
            user_prompt: User message
        
        Returns:
            True if within limit, False otherwise
        """
        total_tokens = (
            self.count_tokens(system_prompt) +
            self.count_tokens(user_prompt)
        )
        
        return total_tokens <= self.max_context_tokens
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        return_metadata: bool = True
    ) -> Dict:
        """
        Generate completion with retry logic and metadata.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            return_metadata: Whether to return metadata (tokens, latency)
        
        Returns:
            Dictionary with response and metadata
        """
        # Check token limit
        if not self.check_token_limit(system_prompt, user_prompt):
            prompt_tokens = (
                self.count_tokens(system_prompt) +
                self.count_tokens(user_prompt)
            )
            return {
                'response': None,
                'error': 'Token limit exceeded',
                'prompt_tokens': prompt_tokens,
                'max_tokens': self.max_context_tokens
            }
        
        # Retry loop
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract response
                answer = response.choices[0].message.content
                
                # Metadata
                if return_metadata:
                    return {
                        'response': answer,
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens,
                        'latency_ms': latency_ms,
                        'model': self.model,
                        'finish_reason': response.choices[0].finish_reason
                    }
                else:
                    return {'response': answer}
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return {
                        'response': None,
                        'error': str(e)
                    }


if __name__ == "__main__":
    # Test LLM client
    print("Testing LLM Client...")
    
    client = LLMClient()
    
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is 2+2?"
    
    print(f"\nQuery: {user_prompt}")
    print("Calling OpenAI API...")
    
    result = client.generate(system_prompt, user_prompt)
    
    if result.get('response'):
        print(f"\n✅ Response: {result['response']}")
        print(f"\nMetadata:")
        print(f"  Prompt tokens: {result['prompt_tokens']}")
        print(f"  Completion tokens: {result['completion_tokens']}")
        print(f"  Latency: {result['latency_ms']:.2f}ms")
    else:
        print(f"\n❌ Error: {result.get('error')}")

