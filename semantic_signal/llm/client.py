"""
LLM client for integrating with various LLM providers.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import openai
import anthropic
import requests

from ..config.settings import settings
from ..processors.redis_client import llm_cache

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.interval:
            sleep_time = self.interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

class LLMClient:
    """Client for interacting with LLM providers."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.openai_api_key = settings.llm.openai_api_key
        self.anthropic_api_key = settings.llm.anthropic_api_key
        self.mistral_api_key = settings.llm.mistral_api_key
        
        self.default_provider = settings.llm.default_provider
        self.enable_caching = settings.llm.enable_caching
        self.cache_ttl = settings.llm.cache_ttl
        
        # Set up rate limiters
        self.rate_limiter = RateLimiter(settings.llm.rate_limit_requests)
        
        # Initialize clients
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        if self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
    
    def _get_cache_key(self, provider: str, model: str, messages: List[Dict[str, str]], 
                      max_tokens: int) -> str:
        """Generate cache key for request.
        
        Args:
            provider: LLM provider
            model: Model name
            messages: Messages for chat completion
            max_tokens: Maximum tokens
            
        Returns:
            str: Cache key
        """
        # Create a deterministic representation of the request
        cache_dict = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        
        # Convert to JSON string and hash
        cache_str = json.dumps(cache_dict, sort_keys=True)
        import hashlib
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
        
        return f"llm:{cache_hash}"
    
    async def generate_text(self, prompt: str, max_tokens: int = 1000, 
                          provider: Optional[str] = None, model: Optional[str] = None) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            provider: LLM provider (defaults to settings)
            model: Model name (defaults to settings)
            
        Returns:
            str: Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.chat_completion(messages, max_tokens, provider, model)
        return response
    
    async def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1000,
                            provider: Optional[str] = None, model: Optional[str] = None) -> str:
        """Generate chat completion.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            provider: LLM provider (defaults to settings)
            model: Model name (defaults to settings)
            
        Returns:
            str: Generated text
        """
        # Set defaults
        if not provider:
            provider = self.default_provider
            
        if not model:
            if provider == "openai":
                model = settings.llm.openai_model
            elif provider == "anthropic":
                model = settings.llm.anthropic_model
            elif provider == "mistral":
                model = settings.llm.mistral_model
        
        # Check cache
        if self.enable_caching:
            cache_key = self._get_cache_key(provider, model, messages, max_tokens)
            cached_response = llm_cache.get(cache_key)
            
            if cached_response:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_response
        
        # Apply rate limiting
        self.rate_limiter.wait()
        
        # Call appropriate provider
        try:
            if provider == "openai":
                response = await self._openai_chat_completion(model, messages, max_tokens)
            elif provider == "anthropic":
                response = await self._anthropic_chat_completion(model, messages, max_tokens)
            elif provider == "mistral":
                response = await self._mistral_chat_completion(model, messages, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Cache response
            if self.enable_caching:
                llm_cache.set(cache_key, response, ttl=self.cache_ttl)
            
            return response
        except Exception as e:
            logger.exception(f"Error in LLM chat completion: {e}")
            raise
    
    async def _openai_chat_completion(self, model: str, messages: List[Dict[str, str]], 
                                    max_tokens: int) -> str:
        """Generate chat completion using OpenAI.
        
        Args:
            model: Model name
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.exception(f"OpenAI API error: {e}")
            raise
    
    async def _anthropic_chat_completion(self, model: str, messages: List[Dict[str, str]], 
                                       max_tokens: int) -> str:
        """Generate chat completion using Anthropic.
        
        Args:
            model: Model name
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        
        try:
            # Convert messages to Anthropic format
            system_message = None
            prompt = ""
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    system_message = content
                elif role == "user":
                    prompt += f"\n\nHuman: {content}"
                elif role == "assistant":
                    prompt += f"\n\nAssistant: {content}"
            
            # Add final assistant prompt
            prompt += "\n\nAssistant:"
            
            # Create message
            response = self.anthropic_client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens_to_sample=max_tokens,
                temperature=0.7,
                top_p=1.0,
                system=system_message
            )
            
            return response.completion.strip()
        except Exception as e:
            logger.exception(f"Anthropic API error: {e}")
            raise
    
    async def _mistral_chat_completion(self, model: str, messages: List[Dict[str, str]], 
                                     max_tokens: int) -> str:
        """Generate chat completion using Mistral.
        
        Args:
            model: Model name
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        if not self.mistral_api_key:
            raise ValueError("Mistral API key not configured")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 1.0
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.exception(f"Mistral API error: {e}")
            raise

# Create client instance
llm_client = LLMClient()
