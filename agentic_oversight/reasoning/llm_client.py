"""
LLM client for the agentic oversight system.
"""

import logging
import json
import time
import os
from typing import Dict, List, Any, Optional, Union
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with LLM providers."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.provider = settings.llm.provider
        self.model = settings.llm.model
        self.api_key = os.environ.get(settings.llm.api_key_env_var)
        self.api_base_url = settings.llm.api_base_url
        self.max_retries = settings.llm.max_retries
        self.timeout_seconds = settings.llm.timeout_seconds
        self.streaming = settings.llm.streaming
        self.fallback_provider = settings.llm.fallback_provider
        self.fallback_model = settings.llm.fallback_model
        self.cache_responses = settings.llm.cache_responses
        self.cache_ttl_seconds = settings.llm.cache_ttl_seconds
        self.max_context_length = settings.llm.max_context_length
        self.token_limit_buffer = settings.llm.token_limit_buffer
        self.cost_tracking_enabled = settings.llm.cost_tracking_enabled
        
        # Initialize response cache
        self.response_cache = {}
        
        # Initialize cost tracking
        self.cost_tracker = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "requests": 0
        }
        
        # Set up API base URL if not provided
        if not self.api_base_url:
            if self.provider == "openai":
                self.api_base_url = "https://api.openai.com/v1"
            elif self.provider == "anthropic":
                self.api_base_url = "https://api.anthropic.com/v1"
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate LLM client configuration."""
        if not self.api_key:
            logger.warning(f"API key not found in environment variable {settings.llm.api_key_env_var}")
        
        if not self.provider:
            raise ValueError("LLM provider not specified")
        
        if not self.model:
            raise ValueError("LLM model not specified")
    
    def _get_cache_key(self, messages, temperature, max_tokens):
        """Get cache key for response caching.
        
        Args:
            messages: Messages
            temperature: Temperature
            max_tokens: Maximum tokens
            
        Returns:
            str: Cache key
        """
        # Create a cache key based on messages, temperature, and max_tokens
        cache_key = json.dumps({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        return hash(cache_key)
    
    def _is_cached_response_valid(self, cache_key):
        """Check if cached response is valid.
        
        Args:
            cache_key: Cache key
            
        Returns:
            bool: Whether cached response is valid
        """
        if cache_key not in self.response_cache:
            return False
        
        cached_response = self.response_cache[cache_key]
        current_time = time.time()
        
        # Check if cached response has expired
        if current_time - cached_response["timestamp"] > self.cache_ttl_seconds:
            return False
        
        return True
    
    def _update_cost_tracker(self, usage):
        """Update cost tracker.
        
        Args:
            usage: Token usage information
        """
        if not self.cost_tracking_enabled:
            return
        
        # Update token counts
        self.cost_tracker["total_tokens"] += usage.get("total_tokens", 0)
        self.cost_tracker["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.cost_tracker["completion_tokens"] += usage.get("completion_tokens", 0)
        self.cost_tracker["requests"] += 1
        
        # Calculate cost based on provider and model
        cost = 0.0
        
        if self.provider == "openai":
            if self.model == "gpt-4":
                # GPT-4 pricing (approximate)
                cost = (usage.get("prompt_tokens", 0) * 0.00003) + (usage.get("completion_tokens", 0) * 0.00006)
            elif self.model == "gpt-3.5-turbo":
                # GPT-3.5 Turbo pricing (approximate)
                cost = (usage.get("prompt_tokens", 0) * 0.000002) + (usage.get("completion_tokens", 0) * 0.000002)
        
        self.cost_tracker["total_cost"] += cost
    
    def get_cost_summary(self):
        """Get cost summary.
        
        Returns:
            Dict[str, Any]: Cost summary
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "total_tokens": self.cost_tracker["total_tokens"],
            "prompt_tokens": self.cost_tracker["prompt_tokens"],
            "completion_tokens": self.cost_tracker["completion_tokens"],
            "total_cost": self.cost_tracker["total_cost"],
            "requests": self.cost_tracker["requests"]
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
    )
    def _call_openai_api(self, messages, temperature, max_tokens):
        """Call OpenAI API.
        
        Args:
            messages: Messages
            temperature: Temperature
            max_tokens: Maximum tokens
            
        Returns:
            Dict[str, Any]: API response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{self.api_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout_seconds
        )
        
        response.raise_for_status()
        return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
    )
    def _call_anthropic_api(self, messages, temperature, max_tokens):
        """Call Anthropic API.
        
        Args:
            messages: Messages
            temperature: Temperature
            max_tokens: Maximum tokens
            
        Returns:
            Dict[str, Any]: API response
        """
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        
        # Convert messages to Anthropic format
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"Human: <system>{message['content']}</system>\n\n"
            elif message["role"] == "user":
                prompt += f"Human: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        
        prompt += "Assistant: "
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens_to_sample": max_tokens
        }
        
        response = requests.post(
            f"{self.api_base_url}/complete",
            headers=headers,
            json=data,
            timeout=self.timeout_seconds
        )
        
        response.raise_for_status()
        return response.json()
    
    def _call_local_api(self, messages, temperature, max_tokens):
        """Call local API.
        
        Args:
            messages: Messages
            temperature: Temperature
            max_tokens: Maximum tokens
            
        Returns:
            Dict[str, Any]: API response
        """
        # This would be implemented for local LLM deployment
        raise NotImplementedError("Local LLM API not implemented")
    
    def _format_response(self, api_response):
        """Format API response.
        
        Args:
            api_response: API response
            
        Returns:
            Dict[str, Any]: Formatted response
        """
        if self.provider == "openai":
            return {
                "content": api_response["choices"][0]["message"]["content"],
                "usage": api_response["usage"]
            }
        elif self.provider == "anthropic":
            # Anthropic doesn't provide token usage in the same format
            return {
                "content": api_response["completion"],
                "usage": {
                    "total_tokens": api_response.get("usage", {}).get("total_tokens", 0),
                    "prompt_tokens": 0,  # Not provided by Anthropic
                    "completion_tokens": 0  # Not provided by Anthropic
                }
            }
        else:
            # Generic format for other providers
            return {
                "content": api_response.get("content", ""),
                "usage": api_response.get("usage", {})
            }
    
    def generate(self, messages, temperature=0.2, max_tokens=2000):
        """Generate response from LLM.
        
        Args:
            messages: Messages
            temperature: Temperature
            max_tokens: Maximum tokens
            
        Returns:
            str: Generated response
        """
        # Check cache if enabled
        if self.cache_responses:
            cache_key = self._get_cache_key(messages, temperature, max_tokens)
            if self._is_cached_response_valid(cache_key):
                logger.debug("Using cached LLM response")
                return self.response_cache[cache_key]["response"]
        
        try:
            # Call appropriate API based on provider
            if self.provider == "openai":
                api_response = self._call_openai_api(messages, temperature, max_tokens)
            elif self.provider == "anthropic":
                api_response = self._call_anthropic_api(messages, temperature, max_tokens)
            elif self.provider == "local":
                api_response = self._call_local_api(messages, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
            # Format response
            formatted_response = self._format_response(api_response)
            
            # Update cost tracker
            self._update_cost_tracker(formatted_response["usage"])
            
            # Cache response if enabled
            if self.cache_responses:
                cache_key = self._get_cache_key(messages, temperature, max_tokens)
                self.response_cache[cache_key] = {
                    "response": formatted_response["content"],
                    "timestamp": time.time()
                }
            
            return formatted_response["content"]
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            
            # Try fallback provider if configured
            if self.fallback_provider and self.fallback_model:
                logger.info(f"Trying fallback provider: {self.fallback_provider}")
                
                # Save current provider and model
                original_provider = self.provider
                original_model = self.model
                
                # Set fallback provider and model
                self.provider = self.fallback_provider
                self.model = self.fallback_model
                
                try:
                    # Call fallback provider
                    response = self.generate(messages, temperature, max_tokens)
                    
                    # Restore original provider and model
                    self.provider = original_provider
                    self.model = original_model
                    
                    return response
                except Exception as fallback_error:
                    logger.error(f"Error with fallback provider: {fallback_error}")
                    
                    # Restore original provider and model
                    self.provider = original_provider
                    self.model = original_model
                    
                    # Re-raise original error
                    raise e
            
            # No fallback or fallback failed
            raise

# Create LLM client instance
llm_client = LLMClient()
