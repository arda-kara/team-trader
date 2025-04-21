"""
Redis client for message queue and caching in the data ingestion layer.
"""

import json
from typing import Any, Dict, List, Optional, Union
import redis

from ..config.settings import settings

# Create Redis connection
redis_client = redis.Redis.from_url(
    settings.redis.connection_string,
    decode_responses=True
)

class RedisQueue:
    """Redis-based message queue implementation."""
    
    def __init__(self, queue_name: str):
        """Initialize a Redis queue.
        
        Args:
            queue_name: Name of the queue
        """
        self.queue_name = queue_name
        self.client = redis_client
    
    def push(self, data: Union[Dict, List, str]) -> bool:
        """Push data to the queue.
        
        Args:
            data: Data to push (will be JSON serialized)
            
        Returns:
            bool: Success status
        """
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        return bool(self.client.lpush(self.queue_name, data))
    
    def pop(self) -> Optional[str]:
        """Pop data from the queue.
        
        Returns:
            Optional[str]: Data or None if queue is empty
        """
        return self.client.rpop(self.queue_name)
    
    def pop_blocking(self, timeout: int = 0) -> Optional[str]:
        """Pop data from the queue with blocking.
        
        Args:
            timeout: Timeout in seconds (0 = infinite)
            
        Returns:
            Optional[str]: Data or None if timeout reached
        """
        result = self.client.brpop(self.queue_name, timeout)
        if result:
            return result[1]
        return None
    
    def length(self) -> int:
        """Get queue length.
        
        Returns:
            int: Number of items in queue
        """
        return self.client.llen(self.queue_name)
    
    def clear(self) -> bool:
        """Clear the queue.
        
        Returns:
            bool: Success status
        """
        return bool(self.client.delete(self.queue_name))

class RedisCache:
    """Redis-based caching implementation."""
    
    def __init__(self, prefix: str = "cache"):
        """Initialize a Redis cache.
        
        Args:
            prefix: Key prefix for cache entries
        """
        self.prefix = prefix
        self.client = redis_client
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key.
        
        Args:
            key: Original key
            
        Returns:
            str: Prefixed key
        """
        return f"{self.prefix}:{key}"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized if dict/list)
            ttl: Time-to-live in seconds (None = no expiry)
            
        Returns:
            bool: Success status
        """
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        prefixed_key = self._make_key(key)
        if ttl is not None:
            return bool(self.client.setex(prefixed_key, ttl, value))
        return bool(self.client.set(prefixed_key, value))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cache value.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Any: Cached value or default
        """
        prefixed_key = self._make_key(key)
        value = self.client.get(prefixed_key)
        
        if value is None:
            return default
        
        # Try to parse as JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    def delete(self, key: str) -> bool:
        """Delete cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            bool: Success status
        """
        prefixed_key = self._make_key(key)
        return bool(self.client.delete(prefixed_key))
    
    def exists(self, key: str) -> bool:
        """Check if key exists.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key exists
        """
        prefixed_key = self._make_key(key)
        return bool(self.client.exists(prefixed_key))
    
    def ttl(self, key: str) -> int:
        """Get remaining TTL for key.
        
        Args:
            key: Cache key
            
        Returns:
            int: Remaining TTL in seconds (-1 if no TTL, -2 if key doesn't exist)
        """
        prefixed_key = self._make_key(key)
        return self.client.ttl(prefixed_key)

# Create default instances
market_data_queue = RedisQueue("market_data")
news_data_queue = RedisQueue("news_data")
economic_data_queue = RedisQueue("economic_data")
data_cache = RedisCache("data_cache")
