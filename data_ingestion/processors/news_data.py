"""
News data processor for storing and normalizing news items.
"""

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.exc import IntegrityError

from ..models.base import NewsItem, NewsSource, SentimentLevel
from ..storage.database import get_db_session
from ..storage.models import NewsItem as NewsItemModel
from ..storage.redis_client import news_data_queue, data_cache
from .base import BaseProcessor, ProcessorRegistry

logger = logging.getLogger(__name__)

class NewsDataProcessor(BaseProcessor):
    """Processor for news data."""
    
    def __init__(self):
        """Initialize news data processor."""
        super().__init__("news_data", news_data_queue, None)
    
    async def _store_news_items(self, news_items: List[Dict]) -> int:
        """Store news items in database.
        
        Args:
            news_items: List of news item dictionaries
            
        Returns:
            int: Number of records stored
        """
        stored_count = 0
        
        for data in news_items:
            try:
                # Parse data
                news_item = NewsItem(**data)
                
                # Create database model
                db_model = NewsItemModel(
                    title=news_item.title,
                    content=news_item.content,
                    source=news_item.source.value,
                    url=news_item.url,
                    timestamp=news_item.timestamp,
                    author=news_item.author,
                    symbols=news_item.symbols,
                    sentiment=news_item.sentiment.value if news_item.sentiment else None,
                    sentiment_score=news_item.sentiment_score,
                    created_at=datetime.datetime.utcnow()
                )
                
                # Store in database
                with get_db_session() as session:
                    # Check if record already exists (by URL)
                    existing = session.query(NewsItemModel).filter(
                        NewsItemModel.url == news_item.url
                    ).first()
                    
                    if existing:
                        # Update existing record if needed
                        if news_item.sentiment and not existing.sentiment:
                            existing.sentiment = news_item.sentiment.value
                            existing.sentiment_score = news_item.sentiment_score
                    else:
                        # Add new record
                        session.add(db_model)
                        
                    stored_count += 1
            except Exception as e:
                logger.error(f"Error storing news item: {e}")
        
        return stored_count
    
    async def _update_cache(self, news_items: List[Dict]) -> int:
        """Update cache with latest news data.
        
        Args:
            news_items: List of news item dictionaries
            
        Returns:
            int: Number of records cached
        """
        cached_count = 0
        
        try:
            # Sort by timestamp (newest first)
            sorted_items = sorted(
                news_items, 
                key=lambda x: datetime.datetime.fromisoformat(x["timestamp"]) 
                if isinstance(x["timestamp"], str) 
                else x["timestamp"],
                reverse=True
            )
            
            # Keep only the latest 100 news items
            latest_news = sorted_items[:100]
            
            # Cache with 1 hour TTL
            cache_key = "news_data:latest"
            data_cache.set(cache_key, latest_news, ttl=3600)
            
            # Also cache by symbol
            symbol_groups = {}
            for item in sorted_items:
                symbols = item.get("symbols", [])
                for symbol in symbols:
                    if symbol not in symbol_groups:
                        symbol_groups[symbol] = []
                    symbol_groups[symbol].append(item)
            
            # Cache each symbol group
            for symbol, items in symbol_groups.items():
                # Keep only the latest 50 news items per symbol
                latest_items = items[:50]
                cache_key = f"news_data:symbol:{symbol}"
                data_cache.set(cache_key, latest_items, ttl=3600)
                cached_count += len(latest_items)
                
        except Exception as e:
            logger.error(f"Error updating news cache: {e}")
        
        return cached_count
    
    async def process(self, data: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Process news data.
        
        Args:
            data: News data to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if not isinstance(data, list):
            data = [data]
            
        try:
            # Store in database
            stored_count = await self._store_news_items(data)
            
            # Update cache
            cached_count = await self._update_cache(data)
            
            return {
                "success": True,
                "records_processed": len(data),
                "records_stored": stored_count,
                "records_cached": cached_count
            }
        except Exception as e:
            logger.exception("Error processing news data")
            return {
                "success": False,
                "error": str(e),
                "records_processed": 0
            }

# Register processor
ProcessorRegistry.register("news_data", NewsDataProcessor)
