"""
News data collector for NewsAPI.
"""

import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from ..config.settings import settings
from ..models.base import NewsItem, NewsSource, SentimentLevel
from ..storage.redis_client import news_data_queue
from .base import BaseCollector, CollectorRegistry

logger = logging.getLogger(__name__)

class NewsAPICollector(BaseCollector):
    """Collector for NewsAPI data."""
    
    def __init__(self):
        """Initialize NewsAPI collector."""
        super().__init__("news_data", "newsapi", news_data_queue)
        self.api_key = settings.news_data.newsapi_key
        self.base_url = "https://newsapi.org/v2"
        
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
    
    async def _fetch_news(self, query: str, sources: Optional[List[str]] = None,
                         from_date: Optional[datetime.datetime] = None,
                         to_date: Optional[datetime.datetime] = None,
                         page_size: int = 100, page: int = 1) -> Dict[str, Any]:
        """Fetch news from NewsAPI.
        
        Args:
            query: Search query
            sources: List of news sources
            from_date: Start date
            to_date: End date
            page_size: Number of results per page
            page: Page number
            
        Returns:
            Dict[str, Any]: API response
        """
        url = f"{self.base_url}/everything"
        
        params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": page_size,
            "page": page,
            "language": "en",
            "sortBy": "publishedAt"
        }
        
        if sources:
            params["sources"] = ",".join(sources)
            
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%d")
            
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%d")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"NewsAPI error: {response.status} - {error_text}")
                
                return await response.json()
    
    def _extract_symbols(self, title: str, content: str) -> List[str]:
        """Extract stock symbols from text.
        
        This is a simple implementation that looks for $ followed by uppercase letters.
        A more sophisticated implementation would use NER from the semantic signal generator.
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            List[str]: Extracted symbols
        """
        import re
        
        # Look for $SYMBOL pattern
        cashtag_pattern = r'\$([A-Z]{1,5})'
        
        # Combine title and content
        full_text = f"{title} {content}"
        
        # Find all matches
        matches = re.findall(cashtag_pattern, full_text)
        
        # Remove duplicates
        unique_symbols = list(set(matches))
        
        return unique_symbols
    
    def _map_source(self, source_name: str) -> NewsSource:
        """Map source name to enum.
        
        Args:
            source_name: Source name from API
            
        Returns:
            NewsSource: Source enum
        """
        source_map = {
            "bloomberg": NewsSource.BLOOMBERG,
            "reuters": NewsSource.REUTERS,
            "financial-times": NewsSource.FT,
            "the-wall-street-journal": NewsSource.WSJ,
            "cnbc": NewsSource.CNBC
        }
        
        return source_map.get(source_name.lower(), NewsSource.CUSTOM)
    
    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process articles into NewsItem models.
        
        Args:
            articles: List of articles from API
            
        Returns:
            List[Dict]: List of NewsItem dictionaries
        """
        results = []
        
        for article in articles:
            try:
                # Extract basic fields
                title = article.get("title", "")
                content = article.get("description", "") or article.get("content", "")
                url = article.get("url", "")
                source_name = article.get("source", {}).get("name", "")
                author = article.get("author", "")
                
                # Parse timestamp
                published_at = article.get("publishedAt")
                if published_at:
                    timestamp = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.datetime.utcnow()
                
                # Extract symbols
                symbols = self._extract_symbols(title, content)
                
                # Map source
                source = self._map_source(source_name)
                
                # Create NewsItem model
                news_item = NewsItem(
                    title=title,
                    content=content,
                    source=source,
                    url=url,
                    timestamp=timestamp,
                    author=author,
                    symbols=symbols,
                    # Sentiment will be added by the semantic signal generator
                    sentiment=None,
                    sentiment_score=None
                )
                
                results.append(news_item.dict())
            except Exception as e:
                logger.error(f"Error processing article: {e}")
        
        return results
    
    async def collect(self, query: Optional[str] = None, 
                     sources: Optional[List[str]] = None,
                     lookback_days: int = 1) -> Dict[str, Any]:
        """Collect news from NewsAPI.
        
        Args:
            query: Search query (defaults to financial terms)
            sources: List of news sources (defaults to settings)
            lookback_days: Days to look back (defaults to 1)
            
        Returns:
            Dict[str, Any]: Collection results
        """
        if not self.api_key:
            return {"success": False, "error": "NewsAPI key not configured", "records_processed": 0}
            
        if not query:
            # Default query for financial news
            query = "finance OR stocks OR market OR economy OR earnings OR investing"
            
        if not sources:
            sources = settings.news_data.default_news_sources
            
        # Calculate time range
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=lookback_days)
        
        try:
            # Fetch news
            news_data = await self._fetch_news(query, sources, start, end)
            
            # Extract articles
            articles = news_data.get("articles", [])
            
            # Process articles
            processed_data = self._process_articles(articles)
            
            # Publish to queue
            if processed_data:
                self._publish_data(processed_data)
            
            return {
                "success": True,
                "records_processed": len(processed_data),
                "query": query,
                "sources": sources,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "total_results": news_data.get("totalResults", 0)
            }
        except Exception as e:
            logger.exception("Error collecting NewsAPI data")
            return {
                "success": False,
                "error": str(e),
                "records_processed": 0
            }

# Register collector
CollectorRegistry.register("news_data", "newsapi", NewsAPICollector)
