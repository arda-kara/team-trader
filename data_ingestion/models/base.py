"""
Base models for data ingestion layer.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

# Enums
class AssetType(str, Enum):
    """Asset type enumeration."""
    STOCK = "stock"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"
    ETF = "etf"
    BOND = "bond"

class DataSource(str, Enum):
    """Data source enumeration."""
    ALPACA = "alpaca"
    BINANCE = "binance"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    CUSTOM = "custom"

class TimeFrame(str, Enum):
    """Time frame enumeration."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class NewsSource(str, Enum):
    """News source enumeration."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    FT = "financial-times"
    WSJ = "wall-street-journal"
    CNBC = "cnbc"
    NEWSAPI = "newsapi"
    CUSTOM = "custom"

class SentimentLevel(str, Enum):
    """Sentiment level enumeration."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

# Base Models
class BaseMarketData(BaseModel):
    """Base model for market data."""
    symbol: str
    asset_type: AssetType
    source: DataSource
    timestamp: datetime
    timeframe: TimeFrame

class OHLCV(BaseMarketData):
    """Open, High, Low, Close, Volume data."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "asset_type": "stock",
                "source": "alpaca",
                "timestamp": "2023-01-01T00:00:00Z",
                "timeframe": "1d",
                "open": 150.0,
                "high": 152.5,
                "low": 149.0,
                "close": 151.75,
                "volume": 1000000
            }
        }

class Trade(BaseMarketData):
    """Individual trade data."""
    price: float
    size: float
    trade_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "asset_type": "stock",
                "source": "alpaca",
                "timestamp": "2023-01-01T12:34:56Z",
                "timeframe": "1m",
                "price": 151.25,
                "size": 100,
                "trade_id": "T123456"
            }
        }

class Quote(BaseMarketData):
    """Bid/ask quote data."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "asset_type": "stock",
                "source": "alpaca",
                "timestamp": "2023-01-01T12:34:56Z",
                "timeframe": "1m",
                "bid_price": 151.0,
                "ask_price": 151.5,
                "bid_size": 200,
                "ask_size": 150
            }
        }

class NewsItem(BaseModel):
    """News item model."""
    title: str
    content: str
    source: NewsSource
    url: str
    timestamp: datetime
    author: Optional[str] = None
    symbols: List[str] = []
    sentiment: Optional[SentimentLevel] = None
    sentiment_score: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Apple Reports Record Quarterly Revenue",
                "content": "Apple today announced financial results for its fiscal 2023 first quarter...",
                "source": "bloomberg",
                "url": "https://www.bloomberg.com/news/articles/...",
                "timestamp": "2023-01-01T12:00:00Z",
                "author": "John Doe",
                "symbols": ["AAPL"],
                "sentiment": "positive",
                "sentiment_score": 0.75
            }
        }

class SocialMediaPost(BaseModel):
    """Social media post model."""
    platform: str
    user_id: str
    content: str
    timestamp: datetime
    url: Optional[str] = None
    symbols: List[str] = []
    sentiment: Optional[SentimentLevel] = None
    sentiment_score: Optional[float] = None
    engagement: Optional[Dict[str, int]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "platform": "twitter",
                "user_id": "user123",
                "content": "Just bought more $AAPL, feeling bullish!",
                "timestamp": "2023-01-01T12:34:56Z",
                "url": "https://twitter.com/user123/status/...",
                "symbols": ["AAPL"],
                "sentiment": "positive",
                "sentiment_score": 0.8,
                "engagement": {"likes": 10, "retweets": 2, "replies": 1}
            }
        }

class EconomicIndicator(BaseModel):
    """Economic indicator model."""
    indicator_id: str
    name: str
    value: float
    timestamp: datetime
    source: str
    frequency: str
    unit: Optional[str] = None
    region: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "indicator_id": "UNRATE",
                "name": "Unemployment Rate",
                "value": 3.7,
                "timestamp": "2023-01-01T00:00:00Z",
                "source": "fred",
                "frequency": "monthly",
                "unit": "percent",
                "region": "US"
            }
        }

# Request/Response Models
class TimeRange(BaseModel):
    """Time range for data queries."""
    start: datetime
    end: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "start": "2023-01-01T00:00:00Z",
                "end": "2023-01-31T23:59:59Z"
            }
        }

class MarketDataRequest(BaseModel):
    """Request model for market data."""
    symbols: List[str]
    asset_types: Optional[List[AssetType]] = None
    timeframe: TimeFrame
    time_range: TimeRange
    source: Optional[DataSource] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "asset_types": ["stock"],
                "timeframe": "1d",
                "time_range": {
                    "start": "2023-01-01T00:00:00Z",
                    "end": "2023-01-31T23:59:59Z"
                },
                "source": "alpaca"
            }
        }

class NewsDataRequest(BaseModel):
    """Request model for news data."""
    symbols: Optional[List[str]] = None
    sources: Optional[List[NewsSource]] = None
    time_range: TimeRange
    keywords: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT"],
                "sources": ["bloomberg", "reuters"],
                "time_range": {
                    "start": "2023-01-01T00:00:00Z",
                    "end": "2023-01-31T23:59:59Z"
                },
                "keywords": ["earnings", "revenue", "growth"]
            }
        }

class EconomicDataRequest(BaseModel):
    """Request model for economic data."""
    indicators: List[str]
    time_range: TimeRange
    region: Optional[str] = None
    frequency: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "indicators": ["UNRATE", "GDP", "CPIAUCSL"],
                "time_range": {
                    "start": "2023-01-01T00:00:00Z",
                    "end": "2023-01-31T23:59:59Z"
                },
                "region": "US",
                "frequency": "monthly"
            }
        }

# Response Models
class DataResponse(BaseModel):
    """Generic data response model."""
    success: bool
    message: Optional[str] = None
    count: Optional[int] = None
    next_page: Optional[str] = None

class MarketDataResponse(DataResponse):
    """Response model for market data."""
    data: List[Union[OHLCV, Trade, Quote]]

class NewsDataResponse(DataResponse):
    """Response model for news data."""
    data: List[Union[NewsItem, SocialMediaPost]]

class EconomicDataResponse(DataResponse):
    """Response model for economic data."""
    data: List[EconomicIndicator]
