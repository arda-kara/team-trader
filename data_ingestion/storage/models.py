"""
Database models for the data ingestion layer.
"""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()

# Enum classes for SQLAlchemy
class AssetTypeEnum(enum.Enum):
    STOCK = "stock"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"
    ETF = "etf"
    BOND = "bond"

class DataSourceEnum(enum.Enum):
    ALPACA = "alpaca"
    BINANCE = "binance"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    CUSTOM = "custom"

class TimeFrameEnum(enum.Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class NewsSourceEnum(enum.Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    FT = "financial-times"
    WSJ = "wall-street-journal"
    CNBC = "cnbc"
    NEWSAPI = "newsapi"
    CUSTOM = "custom"

class SentimentLevelEnum(enum.Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

# Database Models
class Asset(Base):
    """Asset information table."""
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    name = Column(String(100))
    asset_type = Column(Enum(AssetTypeEnum), nullable=False)
    exchange = Column(String(50))
    currency = Column(String(10))
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Relationships
    ohlcv_data = relationship("OHLCV", back_populates="asset")
    trades = relationship("Trade", back_populates="asset")
    quotes = relationship("Quote", back_populates="asset")
    
    def __repr__(self):
        return f"<Asset(symbol='{self.symbol}', type='{self.asset_type}')>"

class OHLCV(Base):
    """OHLCV (Open, High, Low, Close, Volume) data table."""
    __tablename__ = "ohlcv_data"
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(Enum(TimeFrameEnum), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    source = Column(Enum(DataSourceEnum), nullable=False)
    
    # Relationships
    asset = relationship("Asset", back_populates="ohlcv_data")
    
    def __repr__(self):
        return f"<OHLCV(asset_id={self.asset_id}, timestamp='{self.timestamp}', close={self.close})>"

class Trade(Base):
    """Individual trade data table."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    trade_id = Column(String(50))
    source = Column(Enum(DataSourceEnum), nullable=False)
    
    # Relationships
    asset = relationship("Asset", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(asset_id={self.asset_id}, timestamp='{self.timestamp}', price={self.price})>"

class Quote(Base):
    """Bid/ask quote data table."""
    __tablename__ = "quotes"
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    bid_price = Column(Float, nullable=False)
    ask_price = Column(Float, nullable=False)
    bid_size = Column(Float)
    ask_size = Column(Float)
    source = Column(Enum(DataSourceEnum), nullable=False)
    
    # Relationships
    asset = relationship("Asset", back_populates="quotes")
    
    def __repr__(self):
        return f"<Quote(asset_id={self.asset_id}, timestamp='{self.timestamp}', bid={self.bid_price}, ask={self.ask_price})>"

class NewsItem(Base):
    """News item data table."""
    __tablename__ = "news_items"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(Enum(NewsSourceEnum), nullable=False)
    url = Column(String(512), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    author = Column(String(100))
    symbols = Column(JSON)  # Store as JSON array
    sentiment = Column(Enum(SentimentLevelEnum))
    sentiment_score = Column(Float)
    created_at = Column(DateTime)
    
    def __repr__(self):
        return f"<NewsItem(id={self.id}, title='{self.title[:30]}...', source='{self.source}')>"

class SocialMediaPost(Base):
    """Social media post data table."""
    __tablename__ = "social_media_posts"
    
    id = Column(Integer, primary_key=True)
    platform = Column(String(50), nullable=False)
    user_id = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    url = Column(String(512))
    symbols = Column(JSON)  # Store as JSON array
    sentiment = Column(Enum(SentimentLevelEnum))
    sentiment_score = Column(Float)
    engagement = Column(JSON)  # Store as JSON object
    created_at = Column(DateTime)
    
    def __repr__(self):
        return f"<SocialMediaPost(id={self.id}, platform='{self.platform}', user='{self.user_id}')>"

class EconomicIndicator(Base):
    """Economic indicator data table."""
    __tablename__ = "economic_indicators"
    
    id = Column(Integer, primary_key=True)
    indicator_id = Column(String(50), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), nullable=False)
    frequency = Column(String(20), nullable=False)
    unit = Column(String(20))
    region = Column(String(50))
    created_at = Column(DateTime)
    
    def __repr__(self):
        return f"<EconomicIndicator(indicator='{self.indicator_id}', timestamp='{self.timestamp}', value={self.value})>"

class DataCollectionLog(Base):
    """Log table for data collection activities."""
    __tablename__ = "data_collection_logs"
    
    id = Column(Integer, primary_key=True)
    collector_type = Column(String(50), nullable=False)
    source = Column(String(50), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    status = Column(String(20), nullable=False)
    records_processed = Column(Integer, default=0)
    error_message = Column(Text)
    
    def __repr__(self):
        return f"<DataCollectionLog(id={self.id}, type='{self.collector_type}', status='{self.status}')>"
