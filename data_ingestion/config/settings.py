"""
Configuration settings for the Data Ingestion Layer.
"""

import os
from typing import Dict, List, Optional, Union
from pydantic import BaseSettings, Field, validator

class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    host: str = Field("localhost", env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    username: str = Field("postgres", env="DB_USERNAME")
    password: str = Field("postgres", env="DB_PASSWORD")
    database: str = Field("trading_pipeline", env="DB_NAME")
    
    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisSettings(BaseSettings):
    """Redis connection settings."""
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    db: int = Field(0, env="REDIS_DB")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    @property
    def connection_string(self) -> str:
        """Get Redis connection string."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

class MarketDataSettings(BaseSettings):
    """Market data API settings."""
    # Alpaca API settings
    alpaca_api_key: Optional[str] = Field(None, env="ALPACA_API_KEY")
    alpaca_api_secret: Optional[str] = Field(None, env="ALPACA_API_SECRET")
    alpaca_base_url: str = Field("https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    
    # Binance API settings
    binance_api_key: Optional[str] = Field(None, env="BINANCE_API_KEY")
    binance_api_secret: Optional[str] = Field(None, env="BINANCE_API_SECRET")
    
    # Polygon.io API settings
    polygon_api_key: Optional[str] = Field(None, env="POLYGON_API_KEY")
    
    # Yahoo Finance (no auth required)
    
    # Default symbols to track
    default_stocks: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    default_forex: List[str] = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD"]
    default_crypto: List[str] = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD"]

class NewsDataSettings(BaseSettings):
    """News and sentiment data API settings."""
    # Twitter API settings
    twitter_api_key: Optional[str] = Field(None, env="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(None, env="TWITTER_API_SECRET")
    twitter_access_token: Optional[str] = Field(None, env="TWITTER_ACCESS_TOKEN")
    twitter_access_secret: Optional[str] = Field(None, env="TWITTER_ACCESS_SECRET")
    
    # Reddit API settings
    reddit_client_id: Optional[str] = Field(None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(None, env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field("trading_pipeline/1.0", env="REDDIT_USER_AGENT")
    
    # News API settings
    newsapi_key: Optional[str] = Field(None, env="NEWSAPI_KEY")
    
    # Default news sources
    default_news_sources: List[str] = ["bloomberg", "reuters", "financial-times", "wall-street-journal"]
    
    # Default subreddits to track
    default_subreddits: List[str] = ["wallstreetbets", "investing", "stocks", "cryptocurrency"]

class EconomicDataSettings(BaseSettings):
    """Economic indicators API settings."""
    # FRED API settings
    fred_api_key: Optional[str] = Field(None, env="FRED_API_KEY")
    
    # World Bank API (no auth required)
    
    # Default economic indicators to track
    default_indicators: List[str] = [
        "GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "INDPRO", 
        "HOUST", "PAYEMS", "PCE", "M2", "DFF"
    ]

class Settings(BaseSettings):
    """Main settings class for Data Ingestion Layer."""
    # Environment settings
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    
    # API settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    market_data: MarketDataSettings = MarketDataSettings()
    news_data: NewsDataSettings = NewsDataSettings()
    economic_data: EconomicDataSettings = EconomicDataSettings()
    
    # Data collection settings
    collection_interval: Dict[str, int] = {
        "market_data": 60,  # seconds
        "news_data": 300,   # seconds
        "economic_data": 3600  # seconds
    }
    
    # Data retention settings
    data_retention: Dict[str, int] = {
        "market_data": 90,  # days
        "news_data": 30,    # days
        "economic_data": 365  # days
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()
