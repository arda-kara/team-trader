"""
Configuration settings for the Semantic Signal Generator.
"""

import os
from typing import Dict, List, Optional, Union
from pydantic import BaseSettings, Field, validator

class LLMSettings(BaseSettings):
    """LLM integration settings."""
    # OpenAI settings
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-3-opus-20240229", env="ANTHROPIC_MODEL")
    
    # Mistral settings
    mistral_api_key: Optional[str] = Field(None, env="MISTRAL_API_KEY")
    mistral_model: str = Field("mistral-large-latest", env="MISTRAL_MODEL")
    
    # Default model to use
    default_provider: str = Field("openai", env="DEFAULT_LLM_PROVIDER")
    
    # Rate limiting settings
    rate_limit_requests: int = Field(60, env="LLM_RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="LLM_RATE_LIMIT_PERIOD")  # seconds
    
    # Caching settings
    enable_caching: bool = Field(True, env="LLM_ENABLE_CACHING")
    cache_ttl: int = Field(3600, env="LLM_CACHE_TTL")  # seconds

class NLPSettings(BaseSettings):
    """NLP model settings."""
    # spaCy settings
    spacy_model: str = Field("en_core_web_trf", env="SPACY_MODEL")
    
    # Hugging Face settings
    hf_token: Optional[str] = Field(None, env="HF_TOKEN")
    
    # FinBERT settings
    finbert_model: str = Field("ProsusAI/finbert", env="FINBERT_MODEL")
    
    # NER settings
    custom_ner_model: Optional[str] = Field(None, env="CUSTOM_NER_MODEL")
    
    # Embedding settings
    embedding_model: str = Field("sentence-transformers/all-mpnet-base-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(768, env="EMBEDDING_DIMENSION")

class VectorDBSettings(BaseSettings):
    """Vector database settings."""
    # Database type
    db_type: str = Field("qdrant", env="VECTOR_DB_TYPE")
    
    # Qdrant settings
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_collection: str = Field("financial_embeddings", env="QDRANT_COLLECTION")
    
    # Milvus settings
    milvus_host: str = Field("localhost", env="MILVUS_HOST")
    milvus_port: int = Field(19530, env="MILVUS_PORT")
    milvus_collection: str = Field("financial_embeddings", env="MILVUS_COLLECTION")

class EntitySettings(BaseSettings):
    """Entity extraction settings."""
    # Entity types to extract
    entity_types: List[str] = [
        "COMPANY", "PERSON", "PRODUCT", "LOCATION", 
        "FINANCIAL_METRIC", "CURRENCY", "PERCENT", "MONEY"
    ]
    
    # Company name aliases file
    company_aliases_file: str = Field("company_aliases.json", env="COMPANY_ALIASES_FILE")
    
    # Ticker mapping file
    ticker_mapping_file: str = Field("ticker_mapping.json", env="TICKER_MAPPING_FILE")

class EventSettings(BaseSettings):
    """Event extraction settings."""
    # Event types to extract
    event_types: List[str] = [
        "MERGER_ACQUISITION", "EARNINGS_REPORT", "PRODUCT_LAUNCH",
        "LEADERSHIP_CHANGE", "REGULATORY_CHANGE", "MARKET_MOVEMENT",
        "ECONOMIC_INDICATOR", "GEOPOLITICAL_EVENT", "NATURAL_DISASTER"
    ]
    
    # Event classification model
    event_model: str = Field("distilbert-base-uncased", env="EVENT_MODEL")
    
    # Event templates file
    event_templates_file: str = Field("event_templates.json", env="EVENT_TEMPLATES_FILE")

class SentimentSettings(BaseSettings):
    """Sentiment analysis settings."""
    # Sentiment levels
    sentiment_levels: List[str] = [
        "VERY_NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "VERY_POSITIVE"
    ]
    
    # Sentiment thresholds
    sentiment_thresholds: Dict[str, float] = {
        "VERY_NEGATIVE": -0.6,
        "NEGATIVE": -0.2,
        "NEUTRAL": 0.2,
        "POSITIVE": 0.6,
        "VERY_POSITIVE": 1.0
    }
    
    # Sentiment models
    general_sentiment_model: str = Field("ProsusAI/finbert", env="GENERAL_SENTIMENT_MODEL")
    financial_sentiment_model: str = Field("ProsusAI/finbert", env="FINANCIAL_SENTIMENT_MODEL")
    
    # Sentiment aggregation method
    aggregation_method: str = Field("weighted_average", env="SENTIMENT_AGGREGATION_METHOD")

class CausalSettings(BaseSettings):
    """Causal inference settings."""
    # Causal model type
    model_type: str = Field("llm", env="CAUSAL_MODEL_TYPE")  # llm, statistical, or hybrid
    
    # Causal templates file
    causal_templates_file: str = Field("causal_templates.json", env="CAUSAL_TEMPLATES_FILE")
    
    # Confidence threshold
    confidence_threshold: float = Field(0.7, env="CAUSAL_CONFIDENCE_THRESHOLD")
    
    # Historical correlation window (days)
    correlation_window: int = Field(90, env="CORRELATION_WINDOW")

class APISettings(BaseSettings):
    """API settings."""
    # API host and port
    host: str = Field("0.0.0.0", env="SEMANTIC_API_HOST")
    port: int = Field(8001, env="SEMANTIC_API_PORT")
    
    # API rate limiting
    rate_limit_requests: int = Field(100, env="API_RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="API_RATE_LIMIT_PERIOD")  # seconds

class Settings(BaseSettings):
    """Main settings class for Semantic Signal Generator."""
    # Environment settings
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    
    # Component settings
    llm: LLMSettings = LLMSettings()
    nlp: NLPSettings = NLPSettings()
    vector_db: VectorDBSettings = VectorDBSettings()
    entity: EntitySettings = EntitySettings()
    event: EventSettings = EventSettings()
    sentiment: SentimentSettings = SentimentSettings()
    causal: CausalSettings = CausalSettings()
    api: APISettings = APISettings()
    
    # Redis settings (for queue and cache)
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(1, env="REDIS_DB")  # Use different DB than data ingestion
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    # Processing settings
    batch_size: int = Field(10, env="PROCESSING_BATCH_SIZE")
    processing_interval: int = Field(60, env="PROCESSING_INTERVAL")  # seconds
    max_text_length: int = Field(4096, env="MAX_TEXT_LENGTH")
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()
