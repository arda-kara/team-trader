"""
Base models for the semantic signal generator.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator

# Enums
class EntityType(str, Enum):
    """Entity type enumeration."""
    COMPANY = "COMPANY"
    PERSON = "PERSON"
    PRODUCT = "PRODUCT"
    LOCATION = "LOCATION"
    FINANCIAL_METRIC = "FINANCIAL_METRIC"
    CURRENCY = "CURRENCY"
    PERCENT = "PERCENT"
    MONEY = "MONEY"

class EventType(str, Enum):
    """Event type enumeration."""
    MERGER_ACQUISITION = "MERGER_ACQUISITION"
    EARNINGS_REPORT = "EARNINGS_REPORT"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    LEADERSHIP_CHANGE = "LEADERSHIP_CHANGE"
    REGULATORY_CHANGE = "REGULATORY_CHANGE"
    MARKET_MOVEMENT = "MARKET_MOVEMENT"
    ECONOMIC_INDICATOR = "ECONOMIC_INDICATOR"
    GEOPOLITICAL_EVENT = "GEOPOLITICAL_EVENT"
    NATURAL_DISASTER = "NATURAL_DISASTER"

class SentimentLevel(str, Enum):
    """Sentiment level enumeration."""
    VERY_NEGATIVE = "VERY_NEGATIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    VERY_POSITIVE = "VERY_POSITIVE"

class ConfidenceLevel(str, Enum):
    """Confidence level enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class SignalType(str, Enum):
    """Signal type enumeration."""
    ENTITY = "ENTITY"
    EVENT = "EVENT"
    SENTIMENT = "SENTIMENT"
    CAUSAL = "CAUSAL"
    COMPOSITE = "COMPOSITE"

class SignalDirection(str, Enum):
    """Signal direction enumeration."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"

class SignalTimeframe(str, Enum):
    """Signal timeframe enumeration."""
    IMMEDIATE = "IMMEDIATE"  # Hours
    SHORT_TERM = "SHORT_TERM"  # Days
    MEDIUM_TERM = "MEDIUM_TERM"  # Weeks
    LONG_TERM = "LONG_TERM"  # Months

class SignalStrength(str, Enum):
    """Signal strength enumeration."""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

# Base Models
class TextSource(BaseModel):
    """Text source model."""
    id: str
    type: str  # news, social_media, sec_filing, etc.
    url: Optional[str] = None
    title: Optional[str] = None
    content: str
    timestamp: datetime
    source_name: Optional[str] = None
    author: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "news_123",
                "type": "news",
                "url": "https://example.com/article",
                "title": "Company XYZ Reports Strong Earnings",
                "content": "Company XYZ reported earnings that exceeded analyst expectations...",
                "timestamp": "2023-01-01T12:00:00Z",
                "source_name": "Financial Times",
                "author": "John Doe"
            }
        }

class Entity(BaseModel):
    """Entity model."""
    text: str
    type: EntityType
    start_char: int
    end_char: int
    normalized_text: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Apple",
                "type": "COMPANY",
                "start_char": 10,
                "end_char": 15,
                "normalized_text": "Apple Inc.",
                "metadata": {"ticker": "AAPL", "exchange": "NASDAQ"}
            }
        }

class Event(BaseModel):
    """Event model."""
    type: EventType
    text: str
    entities: List[Entity] = []
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "type": "EARNINGS_REPORT",
                "text": "Apple reported quarterly earnings that beat analyst expectations",
                "entities": [
                    {
                        "text": "Apple",
                        "type": "COMPANY",
                        "start_char": 0,
                        "end_char": 5,
                        "normalized_text": "Apple Inc.",
                        "metadata": {"ticker": "AAPL"}
                    }
                ],
                "timestamp": "2023-01-01T16:30:00Z",
                "metadata": {"beat_expectations": True, "revenue_growth": 15.3}
            }
        }

class SentimentAnalysis(BaseModel):
    """Sentiment analysis model."""
    text: str
    sentiment: SentimentLevel
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    entity_sentiments: Optional[Dict[str, Dict[str, float]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Apple reported strong earnings, but Microsoft faced challenges",
                "sentiment": "POSITIVE",
                "score": 0.65,
                "confidence": 0.85,
                "entity_sentiments": {
                    "Apple": {"sentiment": 0.8, "confidence": 0.9},
                    "Microsoft": {"sentiment": -0.3, "confidence": 0.7}
                }
            }
        }

class CausalRelation(BaseModel):
    """Causal relation model."""
    cause: str
    effect: str
    confidence: float  # 0.0 to 1.0
    direction: str  # positive, negative, unknown
    explanation: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "cause": "Fed interest rate hike",
                "effect": "AAPL stock price",
                "confidence": 0.75,
                "direction": "negative",
                "explanation": "Higher interest rates typically reduce growth stock valuations"
            }
        }

class SemanticSignal(BaseModel):
    """Semantic signal model."""
    id: str
    type: SignalType
    source_ids: List[str]
    timestamp: datetime
    symbols: List[str]
    direction: SignalDirection
    timeframe: SignalTimeframe
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    summary: str
    details: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "id": "signal_123",
                "type": "EVENT",
                "source_ids": ["news_123", "news_124"],
                "timestamp": "2023-01-01T12:30:00Z",
                "symbols": ["AAPL"],
                "direction": "BULLISH",
                "timeframe": "SHORT_TERM",
                "strength": "STRONG",
                "confidence": 0.85,
                "summary": "Apple's earnings beat expectations with strong iPhone sales",
                "details": {
                    "event_type": "EARNINGS_REPORT",
                    "beat_expectations": True,
                    "revenue_growth": 15.3,
                    "key_metrics": {"iPhone_sales": "better than expected"}
                }
            }
        }

# Request/Response Models
class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Apple reported quarterly earnings that beat analyst expectations with iPhone sales growing 10%",
                "metadata": {"source": "news", "timestamp": "2023-01-01T12:00:00Z"}
            }
        }

class EntityExtractionRequest(BaseModel):
    """Request model for entity extraction."""
    text: str
    entity_types: Optional[List[EntityType]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Apple and Microsoft reported strong earnings, with Apple's revenue reaching $90 billion",
                "entity_types": ["COMPANY", "FINANCIAL_METRIC", "MONEY"]
            }
        }

class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction."""
    entities: List[Entity]
    
    class Config:
        schema_extra = {
            "example": {
                "entities": [
                    {
                        "text": "Apple",
                        "type": "COMPANY",
                        "start_char": 0,
                        "end_char": 5,
                        "normalized_text": "Apple Inc.",
                        "metadata": {"ticker": "AAPL"}
                    },
                    {
                        "text": "Microsoft",
                        "type": "COMPANY",
                        "start_char": 10,
                        "end_char": 19,
                        "normalized_text": "Microsoft Corporation",
                        "metadata": {"ticker": "MSFT"}
                    },
                    {
                        "text": "$90 billion",
                        "type": "MONEY",
                        "start_char": 73,
                        "end_char": 84,
                        "normalized_text": "90000000000",
                        "metadata": {"currency": "USD"}
                    }
                ]
            }
        }

class EventExtractionRequest(BaseModel):
    """Request model for event extraction."""
    text: str
    event_types: Optional[List[EventType]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Apple announced it will acquire AI startup Acme for $500 million to enhance its machine learning capabilities",
                "event_types": ["MERGER_ACQUISITION", "PRODUCT_LAUNCH"]
            }
        }

class EventExtractionResponse(BaseModel):
    """Response model for event extraction."""
    events: List[Event]
    
    class Config:
        schema_extra = {
            "example": {
                "events": [
                    {
                        "type": "MERGER_ACQUISITION",
                        "text": "Apple announced it will acquire AI startup Acme for $500 million",
                        "entities": [
                            {
                                "text": "Apple",
                                "type": "COMPANY",
                                "start_char": 0,
                                "end_char": 5,
                                "normalized_text": "Apple Inc.",
                                "metadata": {"ticker": "AAPL"}
                            },
                            {
                                "text": "Acme",
                                "type": "COMPANY",
                                "start_char": 37,
                                "end_char": 41,
                                "normalized_text": "Acme AI",
                                "metadata": {"is_private": True}
                            },
                            {
                                "text": "$500 million",
                                "type": "MONEY",
                                "start_char": 46,
                                "end_char": 58,
                                "normalized_text": "500000000",
                                "metadata": {"currency": "USD"}
                            }
                        ],
                        "metadata": {"acquirer": "Apple", "target": "Acme", "amount": 500000000}
                    }
                ]
            }
        }

class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str
    entities: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Apple reported strong earnings, but Microsoft faced challenges in its cloud division",
                "entities": ["Apple", "Microsoft"]
            }
        }

class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: SentimentAnalysis
    
    class Config:
        schema_extra = {
            "example": {
                "sentiment": {
                    "text": "Apple reported strong earnings, but Microsoft faced challenges in its cloud division",
                    "sentiment": "MIXED",
                    "score": 0.2,
                    "confidence": 0.85,
                    "entity_sentiments": {
                        "Apple": {"sentiment": 0.8, "confidence": 0.9},
                        "Microsoft": {"sentiment": -0.4, "confidence": 0.8}
                    }
                }
            }
        }

class CausalInferenceRequest(BaseModel):
    """Request model for causal inference."""
    text: str
    target_entities: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "text": "The Federal Reserve announced a 50 basis point interest rate hike, causing tech stocks to decline",
                "target_entities": ["AAPL", "MSFT", "GOOGL"]
            }
        }

class CausalInferenceResponse(BaseModel):
    """Response model for causal inference."""
    relations: List[CausalRelation]
    
    class Config:
        schema_extra = {
            "example": {
                "relations": [
                    {
                        "cause": "Federal Reserve interest rate hike",
                        "effect": "AAPL stock price",
                        "confidence": 0.75,
                        "direction": "negative",
                        "explanation": "Higher interest rates typically reduce growth stock valuations"
                    },
                    {
                        "cause": "Federal Reserve interest rate hike",
                        "effect": "MSFT stock price",
                        "confidence": 0.72,
                        "direction": "negative",
                        "explanation": "Tech stocks are sensitive to interest rate changes"
                    },
                    {
                        "cause": "Federal Reserve interest rate hike",
                        "effect": "GOOGL stock price",
                        "confidence": 0.70,
                        "direction": "negative",
                        "explanation": "Growth stocks tend to decline with higher interest rates"
                    }
                ]
            }
        }

class SignalRequest(BaseModel):
    """Request model for signal generation."""
    text_sources: List[TextSource]
    
    class Config:
        schema_extra = {
            "example": {
                "text_sources": [
                    {
                        "id": "news_123",
                        "type": "news",
                        "url": "https://example.com/article",
                        "title": "Apple Reports Strong Earnings",
                        "content": "Apple reported quarterly earnings that beat analyst expectations...",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "source_name": "Financial Times",
                        "author": "John Doe"
                    }
                ]
            }
        }

class SignalResponse(BaseModel):
    """Response model for signal generation."""
    signals: List[SemanticSignal]
    
    class Config:
        schema_extra = {
            "example": {
                "signals": [
                    {
                        "id": "signal_123",
                        "type": "EVENT",
                        "source_ids": ["news_123"],
                        "timestamp": "2023-01-01T12:30:00Z",
                        "symbols": ["AAPL"],
                        "direction": "BULLISH",
                        "timeframe": "SHORT_TERM",
                        "strength": "STRONG",
                        "confidence": 0.85,
                        "summary": "Apple's earnings beat expectations with strong iPhone sales",
                        "details": {
                            "event_type": "EARNINGS_REPORT",
                            "beat_expectations": True,
                            "revenue_growth": 15.3,
                            "key_metrics": {"iPhone_sales": "better than expected"}
                        }
                    }
                ]
            }
        }

class SignalQueryRequest(BaseModel):
    """Request model for signal querying."""
    symbols: Optional[List[str]] = None
    signal_types: Optional[List[SignalType]] = None
    directions: Optional[List[SignalDirection]] = None
    timeframes: Optional[List[SignalTimeframe]] = None
    min_confidence: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: Optional[int] = 100
    
    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT"],
                "signal_types": ["EVENT", "SENTIMENT"],
                "directions": ["BULLISH"],
                "timeframes": ["SHORT_TERM", "MEDIUM_TERM"],
                "min_confidence": 0.7,
                "start_time": "2023-01-01T00:00:00Z",
                "end_time": "2023-01-31T23:59:59Z",
                "limit": 50
            }
        }
