"""
Sentiment analysis processor for financial texts.
"""

import logging
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from ..config.settings import settings
from ..models.base import Entity, SentimentAnalysis, SentimentLevel
from ..extractors.entity_extractor import entity_extractor
from ..processors.redis_client import sentiment_cache

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Sentiment analyzer for financial texts."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.sentiment_levels = settings.sentiment.sentiment_levels
        self.sentiment_thresholds = settings.sentiment.sentiment_thresholds
        self.general_model_name = settings.sentiment.general_sentiment_model
        self.financial_model_name = settings.sentiment.financial_sentiment_model
        self.aggregation_method = settings.sentiment.aggregation_method
        
        # Load models
        self.general_model, self.general_tokenizer = self._load_model(self.general_model_name)
        
        # Load financial model if different
        if self.financial_model_name != self.general_model_name:
            self.financial_model, self.financial_tokenizer = self._load_model(self.financial_model_name)
        else:
            self.financial_model, self.financial_tokenizer = self.general_model, self.general_tokenizer
    
    def _load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load sentiment model and tokenizer.
        
        Args:
            model_name: Model name or path
            
        Returns:
            Tuple[Any, Any]: Model and tokenizer
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading sentiment model {model_name}: {e}")
            # Fallback to default model
            fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
            logger.info(f"Loading fallback sentiment model: {fallback_model}")
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
            return model, tokenizer
    
    def _predict_sentiment(self, text: str, model: Any, tokenizer: Any) -> Tuple[float, float]:
        """Predict sentiment score using model.
        
        Args:
            text: Input text
            model: Sentiment model
            tokenizer: Tokenizer
            
        Returns:
            Tuple[float, float]: Sentiment score (-1 to 1) and confidence
        """
        try:
            # Truncate text if too long
            max_length = tokenizer.model_max_length
            if len(text) > max_length:
                text = text[:max_length]
            
            # Tokenize and predict
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
            
            # Convert to score and confidence
            if len(probs) == 2:  # Binary classification (positive/negative)
                score = 2 * probs[1] - 1  # Convert [0,1] to [-1,1]
                confidence = max(probs)
            elif len(probs) == 3:  # Ternary classification (negative/neutral/positive)
                # Convert to score in [-1,1]
                score = probs[2] - probs[0]  # positive - negative
                confidence = max(probs)
            else:
                # Handle other cases
                score = 0.0
                confidence = 0.0
            
            return score, confidence
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            return 0.0, 0.0
    
    def _map_score_to_sentiment(self, score: float) -> SentimentLevel:
        """Map sentiment score to sentiment level.
        
        Args:
            score: Sentiment score (-1 to 1)
            
        Returns:
            SentimentLevel: Sentiment level
        """
        if score <= self.sentiment_thresholds["VERY_NEGATIVE"]:
            return SentimentLevel.VERY_NEGATIVE
        elif score <= self.sentiment_thresholds["NEGATIVE"]:
            return SentimentLevel.NEGATIVE
        elif score <= self.sentiment_thresholds["NEUTRAL"]:
            return SentimentLevel.NEUTRAL
        elif score <= self.sentiment_thresholds["POSITIVE"]:
            return SentimentLevel.POSITIVE
        else:
            return SentimentLevel.VERY_POSITIVE
    
    def _extract_entity_context(self, text: str, entity_text: str, window_size: int = 100) -> str:
        """Extract context around entity mention.
        
        Args:
            text: Full text
            entity_text: Entity text
            window_size: Context window size (characters)
            
        Returns:
            str: Context text
        """
        text_lower = text.lower()
        entity_lower = entity_text.lower()
        
        # Find entity position
        pos = text_lower.find(entity_lower)
        if pos == -1:
            return ""
        
        # Extract context
        start = max(0, pos - window_size)
        end = min(len(text), pos + len(entity_text) + window_size)
        
        return text[start:end]
    
    async def analyze_sentiment(self, text: str, entities: Optional[List[str]] = None) -> SentimentAnalysis:
        """Analyze sentiment in text.
        
        Args:
            text: Input text
            entities: Optional list of entities to analyze
            
        Returns:
            SentimentAnalysis: Sentiment analysis result
        """
        # Check cache
        cache_key = f"sentiment:{hash(text)}"
        if entities:
            cache_key += f":{hash(tuple(entities))}"
            
        cached_result = sentiment_cache.get(cache_key)
        if cached_result:
            return SentimentAnalysis(**cached_result)
        
        # Analyze overall sentiment
        score, confidence = self._predict_sentiment(text, self.general_model, self.general_tokenizer)
        sentiment = self._map_score_to_sentiment(score)
        
        # Analyze entity-specific sentiment if entities provided
        entity_sentiments = None
        if entities:
            entity_sentiments = {}
            
            for entity in entities:
                # Extract context around entity
                context = self._extract_entity_context(text, entity)
                if not context:
                    continue
                
                # Analyze sentiment in context
                entity_score, entity_confidence = self._predict_sentiment(
                    context, self.financial_model, self.financial_tokenizer
                )
                
                entity_sentiments[entity] = {
                    "sentiment": entity_score,
                    "confidence": entity_confidence
                }
        
        # Create result
        result = SentimentAnalysis(
            text=text,
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            entity_sentiments=entity_sentiments
        )
        
        # Cache result
        sentiment_cache.set(cache_key, result.dict(), ttl=3600)  # 1 hour
        
        return result

# Create analyzer instance
sentiment_analyzer = SentimentAnalyzer()
