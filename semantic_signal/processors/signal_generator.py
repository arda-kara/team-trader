"""
Signal generator for creating trading signals from semantic analysis.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config.settings import settings
from ..models.base import (
    Entity, Event, SentimentAnalysis, CausalRelation, 
    SemanticSignal, SignalType, SignalDirection, 
    SignalTimeframe, SignalStrength, TextSource
)
from ..extractors.entity_extractor import entity_extractor
from ..extractors.event_extractor import event_extractor
from ..sentiment.sentiment_analyzer import sentiment_analyzer
from ..inference.causal_engine import causal_inference_engine
from ..processors.redis_client import signal_cache

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generator for semantic trading signals."""
    
    def __init__(self):
        """Initialize signal generator."""
        pass
    
    def _extract_symbols(self, entities: List[Entity]) -> List[str]:
        """Extract stock symbols from entities.
        
        Args:
            entities: List of entities
            
        Returns:
            List[str]: List of stock symbols
        """
        symbols = []
        
        for entity in entities:
            if entity.type == "COMPANY" and "ticker" in entity.metadata:
                symbols.append(entity.metadata["ticker"])
        
        return list(set(symbols))  # Remove duplicates
    
    def _determine_direction(self, event_type: str, sentiment_score: float, 
                           causal_relations: List[CausalRelation]) -> SignalDirection:
        """Determine signal direction.
        
        Args:
            event_type: Event type
            sentiment_score: Sentiment score
            causal_relations: Causal relations
            
        Returns:
            SignalDirection: Signal direction
        """
        # If we have causal relations, use them
        if causal_relations:
            directions = [r.direction for r in causal_relations]
            positive_count = directions.count("positive")
            negative_count = directions.count("negative")
            neutral_count = directions.count("neutral")
            
            if positive_count > negative_count and positive_count > neutral_count:
                return SignalDirection.BULLISH
            elif negative_count > positive_count and negative_count > neutral_count:
                return SignalDirection.BEARISH
            elif neutral_count > positive_count and neutral_count > negative_count:
                return SignalDirection.NEUTRAL
            else:
                return SignalDirection.MIXED
        
        # Otherwise use sentiment
        if sentiment_score >= 0.5:
            return SignalDirection.BULLISH
        elif sentiment_score <= -0.5:
            return SignalDirection.BEARISH
        else:
            return SignalDirection.NEUTRAL
    
    def _determine_timeframe(self, event_type: str) -> SignalTimeframe:
        """Determine signal timeframe.
        
        Args:
            event_type: Event type
            
        Returns:
            SignalTimeframe: Signal timeframe
        """
        # Map event types to timeframes
        event_timeframes = {
            "EARNINGS_REPORT": SignalTimeframe.SHORT_TERM,
            "PRODUCT_LAUNCH": SignalTimeframe.MEDIUM_TERM,
            "MERGER_ACQUISITION": SignalTimeframe.MEDIUM_TERM,
            "LEADERSHIP_CHANGE": SignalTimeframe.SHORT_TERM,
            "REGULATORY_CHANGE": SignalTimeframe.MEDIUM_TERM,
            "MARKET_MOVEMENT": SignalTimeframe.IMMEDIATE,
            "ECONOMIC_INDICATOR": SignalTimeframe.SHORT_TERM,
            "GEOPOLITICAL_EVENT": SignalTimeframe.SHORT_TERM,
            "NATURAL_DISASTER": SignalTimeframe.IMMEDIATE
        }
        
        return event_timeframes.get(event_type, SignalTimeframe.SHORT_TERM)
    
    def _determine_strength(self, confidence: float, sentiment_score: float) -> SignalStrength:
        """Determine signal strength.
        
        Args:
            confidence: Confidence score
            sentiment_score: Sentiment score (absolute value)
            
        Returns:
            SignalStrength: Signal strength
        """
        # Combine confidence and sentiment magnitude
        sentiment_magnitude = abs(sentiment_score)
        combined_score = (confidence + sentiment_magnitude) / 2
        
        if combined_score >= 0.8:
            return SignalStrength.VERY_STRONG
        elif combined_score >= 0.6:
            return SignalStrength.STRONG
        elif combined_score >= 0.4:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _generate_summary(self, event: Event, sentiment: SentimentAnalysis, 
                        causal_relations: List[CausalRelation], symbols: List[str]) -> str:
        """Generate signal summary.
        
        Args:
            event: Event
            sentiment: Sentiment analysis
            causal_relations: Causal relations
            symbols: Stock symbols
            
        Returns:
            str: Signal summary
        """
        # Extract key information
        event_type = event.type
        event_text = event.text[:100] + "..." if len(event.text) > 100 else event.text
        sentiment_level = sentiment.sentiment
        
        # Generate summary based on event type
        if event_type == "EARNINGS_REPORT":
            company = next((e.text for e in event.entities if e.type == "COMPANY"), "Company")
            performance = event.metadata.get("performance", "reported")
            return f"{company} {performance} earnings with {sentiment_level.value.lower()} sentiment, potentially affecting {', '.join(symbols)}"
        
        elif event_type == "MERGER_ACQUISITION":
            acquirer = event.metadata.get("acquirer", "Company")
            target = event.metadata.get("target", "another company")
            return f"{acquirer} is acquiring {target}, potentially affecting {', '.join(symbols)}"
        
        elif event_type == "PRODUCT_LAUNCH":
            company = event.metadata.get("company", "Company")
            product = event.metadata.get("product", "new product")
            return f"{company} launched {product}, potentially affecting {', '.join(symbols)}"
        
        elif event_type == "LEADERSHIP_CHANGE":
            company = event.metadata.get("company", "Company")
            person = event.metadata.get("person", "executive")
            role = event.metadata.get("role", "position")
            change = event.metadata.get("change", "changed")
            return f"{person} {change} as {role} at {company}, potentially affecting {', '.join(symbols)}"
        
        elif event_type == "REGULATORY_CHANGE":
            entity = event.metadata.get("entity", "Regulator")
            target = event.metadata.get("target", "company")
            action = event.metadata.get("action", "regulatory action")
            return f"{entity} took {action} regarding {target}, potentially affecting {', '.join(symbols)}"
        
        else:
            # Generic summary
            return f"Event detected: {event_text} with {sentiment_level.value.lower()} sentiment, potentially affecting {', '.join(symbols)}"
    
    async def generate_signal_from_event(self, event: Event, sentiment: SentimentAnalysis, 
                                      source_ids: List[str]) -> Optional[SemanticSignal]:
        """Generate signal from event.
        
        Args:
            event: Event
            sentiment: Sentiment analysis
            source_ids: Source IDs
            
        Returns:
            Optional[SemanticSignal]: Generated signal or None
        """
        # Extract symbols
        symbols = self._extract_symbols(event.entities)
        if not symbols:
            return None
        
        # Infer causal relations
        causal_relations = await causal_inference_engine.infer_causal_relations(event, symbols)
        
        # Determine signal attributes
        direction = self._determine_direction(event.type, sentiment.score, causal_relations)
        timeframe = self._determine_timeframe(event.type)
        strength = self._determine_strength(sentiment.confidence, sentiment.score)
        
        # Generate summary
        summary = self._generate_summary(event, sentiment, causal_relations, symbols)
        
        # Create signal
        signal = SemanticSignal(
            id=f"signal_{uuid.uuid4().hex[:8]}",
            type=SignalType.EVENT,
            source_ids=source_ids,
            timestamp=datetime.utcnow(),
            symbols=symbols,
            direction=direction,
            timeframe=timeframe,
            strength=strength,
            confidence=sentiment.confidence,
            summary=summary,
            details={
                "event_type": event.type,
                "sentiment": sentiment.sentiment.value,
                "sentiment_score": sentiment.score,
                "metadata": event.metadata
            }
        )
        
        return signal
    
    async def generate_signal_from_sentiment(self, sentiment: SentimentAnalysis, 
                                          entities: List[Entity], source_ids: List[str]) -> Optional[SemanticSignal]:
        """Generate signal from sentiment.
        
        Args:
            sentiment: Sentiment analysis
            entities: Entities
            source_ids: Source IDs
            
        Returns:
            Optional[SemanticSignal]: Generated signal or None
        """
        # Extract symbols
        symbols = self._extract_symbols(entities)
        if not symbols:
            return None
        
        # Determine signal attributes
        direction = SignalDirection.BULLISH if sentiment.score > 0 else SignalDirection.BEARISH
        if -0.2 <= sentiment.score <= 0.2:
            direction = SignalDirection.NEUTRAL
            
        timeframe = SignalTimeframe.SHORT_TERM
        strength = self._determine_strength(sentiment.confidence, sentiment.score)
        
        # Generate summary
        sentiment_level = sentiment.sentiment.value.lower()
        text_preview = sentiment.text[:100] + "..." if len(sentiment.text) > 100 else sentiment.text
        summary = f"Detected {sentiment_level} sentiment regarding {', '.join(symbols)}: {text_preview}"
        
        # Create signal
        signal = SemanticSignal(
            id=f"signal_{uuid.uuid4().hex[:8]}",
            type=SignalType.SENTIMENT,
            source_ids=source_ids,
            timestamp=datetime.utcnow(),
            symbols=symbols,
            direction=direction,
            timeframe=timeframe,
            strength=strength,
            confidence=sentiment.confidence,
            summary=summary,
            details={
                "sentiment": sentiment.sentiment.value,
                "sentiment_score": sentiment.score,
                "entity_sentiments": sentiment.entity_sentiments
            }
        )
        
        return signal
    
    async def process_text_source(self, source: TextSource) -> List[SemanticSignal]:
        """Process text source to generate signals.
        
        Args:
            source: Text source
            
        Returns:
            List[SemanticSignal]: Generated signals
        """
        signals = []
        
        try:
            # Extract entities
            entities = entity_extractor.extract_entities(source.content)
            
            # Extract events
            events = await event_extractor.extract_events(source.content, entities)
            
            # Analyze sentiment
            sentiment = await sentiment_analyzer.analyze_sentiment(
                source.content, 
                [e.text for e in entities if e.type == "COMPANY"]
            )
            
            # Generate signals from events
            for event in events:
                signal = await self.generate_signal_from_event(
                    event, sentiment, [source.id]
                )
                
                if signal:
                    signals.append(signal)
            
            # If no event signals, generate sentiment signal
            if not signals and entities:
                signal = await self.generate_signal_from_sentiment(
                    sentiment, entities, [source.id]
                )
                
                if signal:
                    signals.append(signal)
            
            return signals
        except Exception as e:
            logger.error(f"Error processing text source: {e}")
            return []
    
    async def process_text_sources(self, sources: List[TextSource]) -> List[SemanticSignal]:
        """Process multiple text sources to generate signals.
        
        Args:
            sources: List of text sources
            
        Returns:
            List[SemanticSignal]: Generated signals
        """
        all_signals = []
        
        for source in sources:
            signals = await self.process_text_source(source)
            all_signals.extend(signals)
        
        return all_signals

# Create signal generator instance
signal_generator = SignalGenerator()
