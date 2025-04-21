"""
API endpoints for the semantic signal generator.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..models.base import (
    TextSource, Entity, Event, SentimentAnalysis, CausalRelation, SemanticSignal,
    EntityExtractionRequest, EntityExtractionResponse,
    EventExtractionRequest, EventExtractionResponse,
    SentimentAnalysisRequest, SentimentAnalysisResponse,
    CausalInferenceRequest, CausalInferenceResponse,
    SignalRequest, SignalResponse, SignalQueryRequest
)
from ..extractors.entity_extractor import entity_extractor
from ..extractors.event_extractor import event_extractor
from ..sentiment.sentiment_analyzer import sentiment_analyzer
from ..inference.causal_engine import causal_inference_engine
from ..processors.signal_generator import signal_generator
from ..processors.redis_client import signal_cache

router = APIRouter(prefix="/api/semantic", tags=["semantic"])

logger = logging.getLogger(__name__)

# Entity extraction endpoint
@router.post("/entities", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """Extract entities from text."""
    try:
        entities = entity_extractor.extract_entities(
            request.text,
            [et.value for et in request.entity_types] if request.entity_types else None
        )
        
        return EntityExtractionResponse(entities=entities)
    except Exception as e:
        logger.exception("Error extracting entities")
        raise HTTPException(status_code=500, detail=str(e))

# Event extraction endpoint
@router.post("/events", response_model=EventExtractionResponse)
async def extract_events(request: EventExtractionRequest):
    """Extract events from text."""
    try:
        events = await event_extractor.extract_events(
            request.text,
            None,
            [et.value for et in request.event_types] if request.event_types else None
        )
        
        return EventExtractionResponse(events=events)
    except Exception as e:
        logger.exception("Error extracting events")
        raise HTTPException(status_code=500, detail=str(e))

# Sentiment analysis endpoint
@router.post("/sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment in text."""
    try:
        sentiment = await sentiment_analyzer.analyze_sentiment(
            request.text,
            request.entities
        )
        
        return SentimentAnalysisResponse(sentiment=sentiment)
    except Exception as e:
        logger.exception("Error analyzing sentiment")
        raise HTTPException(status_code=500, detail=str(e))

# Causal inference endpoint
@router.post("/causal", response_model=CausalInferenceResponse)
async def infer_causal_relations(request: CausalInferenceRequest):
    """Infer causal relations between events and target entities."""
    try:
        # First extract events
        events = await event_extractor.extract_events(request.text)
        
        if not events:
            return CausalInferenceResponse(relations=[])
        
        # Use first event for causal inference
        relations = await causal_inference_engine.infer_causal_relations(
            events[0],
            request.target_entities
        )
        
        return CausalInferenceResponse(relations=relations)
    except Exception as e:
        logger.exception("Error inferring causal relations")
        raise HTTPException(status_code=500, detail=str(e))

# Signal generation endpoint
@router.post("/signals", response_model=SignalResponse)
async def generate_signals(request: SignalRequest):
    """Generate signals from text sources."""
    try:
        signals = await signal_generator.process_text_sources(request.text_sources)
        
        return SignalResponse(signals=signals)
    except Exception as e:
        logger.exception("Error generating signals")
        raise HTTPException(status_code=500, detail=str(e))

# Signal query endpoint
@router.post("/signals/query", response_model=SignalResponse)
async def query_signals(request: SignalQueryRequest):
    """Query signals based on criteria."""
    try:
        # In a real implementation, this would query a database
        # For now, we'll return cached signals filtered by the request criteria
        
        # Get all cached signals (in a real implementation, this would be a database query)
        all_signals = []
        for i in range(10):  # Simulate having some cached signals
            cached_signal = signal_cache.get(f"signal:{i}")
            if cached_signal:
                all_signals.append(SemanticSignal(**cached_signal))
        
        # Filter signals based on request criteria
        filtered_signals = all_signals
        
        if request.symbols:
            filtered_signals = [
                s for s in filtered_signals 
                if any(symbol in s.symbols for symbol in request.symbols)
            ]
        
        if request.signal_types:
            filtered_signals = [
                s for s in filtered_signals 
                if s.type in request.signal_types
            ]
        
        if request.directions:
            filtered_signals = [
                s for s in filtered_signals 
                if s.direction in request.directions
            ]
        
        if request.timeframes:
            filtered_signals = [
                s for s in filtered_signals 
                if s.timeframe in request.timeframes
            ]
        
        if request.min_confidence is not None:
            filtered_signals = [
                s for s in filtered_signals 
                if s.confidence >= request.min_confidence
            ]
        
        if request.start_time:
            filtered_signals = [
                s for s in filtered_signals 
                if s.timestamp >= request.start_time
            ]
        
        if request.end_time:
            filtered_signals = [
                s for s in filtered_signals 
                if s.timestamp <= request.end_time
            ]
        
        # Apply limit
        limit = request.limit or 100
        filtered_signals = filtered_signals[:limit]
        
        return SignalResponse(signals=filtered_signals)
    except Exception as e:
        logger.exception("Error querying signals")
        raise HTTPException(status_code=500, detail=str(e))
