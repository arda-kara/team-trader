"""
API endpoints for the data ingestion layer.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..models.base import (
    MarketDataRequest, MarketDataResponse,
    NewsDataRequest, NewsDataResponse,
    EconomicDataRequest, EconomicDataResponse,
    TimeRange
)
from ..storage.database import get_db_session
from ..storage.redis_client import data_cache

router = APIRouter(prefix="/api/data", tags=["data"])

logger = logging.getLogger(__name__)

# Market Data Endpoints
@router.post("/market", response_model=MarketDataResponse)
async def get_market_data(request: MarketDataRequest, db: Session = Depends(get_db_session)):
    """Get market data based on request parameters."""
    try:
        from ..storage.models import Asset, OHLCV
        from sqlalchemy import and_
        
        # Build query
        query = db.query(OHLCV).join(Asset)
        
        # Filter by symbols
        query = query.filter(Asset.symbol.in_(request.symbols))
        
        # Filter by asset types if provided
        if request.asset_types:
            query = query.filter(Asset.asset_type.in_([t.value for t in request.asset_types]))
        
        # Filter by timeframe
        query = query.filter(OHLCV.timeframe == request.timeframe.value)
        
        # Filter by time range
        query = query.filter(OHLCV.timestamp >= request.time_range.start)
        if request.time_range.end:
            query = query.filter(OHLCV.timestamp <= request.time_range.end)
        
        # Filter by source if provided
        if request.source:
            query = query.filter(OHLCV.source == request.source.value)
        
        # Order by timestamp
        query = query.order_by(OHLCV.timestamp.desc())
        
        # Execute query
        results = query.all()
        
        # Convert to response model
        from ..models.base import OHLCV as OHLCVModel, AssetType, DataSource, TimeFrame
        
        data = []
        for result in results:
            asset = result.asset
            data.append(OHLCVModel(
                symbol=asset.symbol,
                asset_type=AssetType(asset.asset_type.value),
                source=DataSource(result.source.value),
                timestamp=result.timestamp,
                timeframe=TimeFrame(result.timeframe.value),
                open=result.open,
                high=result.high,
                low=result.low,
                close=result.close,
                volume=result.volume
            ))
        
        return MarketDataResponse(
            success=True,
            count=len(data),
            data=data
        )
    except Exception as e:
        logger.exception("Error retrieving market data")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/symbols", response_model=List[str])
async def get_market_symbols(db: Session = Depends(get_db_session)):
    """Get available market symbols."""
    try:
        from ..storage.models import Asset
        
        # Query distinct symbols
        results = db.query(Asset.symbol).distinct().all()
        
        # Extract symbols
        symbols = [result[0] for result in results]
        
        return symbols
    except Exception as e:
        logger.exception("Error retrieving market symbols")
        raise HTTPException(status_code=500, detail=str(e))

# News Data Endpoints
@router.post("/news", response_model=NewsDataResponse)
async def get_news_data(request: NewsDataRequest, db: Session = Depends(get_db_session)):
    """Get news data based on request parameters."""
    try:
        from ..storage.models import NewsItem
        from sqlalchemy import or_, func
        
        # Build query
        query = db.query(NewsItem)
        
        # Filter by time range
        query = query.filter(NewsItem.timestamp >= request.time_range.start)
        if request.time_range.end:
            query = query.filter(NewsItem.timestamp <= request.time_range.end)
        
        # Filter by sources if provided
        if request.sources:
            query = query.filter(NewsItem.source.in_([s.value for s in request.sources]))
        
        # Filter by symbols if provided
        if request.symbols:
            # This assumes symbols are stored as a JSON array
            # The implementation depends on the database used
            for symbol in request.symbols:
                # For PostgreSQL, use the @> operator for JSON containment
                query = query.filter(NewsItem.symbols.contains([symbol]))
        
        # Filter by keywords if provided
        if request.keywords:
            keyword_filters = []
            for keyword in request.keywords:
                keyword_filters.append(NewsItem.title.ilike(f"%{keyword}%"))
                keyword_filters.append(NewsItem.content.ilike(f"%{keyword}%"))
            query = query.filter(or_(*keyword_filters))
        
        # Order by timestamp
        query = query.order_by(NewsItem.timestamp.desc())
        
        # Execute query
        results = query.all()
        
        # Convert to response model
        from ..models.base import NewsItem as NewsItemModel, NewsSource, SentimentLevel
        
        data = []
        for result in results:
            data.append(NewsItemModel(
                title=result.title,
                content=result.content,
                source=NewsSource(result.source.value) if result.source else NewsSource.CUSTOM,
                url=result.url,
                timestamp=result.timestamp,
                author=result.author,
                symbols=result.symbols or [],
                sentiment=SentimentLevel(result.sentiment.value) if result.sentiment else None,
                sentiment_score=result.sentiment_score
            ))
        
        return NewsDataResponse(
            success=True,
            count=len(data),
            data=data
        )
    except Exception as e:
        logger.exception("Error retrieving news data")
        raise HTTPException(status_code=500, detail=str(e))

# Economic Data Endpoints
@router.post("/economic", response_model=EconomicDataResponse)
async def get_economic_data(request: EconomicDataRequest, db: Session = Depends(get_db_session)):
    """Get economic data based on request parameters."""
    try:
        from ..storage.models import EconomicIndicator
        
        # Build query
        query = db.query(EconomicIndicator)
        
        # Filter by indicators
        query = query.filter(EconomicIndicator.indicator_id.in_(request.indicators))
        
        # Filter by time range
        query = query.filter(EconomicIndicator.timestamp >= request.time_range.start)
        if request.time_range.end:
            query = query.filter(EconomicIndicator.timestamp <= request.time_range.end)
        
        # Filter by region if provided
        if request.region:
            query = query.filter(EconomicIndicator.region == request.region)
        
        # Filter by frequency if provided
        if request.frequency:
            query = query.filter(EconomicIndicator.frequency == request.frequency)
        
        # Order by timestamp
        query = query.order_by(EconomicIndicator.timestamp.desc())
        
        # Execute query
        results = query.all()
        
        # Convert to response model
        from ..models.base import EconomicIndicator as EconomicIndicatorModel
        
        data = []
        for result in results:
            data.append(EconomicIndicatorModel(
                indicator_id=result.indicator_id,
                name=result.name,
                value=result.value,
                timestamp=result.timestamp,
                source=result.source,
                frequency=result.frequency,
                unit=result.unit,
                region=result.region
            ))
        
        return EconomicDataResponse(
            success=True,
            count=len(data),
            data=data
        )
    except Exception as e:
        logger.exception("Error retrieving economic data")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/economic/indicators", response_model=List[str])
async def get_economic_indicators(db: Session = Depends(get_db_session)):
    """Get available economic indicators."""
    try:
        from ..storage.models import EconomicIndicator
        
        # Query distinct indicator IDs
        results = db.query(EconomicIndicator.indicator_id).distinct().all()
        
        # Extract indicator IDs
        indicators = [result[0] for result in results]
        
        return indicators
    except Exception as e:
        logger.exception("Error retrieving economic indicators")
        raise HTTPException(status_code=500, detail=str(e))
