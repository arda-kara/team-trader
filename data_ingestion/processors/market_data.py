"""
Market data processor for normalizing and storing OHLCV data.
"""

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.exc import IntegrityError

from ..models.base import OHLCV, AssetType, DataSource, TimeFrame
from ..storage.database import get_db_session
from ..storage.models import Asset, OHLCV as OHLCVModel
from ..storage.redis_client import market_data_queue, data_cache
from .base import BaseProcessor, ProcessorRegistry

logger = logging.getLogger(__name__)

class MarketDataProcessor(BaseProcessor):
    """Processor for market data."""
    
    def __init__(self):
        """Initialize market data processor."""
        super().__init__("market_data", market_data_queue, None)
    
    async def _ensure_asset_exists(self, symbol: str, asset_type: AssetType) -> int:
        """Ensure asset exists in database.
        
        Args:
            symbol: Asset symbol
            asset_type: Asset type
            
        Returns:
            int: Asset ID
        """
        with get_db_session() as session:
            # Check if asset exists
            asset = session.query(Asset).filter(
                Asset.symbol == symbol,
                Asset.asset_type == asset_type.value
            ).first()
            
            if asset:
                return asset.id
                
            # Create new asset
            now = datetime.datetime.utcnow()
            new_asset = Asset(
                symbol=symbol,
                name=symbol,  # Default name to symbol
                asset_type=asset_type.value,
                is_active=1,
                created_at=now,
                updated_at=now
            )
            
            session.add(new_asset)
            session.flush()
            
            return new_asset.id
    
    async def _store_ohlcv(self, ohlcv_data: List[Dict]) -> int:
        """Store OHLCV data in database.
        
        Args:
            ohlcv_data: List of OHLCV dictionaries
            
        Returns:
            int: Number of records stored
        """
        stored_count = 0
        
        for data in ohlcv_data:
            try:
                # Parse data
                ohlcv = OHLCV(**data)
                
                # Ensure asset exists
                asset_id = await self._ensure_asset_exists(ohlcv.symbol, ohlcv.asset_type)
                
                # Create database model
                db_model = OHLCVModel(
                    asset_id=asset_id,
                    timestamp=ohlcv.timestamp,
                    timeframe=ohlcv.timeframe.value,
                    open=ohlcv.open,
                    high=ohlcv.high,
                    low=ohlcv.low,
                    close=ohlcv.close,
                    volume=ohlcv.volume,
                    source=ohlcv.source.value
                )
                
                # Store in database
                with get_db_session() as session:
                    # Check if record already exists
                    existing = session.query(OHLCVModel).filter(
                        OHLCVModel.asset_id == asset_id,
                        OHLCVModel.timestamp == ohlcv.timestamp,
                        OHLCVModel.timeframe == ohlcv.timeframe.value
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.open = ohlcv.open
                        existing.high = ohlcv.high
                        existing.low = ohlcv.low
                        existing.close = ohlcv.close
                        existing.volume = ohlcv.volume
                        existing.source = ohlcv.source.value
                    else:
                        # Add new record
                        session.add(db_model)
                        
                    stored_count += 1
            except Exception as e:
                logger.error(f"Error storing OHLCV data: {e}")
        
        return stored_count
    
    async def _update_cache(self, ohlcv_data: List[Dict]) -> int:
        """Update cache with latest market data.
        
        Args:
            ohlcv_data: List of OHLCV dictionaries
            
        Returns:
            int: Number of records cached
        """
        cached_count = 0
        
        # Group by symbol and timeframe
        grouped_data = {}
        for data in ohlcv_data:
            symbol = data.get("symbol")
            timeframe = data.get("timeframe")
            
            if not symbol or not timeframe:
                continue
                
            key = f"{symbol}:{timeframe}"
            if key not in grouped_data:
                grouped_data[key] = []
                
            grouped_data[key].append(data)
        
        # Update cache for each symbol/timeframe
        for key, items in grouped_data.items():
            try:
                # Sort by timestamp (newest first)
                sorted_items = sorted(
                    items, 
                    key=lambda x: datetime.datetime.fromisoformat(x["timestamp"]) 
                    if isinstance(x["timestamp"], str) 
                    else x["timestamp"],
                    reverse=True
                )
                
                # Keep only the latest 100 bars
                latest_bars = sorted_items[:100]
                
                # Cache with 1 hour TTL
                cache_key = f"market_data:{key}"
                data_cache.set(cache_key, latest_bars, ttl=3600)
                
                cached_count += len(latest_bars)
            except Exception as e:
                logger.error(f"Error updating cache for {key}: {e}")
        
        return cached_count
    
    async def process(self, data: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Process market data.
        
        Args:
            data: Market data to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if not isinstance(data, list):
            data = [data]
            
        try:
            # Store in database
            stored_count = await self._store_ohlcv(data)
            
            # Update cache
            cached_count = await self._update_cache(data)
            
            return {
                "success": True,
                "records_processed": len(data),
                "records_stored": stored_count,
                "records_cached": cached_count
            }
        except Exception as e:
            logger.exception("Error processing market data")
            return {
                "success": False,
                "error": str(e),
                "records_processed": 0
            }

# Register processor
ProcessorRegistry.register("market_data", MarketDataProcessor)
