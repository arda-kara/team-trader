"""
Economic data processor for storing and normalizing economic indicators.
"""

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.exc import IntegrityError

from ..models.base import EconomicIndicator
from ..storage.database import get_db_session
from ..storage.models import EconomicIndicator as EconomicIndicatorModel
from ..storage.redis_client import economic_data_queue, data_cache
from .base import BaseProcessor, ProcessorRegistry

logger = logging.getLogger(__name__)

class EconomicDataProcessor(BaseProcessor):
    """Processor for economic data."""
    
    def __init__(self):
        """Initialize economic data processor."""
        super().__init__("economic_data", economic_data_queue, None)
    
    async def _store_indicators(self, indicators: List[Dict]) -> int:
        """Store economic indicators in database.
        
        Args:
            indicators: List of economic indicator dictionaries
            
        Returns:
            int: Number of records stored
        """
        stored_count = 0
        
        for data in indicators:
            try:
                # Parse data
                indicator = EconomicIndicator(**data)
                
                # Create database model
                db_model = EconomicIndicatorModel(
                    indicator_id=indicator.indicator_id,
                    name=indicator.name,
                    value=indicator.value,
                    timestamp=indicator.timestamp,
                    source=indicator.source,
                    frequency=indicator.frequency,
                    unit=indicator.unit,
                    region=indicator.region,
                    created_at=datetime.datetime.utcnow()
                )
                
                # Store in database
                with get_db_session() as session:
                    # Check if record already exists
                    existing = session.query(EconomicIndicatorModel).filter(
                        EconomicIndicatorModel.indicator_id == indicator.indicator_id,
                        EconomicIndicatorModel.timestamp == indicator.timestamp
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.value = indicator.value
                    else:
                        # Add new record
                        session.add(db_model)
                        
                    stored_count += 1
            except Exception as e:
                logger.error(f"Error storing economic indicator: {e}")
        
        return stored_count
    
    async def _update_cache(self, indicators: List[Dict]) -> int:
        """Update cache with latest economic data.
        
        Args:
            indicators: List of economic indicator dictionaries
            
        Returns:
            int: Number of records cached
        """
        cached_count = 0
        
        # Group by indicator ID
        grouped_data = {}
        for data in indicators:
            indicator_id = data.get("indicator_id")
            
            if not indicator_id:
                continue
                
            if indicator_id not in grouped_data:
                grouped_data[indicator_id] = []
                
            grouped_data[indicator_id].append(data)
        
        # Update cache for each indicator
        for indicator_id, items in grouped_data.items():
            try:
                # Sort by timestamp (newest first)
                sorted_items = sorted(
                    items, 
                    key=lambda x: datetime.datetime.fromisoformat(x["timestamp"]) 
                    if isinstance(x["timestamp"], str) 
                    else x["timestamp"],
                    reverse=True
                )
                
                # Keep only the latest value
                latest_value = sorted_items[0]
                
                # Cache with 1 day TTL
                cache_key = f"economic_data:latest:{indicator_id}"
                data_cache.set(cache_key, latest_value, ttl=86400)
                
                # Also cache historical data (last 100 values)
                historical_values = sorted_items[:100]
                cache_key = f"economic_data:history:{indicator_id}"
                data_cache.set(cache_key, historical_values, ttl=86400)
                
                cached_count += 1 + len(historical_values)
            except Exception as e:
                logger.error(f"Error updating cache for {indicator_id}: {e}")
        
        # Cache all latest indicators
        try:
            latest_indicators = {}
            for indicator_id, items in grouped_data.items():
                sorted_items = sorted(
                    items, 
                    key=lambda x: datetime.datetime.fromisoformat(x["timestamp"]) 
                    if isinstance(x["timestamp"], str) 
                    else x["timestamp"],
                    reverse=True
                )
                if sorted_items:
                    latest_indicators[indicator_id] = sorted_items[0]
            
            cache_key = "economic_data:all_latest"
            data_cache.set(cache_key, latest_indicators, ttl=86400)
        except Exception as e:
            logger.error(f"Error updating all_latest cache: {e}")
        
        return cached_count
    
    async def process(self, data: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Process economic data.
        
        Args:
            data: Economic data to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if not isinstance(data, list):
            data = [data]
            
        try:
            # Store in database
            stored_count = await self._store_indicators(data)
            
            # Update cache
            cached_count = await self._update_cache(data)
            
            return {
                "success": True,
                "records_processed": len(data),
                "records_stored": stored_count,
                "records_cached": cached_count
            }
        except Exception as e:
            logger.exception("Error processing economic data")
            return {
                "success": False,
                "error": str(e),
                "records_processed": 0
            }

# Register processor
ProcessorRegistry.register("economic_data", EconomicDataProcessor)
