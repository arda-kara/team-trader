"""
Economic data collector for FRED API.
"""

import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

from ..config.settings import settings
from ..models.base import EconomicIndicator
from ..storage.redis_client import economic_data_queue
from .base import BaseCollector, CollectorRegistry

logger = logging.getLogger(__name__)

class FREDEconomicDataCollector(BaseCollector):
    """Collector for FRED economic data."""
    
    def __init__(self):
        """Initialize FRED economic data collector."""
        super().__init__("economic_data", "fred", economic_data_queue)
        self.api_key = settings.economic_data.fred_api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        
        if not self.api_key:
            logger.warning("FRED API key not configured")
    
    async def _fetch_series(self, series_id: str) -> Dict[str, Any]:
        """Fetch series information from FRED.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dict[str, Any]: API response
        """
        url = f"{self.base_url}/series"
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"FRED API error: {response.status} - {error_text}")
                
                return await response.json()
    
    async def _fetch_observations(self, series_id: str, 
                                 start_date: Optional[datetime.datetime] = None,
                                 end_date: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """Fetch observations for a series from FRED.
        
        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict[str, Any]: API response
        """
        url = f"{self.base_url}/series/observations"
        
        # Format dates for FRED API
        if start_date:
            start_str = start_date.strftime("%Y-%m-%d")
        else:
            # Default to 1 year ago
            start_str = (datetime.datetime.utcnow() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            
        if end_date:
            end_str = end_date.strftime("%Y-%m-%d")
        else:
            end_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_str,
            "observation_end": end_str,
            "sort_order": "desc"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"FRED API error: {response.status} - {error_text}")
                
                return await response.json()
    
    def _process_observations(self, series_id: str, series_info: Dict[str, Any], 
                             observations: Dict[str, Any]) -> List[Dict]:
        """Process observations into EconomicIndicator models.
        
        Args:
            series_id: FRED series ID
            series_info: Series information
            observations: Observations data
            
        Returns:
            List[Dict]: List of EconomicIndicator dictionaries
        """
        results = []
        
        # Extract series metadata
        series = series_info.get("seriess", [{}])[0]
        name = series.get("title", "")
        frequency = series.get("frequency_short", "")
        units = series.get("units_short", "")
        
        # Process observations
        for obs in observations.get("observations", []):
            try:
                # Skip missing values
                value = obs.get("value")
                if value == "." or value is None:
                    continue
                    
                # Parse date
                date_str = obs.get("date")
                if not date_str:
                    continue
                    
                date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                
                # Create EconomicIndicator model
                indicator = EconomicIndicator(
                    indicator_id=series_id,
                    name=name,
                    value=float(value),
                    timestamp=date,
                    source="fred",
                    frequency=frequency,
                    unit=units,
                    region="US"  # Most FRED data is US-based
                )
                
                results.append(indicator.dict())
            except Exception as e:
                logger.error(f"Error processing observation for {series_id}: {e}")
        
        return results
    
    async def collect(self, indicators: Optional[List[str]] = None,
                     lookback_days: int = 365) -> Dict[str, Any]:
        """Collect economic data from FRED.
        
        Args:
            indicators: List of indicator IDs (defaults to settings)
            lookback_days: Days to look back (defaults to 365)
            
        Returns:
            Dict[str, Any]: Collection results
        """
        if not self.api_key:
            return {"success": False, "error": "FRED API key not configured", "records_processed": 0}
            
        if not indicators:
            indicators = settings.economic_data.default_indicators
            
        if not indicators:
            return {"success": False, "error": "No indicators specified", "records_processed": 0}
            
        # Calculate time range
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=lookback_days)
        
        all_results = []
        failed_indicators = []
        
        for indicator_id in indicators:
            try:
                # Fetch series info
                series_info = await self._fetch_series(indicator_id)
                
                # Fetch observations
                observations = await self._fetch_observations(indicator_id, start, end)
                
                # Process observations
                processed_data = self._process_observations(indicator_id, series_info, observations)
                
                # Add to results
                all_results.extend(processed_data)
            except Exception as e:
                logger.error(f"Error collecting data for indicator {indicator_id}: {e}")
                failed_indicators.append(indicator_id)
        
        # Publish to queue
        if all_results:
            self._publish_data(all_results)
        
        return {
            "success": len(failed_indicators) < len(indicators),
            "records_processed": len(all_results),
            "indicators": indicators,
            "failed_indicators": failed_indicators,
            "start": start.isoformat(),
            "end": end.isoformat()
        }

# Register collector
CollectorRegistry.register("economic_data", "fred", FREDEconomicDataCollector)
