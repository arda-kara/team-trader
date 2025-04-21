"""
Data client for accessing market and semantic data from APIs.
"""

import logging
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..config.settings import settings
from ..generators.redis_client import market_data_cache
from ..models.base import MarketData, Signal, TimeFrame

logger = logging.getLogger(__name__)

class DataClient:
    """Client for accessing market and semantic data."""
    
    def __init__(self):
        """Initialize data client."""
        self.data_api_url = settings.data_api_url
        self.semantic_api_url = settings.semantic_api_url
    
    async def _make_request(self, method: str, url: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            data: Request data for POST/PUT
            
        Returns:
            Dict: Response data
        """
        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
    
    async def get_market_data(self, symbols: List[str], timeframe: TimeFrame, 
                            start_date: datetime, end_date: datetime,
                            use_cache: bool = True) -> List[MarketData]:
        """Get market data for symbols.
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache
            
        Returns:
            List[MarketData]: Market data
        """
        # Check cache if enabled
        if use_cache:
            cache_key = f"market_data:{','.join(symbols)}:{timeframe}:{start_date.isoformat()}:{end_date.isoformat()}"
            cached_data = market_data_cache.get(cache_key)
            
            if cached_data:
                # Convert cached data to MarketData objects
                return [MarketData(**item) for item in cached_data]
        
        # Prepare request
        url = f"{self.data_api_url}/market"
        data = {
            "symbols": symbols,
            "timeframe": timeframe,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }
        
        try:
            # Make request
            response = await self._make_request("POST", url, data)
            
            # Parse response
            market_data = []
            for item in response.get("data", []):
                market_data.append(MarketData(
                    symbol=item["symbol"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    open=item["open"],
                    high=item["high"],
                    low=item["low"],
                    close=item["close"],
                    volume=item["volume"],
                    timeframe=timeframe
                ))
            
            # Cache result if enabled
            if use_cache:
                market_data_cache.set(
                    cache_key,
                    [item.dict() for item in market_data],
                    ttl=3600  # 1 hour
                )
            
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return []
    
    async def get_semantic_signals(self, symbols: List[str], start_date: datetime, 
                                 end_date: datetime, signal_types: Optional[List[str]] = None,
                                 min_confidence: float = 0.5) -> List[Signal]:
        """Get semantic signals for symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            signal_types: Optional list of signal types
            min_confidence: Minimum confidence threshold
            
        Returns:
            List[Signal]: Semantic signals
        """
        # Prepare request
        url = f"{self.semantic_api_url}/signals/query"
        data = {
            "symbols": symbols,
            "start_time": start_date.isoformat(),
            "end_time": end_date.isoformat(),
            "min_confidence": min_confidence
        }
        
        if signal_types:
            data["signal_types"] = signal_types
        
        try:
            # Make request
            response = await self._make_request("POST", url, data)
            
            # Parse response
            signals = []
            for item in response.get("signals", []):
                # Map semantic signal to strategy signal
                signal = Signal(
                    id=item["id"],
                    symbol=item["symbols"][0] if item["symbols"] else "",  # Take first symbol
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    type="sentiment" if item["type"] == "SENTIMENT" else "event",
                    direction="bullish" if item["direction"] == "BULLISH" else 
                              "bearish" if item["direction"] == "BEARISH" else "neutral",
                    strength=item["confidence"],
                    timeframe=TimeFrame.DAY_1,  # Default to daily
                    source="semantic_analysis",
                    metadata=item["details"]
                )
                signals.append(signal)
            
            return signals
        except Exception as e:
            logger.error(f"Error getting semantic signals: {e}")
            return []
    
    async def get_economic_indicators(self, indicators: List[str], start_date: datetime, 
                                    end_date: datetime) -> Dict[str, List[Dict]]:
        """Get economic indicators.
        
        Args:
            indicators: List of indicator IDs
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict[str, List[Dict]]: Economic indicators by ID
        """
        # Prepare request
        url = f"{self.data_api_url}/economic"
        data = {
            "indicators": indicators,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }
        
        try:
            # Make request
            response = await self._make_request("POST", url, data)
            
            # Parse response
            result = {}
            for item in response.get("data", []):
                indicator_id = item["indicator_id"]
                if indicator_id not in result:
                    result[indicator_id] = []
                
                result[indicator_id].append({
                    "timestamp": datetime.fromisoformat(item["timestamp"]),
                    "value": item["value"],
                    "name": item["name"],
                    "unit": item.get("unit", ""),
                    "region": item.get("region", "")
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}")
            return {}

# Create client instance
data_client = DataClient()
