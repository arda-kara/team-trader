"""
Market data collector for Alpaca API.
"""

import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from pydantic import ValidationError

from ..config.settings import settings
from ..models.base import OHLCV, AssetType, DataSource, TimeFrame
from ..storage.redis_client import market_data_queue
from .base import BaseCollector, CollectorRegistry

logger = logging.getLogger(__name__)

class AlpacaMarketDataCollector(BaseCollector):
    """Collector for Alpaca market data."""
    
    def __init__(self):
        """Initialize Alpaca market data collector."""
        super().__init__("market_data", "alpaca", market_data_queue)
        self.api_key = settings.market_data.alpaca_api_key
        self.api_secret = settings.market_data.alpaca_api_secret
        self.base_url = settings.market_data.alpaca_base_url
        self.data_url = "https://data.alpaca.markets"
        
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca API credentials not configured")
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get API request headers.
        
        Returns:
            Dict[str, str]: Headers dictionary
        """
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
    
    async def _fetch_bars(self, symbols: List[str], timeframe: str, start: datetime.datetime, 
                         end: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """Fetch OHLCV bars from Alpaca.
        
        Args:
            symbols: List of symbols
            timeframe: Time frame (e.g., "1D", "1H")
            start: Start datetime
            end: End datetime (defaults to now)
            
        Returns:
            Dict[str, Any]: API response
        """
        if not end:
            end = datetime.datetime.utcnow()
            
        # Format dates for Alpaca API
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Convert timeframe to Alpaca format
        timeframe_map = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "1d": "1Day"
        }
        alpaca_timeframe = timeframe_map.get(timeframe, "1Day")
        
        url = f"{self.data_url}/v2/stocks/bars"
        params = {
            "symbols": ",".join(symbols),
            "timeframe": alpaca_timeframe,
            "start": start_str,
            "end": end_str,
            "limit": 10000,
            "adjustment": "raw"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=await self._get_headers(), params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Alpaca API error: {response.status} - {error_text}")
                
                return await response.json()
    
    async def _process_bars(self, bars_data: Dict[str, Any], timeframe: str) -> List[Dict]:
        """Process bars data into OHLCV models.
        
        Args:
            bars_data: Bars data from API
            timeframe: Time frame string
            
        Returns:
            List[Dict]: List of OHLCV dictionaries
        """
        results = []
        
        for symbol, bars in bars_data.get("bars", {}).items():
            for bar in bars:
                try:
                    # Convert timeframe to enum value
                    tf_value = TimeFrame(timeframe)
                    
                    # Create OHLCV model
                    ohlcv = OHLCV(
                        symbol=symbol,
                        asset_type=AssetType.STOCK,
                        source=DataSource.ALPACA,
                        timestamp=datetime.datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
                        timeframe=tf_value,
                        open=bar["o"],
                        high=bar["h"],
                        low=bar["l"],
                        close=bar["c"],
                        volume=bar["v"]
                    )
                    
                    results.append(ohlcv.dict())
                except ValidationError as e:
                    logger.error(f"Validation error for bar data: {e}")
                except Exception as e:
                    logger.error(f"Error processing bar data: {e}")
        
        return results
    
    async def collect(self, symbols: Optional[List[str]] = None, 
                     timeframe: str = "1d",
                     lookback_days: int = 1) -> Dict[str, Any]:
        """Collect market data from Alpaca.
        
        Args:
            symbols: List of symbols (defaults to settings)
            timeframe: Time frame (defaults to daily)
            lookback_days: Days to look back (defaults to 1)
            
        Returns:
            Dict[str, Any]: Collection results
        """
        if not symbols:
            symbols = settings.market_data.default_stocks
            
        if not symbols:
            return {"success": False, "error": "No symbols specified", "records_processed": 0}
            
        # Calculate time range
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=lookback_days)
        
        try:
            # Fetch data
            bars_data = await self._fetch_bars(symbols, timeframe, start, end)
            
            # Process data
            processed_data = await self._process_bars(bars_data, timeframe)
            
            # Publish to queue
            if processed_data:
                self._publish_data(processed_data)
            
            return {
                "success": True,
                "records_processed": len(processed_data),
                "symbols": symbols,
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat()
            }
        except Exception as e:
            logger.exception("Error collecting Alpaca market data")
            return {
                "success": False,
                "error": str(e),
                "records_processed": 0
            }

# Register collector
CollectorRegistry.register("market_data", "alpaca", AlpacaMarketDataCollector)
