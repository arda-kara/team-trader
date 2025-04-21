"""
Yahoo Finance market data collector.
"""

import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import yfinance as yf

from ..config.settings import settings
from ..models.base import OHLCV, AssetType, DataSource, TimeFrame
from ..storage.redis_client import market_data_queue
from .base import BaseCollector, CollectorRegistry

logger = logging.getLogger(__name__)

class YahooFinanceMarketDataCollector(BaseCollector):
    """Collector for Yahoo Finance market data."""
    
    def __init__(self):
        """Initialize Yahoo Finance market data collector."""
        super().__init__("market_data", "yahoo", market_data_queue)
    
    def _get_asset_type(self, symbol: str) -> AssetType:
        """Determine asset type from symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            AssetType: Asset type enum
        """
        # Simple heuristic based on symbol format
        if "=" in symbol or "/" in symbol:  # Currency pairs often use these formats
            return AssetType.FOREX
        elif symbol.endswith("-USD") or symbol.endswith("USDT"):
            return AssetType.CRYPTO
        elif symbol.startswith("^"):  # Yahoo Finance index format
            return AssetType.INDEX
        elif "." in symbol:  # Some exchanges use dot notation
            return AssetType.STOCK
        else:
            return AssetType.STOCK
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Yahoo Finance format.
        
        Args:
            timeframe: Time frame string
            
        Returns:
            str: Yahoo Finance interval string
        """
        # Map our timeframes to Yahoo Finance intervals
        timeframe_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo"
        }
        return timeframe_map.get(timeframe, "1d")
    
    async def _fetch_data(self, symbols: List[str], timeframe: str, 
                         start: datetime.datetime, 
                         end: Optional[datetime.datetime] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data from Yahoo Finance.
        
        Args:
            symbols: List of symbols
            timeframe: Time frame string
            start: Start datetime
            end: End datetime
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of dataframes by symbol
        """
        if not end:
            end = datetime.datetime.utcnow()
            
        # Convert timeframe to Yahoo Finance format
        interval = self._convert_timeframe(timeframe)
        
        # Use yfinance to download data
        # Note: yfinance is synchronous, so we run it in a separate thread
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: yf.download(
                tickers=" ".join(symbols),
                start=start,
                end=end,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                prepost=False,
                threads=True
            )
        )
        
        # Handle single symbol case
        if len(symbols) == 1:
            return {symbols[0]: data}
        
        # Handle multiple symbols case
        result = {}
        for symbol in symbols:
            if (symbol, 'Open') in data.columns:
                symbol_data = data[symbol].copy()
                result[symbol] = symbol_data
        
        return result
    
    def _process_data(self, data_dict: Dict[str, pd.DataFrame], timeframe: str) -> List[Dict]:
        """Process Yahoo Finance data into OHLCV models.
        
        Args:
            data_dict: Dictionary of dataframes by symbol
            timeframe: Time frame string
            
        Returns:
            List[Dict]: List of OHLCV dictionaries
        """
        results = []
        
        for symbol, df in data_dict.items():
            # Skip empty dataframes
            if df.empty:
                continue
                
            # Ensure dataframe has required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Dataframe for {symbol} missing required columns")
                continue
                
            # Determine asset type
            asset_type = self._get_asset_type(symbol)
            
            # Convert timeframe to enum value
            try:
                tf_value = TimeFrame(timeframe)
            except ValueError:
                logger.warning(f"Invalid timeframe: {timeframe}")
                continue
                
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Create OHLCV model
                    ohlcv = OHLCV(
                        symbol=symbol,
                        asset_type=asset_type,
                        source=DataSource.YAHOO,
                        timestamp=index.to_pydatetime(),
                        timeframe=tf_value,
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row['Volume'])
                    )
                    
                    results.append(ohlcv.dict())
                except Exception as e:
                    logger.error(f"Error processing row for {symbol}: {e}")
        
        return results
    
    async def collect(self, symbols: Optional[List[str]] = None, 
                     timeframe: str = "1d",
                     lookback_days: int = 1) -> Dict[str, Any]:
        """Collect market data from Yahoo Finance.
        
        Args:
            symbols: List of symbols (defaults to settings)
            timeframe: Time frame (defaults to daily)
            lookback_days: Days to look back (defaults to 1)
            
        Returns:
            Dict[str, Any]: Collection results
        """
        if not symbols:
            # Combine different asset types
            symbols = (
                settings.market_data.default_stocks + 
                [s.replace("/", "-") for s in settings.market_data.default_forex] +
                [s.replace("/", "-") for s in settings.market_data.default_crypto]
            )
            
        if not symbols:
            return {"success": False, "error": "No symbols specified", "records_processed": 0}
            
        # Calculate time range
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=lookback_days)
        
        try:
            # Fetch data
            data_dict = await self._fetch_data(symbols, timeframe, start, end)
            
            # Process data
            processed_data = self._process_data(data_dict, timeframe)
            
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
            logger.exception("Error collecting Yahoo Finance market data")
            return {
                "success": False,
                "error": str(e),
                "records_processed": 0
            }

# Register collector
CollectorRegistry.register("market_data", "yahoo", YahooFinanceMarketDataCollector)
