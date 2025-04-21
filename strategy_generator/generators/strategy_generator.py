"""
Strategy generator for creating trading strategies based on market data and semantic signals.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..models.base import (
    Strategy, StrategyType, TimeFrame, PositionSizingMethod,
    StrategyParameter, Signal, MarketData
)
from ..generators.data_client import data_client
from ..generators.redis_client import strategy_cache

logger = logging.getLogger(__name__)

class StrategyGenerator:
    """Generator for trading strategies."""
    
    def __init__(self):
        """Initialize strategy generator."""
        self.strategy_types = settings.strategy.available_types
        self.default_params = settings.strategy.default_params
        self.signal_weights = settings.strategy.signal_weights
    
    def _generate_strategy_id(self) -> str:
        """Generate unique strategy ID.
        
        Returns:
            str: Strategy ID
        """
        return f"strategy_{uuid.uuid4().hex[:8]}"
    
    def _get_default_parameters(self, strategy_type: str) -> List[StrategyParameter]:
        """Get default parameters for strategy type.
        
        Args:
            strategy_type: Strategy type
            
        Returns:
            List[StrategyParameter]: Default parameters
        """
        if strategy_type not in self.default_params:
            return []
        
        params = []
        for name, value in self.default_params[strategy_type].items():
            # Create parameter with default values
            if isinstance(value, int):
                param = StrategyParameter(
                    name=name,
                    value=value,
                    min_value=max(1, int(value * 0.5)),
                    max_value=int(value * 2),
                    step=1
                )
            elif isinstance(value, float):
                param = StrategyParameter(
                    name=name,
                    value=value,
                    min_value=max(0.1, value * 0.5),
                    max_value=value * 2,
                    step=0.1
                )
            else:
                param = StrategyParameter(
                    name=name,
                    value=value
                )
            
            params.append(param)
        
        return params
    
    def _get_entry_exit_rules(self, strategy_type: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get default entry and exit rules for strategy type.
        
        Args:
            strategy_type: Strategy type
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Entry and exit rules
        """
        # Default rules based on strategy type
        if strategy_type == "trend_following":
            entry_rules = {
                "condition": "macd_crossover",
                "direction": "bullish"
            }
            exit_rules = {
                "condition": "macd_crossover",
                "direction": "bearish"
            }
        
        elif strategy_type == "mean_reversion":
            entry_rules = {
                "condition": "bollinger_bands",
                "direction": "oversold"
            }
            exit_rules = {
                "condition": "bollinger_bands",
                "direction": "overbought"
            }
        
        elif strategy_type == "breakout":
            entry_rules = {
                "condition": "price_breakout",
                "direction": "bullish",
                "lookback_period": 20
            }
            exit_rules = {
                "condition": "trailing_stop",
                "atr_multiplier": 2.0
            }
        
        elif strategy_type == "momentum":
            entry_rules = {
                "condition": "rsi",
                "direction": "bullish",
                "threshold": 50
            }
            exit_rules = {
                "condition": "rsi",
                "direction": "bearish",
                "threshold": 50
            }
        
        elif strategy_type == "sentiment_based":
            entry_rules = {
                "condition": "sentiment",
                "direction": "bullish",
                "threshold": 0.5
            }
            exit_rules = {
                "condition": "sentiment",
                "direction": "bearish",
                "threshold": -0.2
            }
        
        elif strategy_type == "event_driven":
            entry_rules = {
                "condition": "event",
                "event_types": ["EARNINGS_REPORT", "PRODUCT_LAUNCH"],
                "direction": "bullish",
                "confidence": 0.7
            }
            exit_rules = {
                "condition": "time_based",
                "holding_period": 5  # days
            }
        
        elif strategy_type == "ml_based":
            entry_rules = {
                "condition": "ml_prediction",
                "direction": "bullish",
                "confidence": 0.6
            }
            exit_rules = {
                "condition": "ml_prediction",
                "direction": "bearish",
                "confidence": 0.6
            }
        
        else:
            # Default generic rules
            entry_rules = {
                "condition": "custom",
                "direction": "bullish"
            }
            exit_rules = {
                "condition": "custom",
                "direction": "bearish"
            }
        
        return entry_rules, exit_rules
    
    def _get_risk_management_rules(self, strategy_type: str) -> Dict[str, Any]:
        """Get default risk management rules for strategy type.
        
        Args:
            strategy_type: Strategy type
            
        Returns:
            Dict[str, Any]: Risk management rules
        """
        # Default risk management based on strategy type
        if strategy_type == "trend_following":
            return {
                "stop_loss": 0.02,  # 2%
                "trailing_stop": True,
                "trailing_stop_activation": 0.01,  # 1%
                "trailing_stop_distance": 0.02,  # 2%
                "take_profit": None
            }
        
        elif strategy_type == "mean_reversion":
            return {
                "stop_loss": 0.03,  # 3%
                "trailing_stop": False,
                "take_profit": 0.02  # 2%
            }
        
        elif strategy_type == "breakout":
            return {
                "stop_loss": 0.02,  # 2%
                "trailing_stop": True,
                "trailing_stop_activation": 0.015,  # 1.5%
                "trailing_stop_distance": 0.02,  # 2%
                "take_profit": 0.04  # 4%
            }
        
        elif strategy_type == "momentum":
            return {
                "stop_loss": 0.025,  # 2.5%
                "trailing_stop": True,
                "trailing_stop_activation": 0.02,  # 2%
                "trailing_stop_distance": 0.025,  # 2.5%
                "take_profit": 0.05  # 5%
            }
        
        elif strategy_type == "sentiment_based":
            return {
                "stop_loss": 0.03,  # 3%
                "trailing_stop": False,
                "take_profit": 0.04  # 4%
            }
        
        elif strategy_type == "event_driven":
            return {
                "stop_loss": 0.04,  # 4%
                "trailing_stop": False,
                "take_profit": 0.06  # 6%
            }
        
        elif strategy_type == "ml_based":
            return {
                "stop_loss": 0.025,  # 2.5%
                "trailing_stop": True,
                "trailing_stop_activation": 0.015,  # 1.5%
                "trailing_stop_distance": 0.02,  # 2%
                "take_profit": None
            }
        
        else:
            # Default generic risk management
            return {
                "stop_loss": 0.02,  # 2%
                "trailing_stop": False,
                "take_profit": 0.03  # 3%
            }
    
    def create_strategy(self, name: str, strategy_type: str, symbols: List[str], 
                      timeframe: TimeFrame, description: Optional[str] = None,
                      parameters: Optional[List[StrategyParameter]] = None,
                      position_sizing: Optional[PositionSizingMethod] = None,
                      position_size: Optional[float] = None,
                      entry_rules: Optional[Dict[str, Any]] = None,
                      exit_rules: Optional[Dict[str, Any]] = None,
                      risk_management: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Strategy:
        """Create a new trading strategy.
        
        Args:
            name: Strategy name
            strategy_type: Strategy type
            symbols: List of symbols
            timeframe: Timeframe
            description: Optional strategy description
            parameters: Optional strategy parameters
            position_sizing: Optional position sizing method
            position_size: Optional position size
            entry_rules: Optional entry rules
            exit_rules: Optional exit rules
            risk_management: Optional risk management rules
            metadata: Optional metadata
            
        Returns:
            Strategy: Created strategy
        """
        # Generate strategy ID
        strategy_id = self._generate_strategy_id()
        
        # Set defaults if not provided
        if description is None:
            description = f"{name} - A {strategy_type} strategy for {', '.join(symbols)}"
        
        if parameters is None:
            parameters = self._get_default_parameters(strategy_type)
        
        if position_sizing is None:
            position_sizing = PositionSizingMethod(settings.strategy.default_position_sizing)
        
        if position_size is None:
            position_size = settings.strategy.default_position_size
        
        if entry_rules is None or exit_rules is None:
            default_entry, default_exit = self._get_entry_exit_rules(strategy_type)
            
            if entry_rules is None:
                entry_rules = default_entry
            
            if exit_rules is None:
                exit_rules = default_exit
        
        if risk_management is None:
            risk_management = self._get_risk_management_rules(strategy_type)
        
        if metadata is None:
            metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        
        # Create strategy
        strategy = Strategy(
            id=strategy_id,
            name=name,
            type=StrategyType(strategy_type),
            description=description,
            parameters=parameters,
            symbols=symbols,
            timeframe=timeframe,
            position_sizing=position_sizing,
            position_size=position_size,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            risk_management=risk_management,
            metadata=metadata
        )
        
        # Cache strategy
        strategy_cache.set(f"strategy:{strategy_id}", strategy.dict())
        
        return strategy
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Get strategy by ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Optional[Strategy]: Strategy or None if not found
        """
        # Get from cache
        strategy_data = strategy_cache.get(f"strategy:{strategy_id}")
        
        if strategy_data:
            return Strategy(**strategy_data)
        
        return None
    
    def update_strategy(self, strategy: Strategy) -> Strategy:
        """Update strategy.
        
        Args:
            strategy: Strategy to update
            
        Returns:
            Strategy: Updated strategy
        """
        # Update metadata
        if "updated_at" in strategy.metadata:
            strategy.metadata["updated_at"] = datetime.utcnow().isoformat()
        
        # Cache updated strategy
        strategy_cache.set(f"strategy:{strategy.id}", strategy.dict())
        
        return strategy
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """Delete strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            bool: Success status
        """
        return strategy_cache.delete(f"strategy:{strategy_id}")
    
    async def generate_signals(self, strategy: Strategy, timestamp: datetime) -> List[Signal]:
        """Generate signals for strategy at timestamp.
        
        Args:
            strategy: Strategy
            timestamp: Timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        signals = []
        
        try:
            # Get market data
            start_date = timestamp - timedelta(days=30)  # Get 30 days of data
            end_date = timestamp
            
            market_data = await data_client.get_market_data(
                strategy.symbols,
                strategy.timeframe,
                start_date,
                end_date
            )
            
            # Get semantic signals if needed
            semantic_signals = []
            if strategy.type in [StrategyType.SENTIMENT_BASED, StrategyType.EVENT_DRIVEN]:
                semantic_signals = await data_client.get_semantic_signals(
                    strategy.symbols,
                    start_date,
                    end_date
                )
            
            # Generate signals based on strategy type
            if strategy.type == StrategyType.TREND_FOLLOWING:
                signals = self._generate_trend_following_signals(strategy, market_data, timestamp)
            
            elif strategy.type == StrategyType.MEAN_REVERSION:
                signals = self._generate_mean_reversion_signals(strategy, market_data, timestamp)
            
            elif strategy.type == StrategyType.BREAKOUT:
                signals = self._generate_breakout_signals(strategy, market_data, timestamp)
            
            elif strategy.type == StrategyType.MOMENTUM:
                signals = self._generate_momentum_signals(strategy, market_data, timestamp)
            
            elif strategy.type == StrategyType.SENTIMENT_BASED:
                signals = self._generate_sentiment_signals(strategy, market_data, semantic_signals, timestamp)
            
            elif strategy.type == StrategyType.EVENT_DRIVEN:
                signals = self._generate_event_signals(strategy, market_data, semantic_signals, timestamp)
            
            elif strategy.type == StrategyType.ML_BASED:
                signals = await self._generate_ml_signals(strategy, market_data, timestamp)
            
            return signals
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def _generate_trend_following_signals(self, strategy: Strategy, 
                                        market_data: List[MarketData],
                                        timestamp: datetime) -> List[Signal]:
        """Generate trend following signals.
        
        Args:
            strategy: Strategy
            market_data: Market data
            timestamp: Timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        signals = []
        
        # Get parameters
        fast_period = next((p.value for p in strategy.parameters if p.name == "fast_period"), 12)
        slow_period = next((p.value for p in strategy.parameters if p.name == "slow_period"), 26)
        signal_period = next((p.value for p in strategy.parameters if p.name == "signal_period"), 9)
        
        # Process each symbol
        for symbol in strategy.symbols:
            # Filter data for symbol
            symbol_data = [d for d in market_data if d.symbol == symbol]
            
            if not symbol_data:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                "timestamp": d.timestamp,
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "volume": d.volume
            } for d in symbol_data])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate MACD
            ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal_period, adjust=False).mean()
            histogram = macd - signal_line
            
            # Check for crossover
            if len(histogram) >= 2:
                prev_histogram = histogram.iloc[-2]
                curr_histogram = histogram.iloc[-1]
                
                # Bullish crossover (histogram crosses above zero)
                if prev_histogram < 0 and curr_histogram > 0:
                    signal = Signal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        timestamp=timestamp,
                        type="technical",
                        direction="bullish",
                        strength=min(1.0, abs(curr_histogram) * 10),  # Scale strength
                        timeframe=strategy.timeframe,
                        source="macd_crossover",
                        metadata={
                            "fast_period": fast_period,
                            "slow_period": slow_period,
                            "signal_period": signal_period,
                            "macd": macd.iloc[-1],
                            "signal_line": signal_line.iloc[-1],
                            "histogram": curr_histogram
                        }
                    )
                    signals.append(signal)
                
                # Bearish crossover (histogram crosses below zero)
                elif prev_histogram > 0 and curr_histogram < 0:
                    signal = Signal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        timestamp=timestamp,
                        type="technical",
                        direction="bearish",
                        strength=min(1.0, abs(curr_histogram) * 10),  # Scale strength
                        timeframe=strategy.timeframe,
                        source="macd_crossover",
                        metadata={
                            "fast_period": fast_period,
                            "slow_period": slow_period,
                            "signal_period": signal_period,
                            "macd": macd.iloc[-1],
                            "signal_line": signal_line.iloc[-1],
                            "histogram": curr_histogram
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def _generate_mean_reversion_signals(self, strategy: Strategy, 
                                       market_data: List[MarketData],
                                       timestamp: datetime) -> List[Signal]:
        """Generate mean reversion signals.
        
        Args:
            strategy: Strategy
            market_data: Market data
            timestamp: Timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        signals = []
        
        # Get parameters
        lookback_period = next((p.value for p in strategy.parameters if p.name == "lookback_period"), 20)
        entry_zscore = next((p.value for p in strategy.parameters if p.name == "entry_zscore"), 2.0)
        
        # Process each symbol
        for symbol in strategy.symbols:
            # Filter data for symbol
            symbol_data = [d for d in market_data if d.symbol == symbol]
            
            if not symbol_data:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                "timestamp": d.timestamp,
                "close": d.close
            } for d in symbol_data])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate Bollinger Bands
            df["sma"] = df["close"].rolling(window=lookback_period).mean()
            df["std"] = df["close"].rolling(window=lookback_period).std()
            df["upper_band"] = df["sma"] + (df["std"] * entry_zscore)
            df["lower_band"] = df["sma"] - (df["std"] * entry_zscore)
            df["zscore"] = (df["close"] - df["sma"]) / df["std"]
            
            # Check for signals
            if len(df) >= lookback_period:
                last_row = df.iloc[-1]
                
                # Oversold signal (price below lower band)
                if last_row["close"] < last_row["lower_band"]:
                    zscore = last_row["zscore"]
                    signal = Signal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        timestamp=timestamp,
                        type="technical",
                        direction="bullish",
                        strength=min(1.0, abs(zscore) / entry_zscore),  # Scale strength
                        timeframe=strategy.timeframe,
                        source="bollinger_bands_oversold",
                        metadata={
                            "lookback_period": lookback_period,
                            "entry_zscore": entry_zscore,
                            "current_zscore": zscore,
                            "sma": last_row["sma"],
                            "lower_band": last_row["lower_band"],
                            "price": last_row["close"]
                        }
                    )
                    signals.append(signal)
                
                # Overbought signal (price above upper band)
                elif last_row["close"] > last_row["upper_band"]:
                    zscore = last_row["zscore"]
                    signal = Signal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        timestamp=timestamp,
                        type="technical",
                        direction="bearish",
                        strength=min(1.0, abs(zscore) / entry_zscore),  # Scale strength
                        timeframe=strategy.timeframe,
                        source="bollinger_bands_overbought",
                        metadata={
                            "lookback_period": lookback_period,
                            "entry_zscore": entry_zscore,
                            "current_zscore": zscore,
                            "sma": last_row["sma"],
                            "upper_band": last_row["upper_band"],
                            "price": last_row["close"]
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def _generate_breakout_signals(self, strategy: Strategy, 
                                 market_data: List[MarketData],
                                 timestamp: datetime) -> List[Signal]:
        """Generate breakout signals.
        
        Args:
            strategy: Strategy
            market_data: Market data
            timestamp: Timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        signals = []
        
        # Get parameters
        channel_period = next((p.value for p in strategy.parameters if p.name == "channel_period"), 20)
        min_volume_multiplier = next((p.value for p in strategy.parameters if p.name == "min_volume_multiplier"), 1.5)
        
        # Process each symbol
        for symbol in strategy.symbols:
            # Filter data for symbol
            symbol_data = [d for d in market_data if d.symbol == symbol]
            
            if not symbol_data:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                "timestamp": d.timestamp,
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "volume": d.volume
            } for d in symbol_data])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate channel
            df["highest_high"] = df["high"].rolling(window=channel_period).max()
            df["lowest_low"] = df["low"].rolling(window=channel_period).min()
            
            # Calculate volume metrics
            df["avg_volume"] = df["volume"].rolling(window=channel_period).mean()
            df["volume_ratio"] = df["volume"] / df["avg_volume"]
            
            # Check for breakout
            if len(df) >= channel_period + 1:
                prev_row = df.iloc[-2]
                last_row = df.iloc[-1]
                
                # Bullish breakout (price breaks above highest high with increased volume)
                if (prev_row["close"] < prev_row["highest_high"] and 
                    last_row["close"] > last_row["highest_high"] and 
                    last_row["volume_ratio"] >= min_volume_multiplier):
                    
                    # Calculate breakout strength
                    breakout_pct = (last_row["close"] - last_row["highest_high"]) / last_row["highest_high"]
                    strength = min(1.0, breakout_pct * 100)  # Scale strength
                    
                    signal = Signal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        timestamp=timestamp,
                        type="technical",
                        direction="bullish",
                        strength=strength,
                        timeframe=strategy.timeframe,
                        source="price_breakout_bullish",
                        metadata={
                            "channel_period": channel_period,
                            "breakout_level": last_row["highest_high"],
                            "breakout_price": last_row["close"],
                            "breakout_pct": breakout_pct,
                            "volume_ratio": last_row["volume_ratio"],
                            "min_volume_multiplier": min_volume_multiplier
                        }
                    )
                    signals.append(signal)
                
                # Bearish breakout (price breaks below lowest low with increased volume)
                elif (prev_row["close"] > prev_row["lowest_low"] and 
                      last_row["close"] < last_row["lowest_low"] and 
                      last_row["volume_ratio"] >= min_volume_multiplier):
                    
                    # Calculate breakout strength
                    breakout_pct = (last_row["lowest_low"] - last_row["close"]) / last_row["lowest_low"]
                    strength = min(1.0, breakout_pct * 100)  # Scale strength
                    
                    signal = Signal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        timestamp=timestamp,
                        type="technical",
                        direction="bearish",
                        strength=strength,
                        timeframe=strategy.timeframe,
                        source="price_breakout_bearish",
                        metadata={
                            "channel_period": channel_period,
                            "breakout_level": last_row["lowest_low"],
                            "breakout_price": last_row["close"],
                            "breakout_pct": breakout_pct,
                            "volume_ratio": last_row["volume_ratio"],
                            "min_volume_multiplier": min_volume_multiplier
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def _generate_momentum_signals(self, strategy: Strategy, 
                                 market_data: List[MarketData],
                                 timestamp: datetime) -> List[Signal]:
        """Generate momentum signals.
        
        Args:
            strategy: Strategy
            market_data: Market data
            timestamp: Timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        signals = []
        
        # Get parameters
        momentum_period = next((p.value for p in strategy.parameters if p.name == "momentum_period"), 14)
        signal_period = next((p.value for p in strategy.parameters if p.name == "signal_period"), 3)
        threshold = next((p.value for p in strategy.parameters if p.name == "threshold"), 0.0)
        
        # Process each symbol
        for symbol in strategy.symbols:
            # Filter data for symbol
            symbol_data = [d for d in market_data if d.symbol == symbol]
            
            if not symbol_data:
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                "timestamp": d.timestamp,
                "close": d.close
            } for d in symbol_data])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=momentum_period).mean()
            avg_loss = loss.rolling(window=momentum_period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate RSI signal line
            rsi_signal = rsi.rolling(window=signal_period).mean()
            
            # Check for signals
            if len(rsi) >= momentum_period + signal_period:
                last_rsi = rsi.iloc[-1]
                last_signal = rsi_signal.iloc[-1]
                
                # Bullish signal (RSI crosses above signal line and above threshold)
                if rsi.iloc[-2] < rsi_signal.iloc[-2] and last_rsi > last_signal and last_rsi > threshold:
                    # Calculate strength based on RSI value
                    strength = min(1.0, (last_rsi - threshold) / (100 - threshold))
                    
                    signal = Signal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        timestamp=timestamp,
                        type="technical",
                        direction="bullish",
                        strength=strength,
                        timeframe=strategy.timeframe,
                        source="rsi_bullish",
                        metadata={
                            "momentum_period": momentum_period,
                            "signal_period": signal_period,
                            "threshold": threshold,
                            "rsi": last_rsi,
                            "rsi_signal": last_signal
                        }
                    )
                    signals.append(signal)
                
                # Bearish signal (RSI crosses below signal line and below 100-threshold)
                elif rsi.iloc[-2] > rsi_signal.iloc[-2] and last_rsi < last_signal and last_rsi < (100 - threshold):
                    # Calculate strength based on RSI value
                    strength = min(1.0, ((100 - threshold) - last_rsi) / (100 - threshold))
                    
                    signal = Signal(
                        id=f"signal_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        timestamp=timestamp,
                        type="technical",
                        direction="bearish",
                        strength=strength,
                        timeframe=strategy.timeframe,
                        source="rsi_bearish",
                        metadata={
                            "momentum_period": momentum_period,
                            "signal_period": signal_period,
                            "threshold": threshold,
                            "rsi": last_rsi,
                            "rsi_signal": last_signal
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def _generate_sentiment_signals(self, strategy: Strategy, 
                                  market_data: List[MarketData],
                                  semantic_signals: List[Signal],
                                  timestamp: datetime) -> List[Signal]:
        """Generate sentiment-based signals.
        
        Args:
            strategy: Strategy
            market_data: Market data
            semantic_signals: Semantic signals
            timestamp: Timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        signals = []
        
        # Get parameters
        sentiment_threshold = next((p.value for p in strategy.parameters if p.name == "sentiment_threshold"), 0.5)
        lookback_period = next((p.value for p in strategy.parameters if p.name == "lookback_period"), 3)
        signal_weight = next((p.value for p in strategy.parameters if p.name == "signal_weight"), 0.7)
        price_weight = next((p.value for p in strategy.parameters if p.name == "price_weight"), 0.3)
        
        # Process each symbol
        for symbol in strategy.symbols:
            # Filter semantic signals for symbol
            symbol_semantic = [s for s in semantic_signals if s.symbol == symbol]
            
            # Filter by recency (last lookback_period days)
            cutoff_time = timestamp - timedelta(days=lookback_period)
            recent_signals = [s for s in symbol_semantic if s.timestamp >= cutoff_time]
            
            if not recent_signals:
                continue
            
            # Calculate average sentiment
            bullish_signals = [s for s in recent_signals if s.direction == "bullish"]
            bearish_signals = [s for s in recent_signals if s.direction == "bearish"]
            
            bullish_strength = sum(s.strength for s in bullish_signals) / max(1, len(bullish_signals))
            bearish_strength = sum(s.strength for s in bearish_signals) / max(1, len(bearish_signals))
            
            # Get price trend
            symbol_data = [d for d in market_data if d.symbol == symbol]
            price_trend = 0.0
            
            if len(symbol_data) >= 2:
                # Sort by timestamp
                sorted_data = sorted(symbol_data, key=lambda x: x.timestamp)
                first_price = sorted_data[0].close
                last_price = sorted_data[-1].close
                price_change = (last_price - first_price) / first_price
                price_trend = min(1.0, max(-1.0, price_change * 10))  # Scale to [-1, 1]
            
            # Combine sentiment and price trend
            combined_bullish = (bullish_strength * signal_weight) + (max(0, price_trend) * price_weight)
            combined_bearish = (bearish_strength * signal_weight) + (max(0, -price_trend) * price_weight)
            
            # Generate signals if above threshold
            if combined_bullish > sentiment_threshold:
                signal = Signal(
                    id=f"signal_{uuid.uuid4().hex[:8]}",
                    symbol=symbol,
                    timestamp=timestamp,
                    type="sentiment",
                    direction="bullish",
                    strength=combined_bullish,
                    timeframe=strategy.timeframe,
                    source="sentiment_analysis",
                    metadata={
                        "sentiment_threshold": sentiment_threshold,
                        "lookback_period": lookback_period,
                        "bullish_signals": len(bullish_signals),
                        "bearish_signals": len(bearish_signals),
                        "bullish_strength": bullish_strength,
                        "price_trend": price_trend,
                        "combined_strength": combined_bullish
                    }
                )
                signals.append(signal)
            
            elif combined_bearish > sentiment_threshold:
                signal = Signal(
                    id=f"signal_{uuid.uuid4().hex[:8]}",
                    symbol=symbol,
                    timestamp=timestamp,
                    type="sentiment",
                    direction="bearish",
                    strength=combined_bearish,
                    timeframe=strategy.timeframe,
                    source="sentiment_analysis",
                    metadata={
                        "sentiment_threshold": sentiment_threshold,
                        "lookback_period": lookback_period,
                        "bullish_signals": len(bullish_signals),
                        "bearish_signals": len(bearish_signals),
                        "bearish_strength": bearish_strength,
                        "price_trend": price_trend,
                        "combined_strength": combined_bearish
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_event_signals(self, strategy: Strategy, 
                              market_data: List[MarketData],
                              semantic_signals: List[Signal],
                              timestamp: datetime) -> List[Signal]:
        """Generate event-driven signals.
        
        Args:
            strategy: Strategy
            market_data: Market data
            semantic_signals: Semantic signals
            timestamp: Timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        signals = []
        
        # Get parameters
        event_window = next((p.value for p in strategy.parameters if p.name == "event_window"), 5)
        min_confidence = next((p.value for p in strategy.parameters if p.name == "min_confidence"), 0.7)
        
        # Get event types from entry rules
        event_types = strategy.entry_rules.get("event_types", ["EARNINGS_REPORT", "PRODUCT_LAUNCH"])
        
        # Process each symbol
        for symbol in strategy.symbols:
            # Filter semantic signals for symbol
            symbol_semantic = [s for s in semantic_signals if s.symbol == symbol]
            
            # Filter by recency (last event_window days)
            cutoff_time = timestamp - timedelta(days=event_window)
            recent_signals = [s for s in symbol_semantic if s.timestamp >= cutoff_time]
            
            if not recent_signals:
                continue
            
            # Filter by event type and confidence
            event_signals = []
            for s in recent_signals:
                if s.type == "event" and s.strength >= min_confidence:
                    event_type = s.metadata.get("event_type", "")
                    if event_type in event_types:
                        event_signals.append(s)
            
            if not event_signals:
                continue
            
            # Generate signal based on most recent event
            latest_event = max(event_signals, key=lambda x: x.timestamp)
            
            signal = Signal(
                id=f"signal_{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                timestamp=timestamp,
                type="event",
                direction=latest_event.direction,
                strength=latest_event.strength,
                timeframe=strategy.timeframe,
                source="event_analysis",
                metadata={
                    "event_type": latest_event.metadata.get("event_type", ""),
                    "event_timestamp": latest_event.timestamp.isoformat(),
                    "event_window": event_window,
                    "min_confidence": min_confidence,
                    "original_signal_id": latest_event.id
                }
            )
            signals.append(signal)
        
        return signals
    
    async def _generate_ml_signals(self, strategy: Strategy, 
                                market_data: List[MarketData],
                                timestamp: datetime) -> List[Signal]:
        """Generate ML-based signals.
        
        Args:
            strategy: Strategy
            market_data: Market data
            timestamp: Timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        # This is a placeholder for ML-based signal generation
        # In a real implementation, this would use trained ML models
        # to generate predictions and signals
        
        # For now, return empty list
        return []

# Create generator instance
strategy_generator = StrategyGenerator()
