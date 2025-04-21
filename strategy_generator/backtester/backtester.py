"""
Backtester for evaluating trading strategies.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..models.base import (
    Strategy, BacktestResult, Trade, MarketData, Signal,
    OrderType, OrderSide, OrderStatus, TimeFrame
)
from ..generators.data_client import data_client
from ..generators.strategy_generator import strategy_generator
from ..generators.redis_client import backtest_cache

logger = logging.getLogger(__name__)

class Backtester:
    """Backtester for evaluating trading strategies."""
    
    def __init__(self):
        """Initialize backtester."""
        self.commission_rate = settings.backtest.commission_rate
        self.slippage_rate = settings.backtest.slippage_rate
        self.save_trades = settings.backtest.save_trades
        self.plot_results = settings.backtest.plot_results
    
    def _generate_backtest_id(self) -> str:
        """Generate unique backtest ID.
        
        Returns:
            str: Backtest ID
        """
        return f"backtest_{uuid.uuid4().hex[:8]}"
    
    async def run_backtest(self, strategy_id: str, start_date: datetime, end_date: datetime,
                         initial_capital: float = 10000.0, 
                         parameters: Optional[Dict[str, Any]] = None) -> BacktestResult:
        """Run backtest for strategy.
        
        Args:
            strategy_id: Strategy ID
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            parameters: Optional parameter overrides
            
        Returns:
            BacktestResult: Backtest result
        """
        # Get strategy
        strategy = strategy_generator.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        # Apply parameter overrides if provided
        if parameters:
            for param in strategy.parameters:
                if param.name in parameters:
                    param.value = parameters[param.name]
        
        # Get market data
        market_data = await data_client.get_market_data(
            strategy.symbols,
            strategy.timeframe,
            start_date,
            end_date
        )
        
        # Get semantic signals if needed
        semantic_signals = []
        if strategy.type in ["sentiment_based", "event_driven"]:
            semantic_signals = await data_client.get_semantic_signals(
                strategy.symbols,
                start_date,
                end_date
            )
        
        # Run backtest
        result = await self._run_backtest_simulation(
            strategy, market_data, semantic_signals, 
            start_date, end_date, initial_capital
        )
        
        # Cache result
        backtest_cache.set(f"backtest:{result.id}", result.dict())
        
        return result
    
    async def _run_backtest_simulation(self, strategy: Strategy, market_data: List[MarketData],
                                     semantic_signals: List[Signal], start_date: datetime,
                                     end_date: datetime, initial_capital: float) -> BacktestResult:
        """Run backtest simulation.
        
        Args:
            strategy: Strategy
            market_data: Market data
            semantic_signals: Semantic signals
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            
        Returns:
            BacktestResult: Backtest result
        """
        # Generate backtest ID
        backtest_id = self._generate_backtest_id()
        
        # Initialize portfolio
        portfolio = {
            "cash": initial_capital,
            "positions": {symbol: 0 for symbol in strategy.symbols},
            "equity": initial_capital,
            "equity_curve": []
        }
        
        # Initialize trades list
        trades = []
        
        # Get unique dates from market data
        dates = sorted(set(data.timestamp.date() for data in market_data))
        
        # Filter dates within range
        dates = [d for d in dates if start_date.date() <= d <= end_date.date()]
        
        # Run simulation for each date
        for date in dates:
            # Get date's timestamp (end of day)
            timestamp = datetime.combine(date, datetime.max.time())
            
            # Update portfolio value
            portfolio = self._update_portfolio_value(portfolio, market_data, timestamp)
            
            # Generate signals for this date
            signals = await self._generate_signals_for_date(strategy, market_data, semantic_signals, timestamp)
            
            # Process signals
            new_trades = self._process_signals(signals, strategy, portfolio, market_data, timestamp)
            trades.extend(new_trades)
            
            # Update portfolio after trades
            portfolio = self._update_portfolio_value(portfolio, market_data, timestamp)
            
            # Record equity curve
            portfolio["equity_curve"].append({
                "timestamp": timestamp,
                "equity": portfolio["equity"]
            })
        
        # Close all open positions at the end
        final_timestamp = datetime.combine(end_date.date(), datetime.max.time())
        close_trades = self._close_all_positions(portfolio, trades, market_data, final_timestamp)
        trades.extend(close_trades)
        
        # Update final portfolio value
        portfolio = self._update_portfolio_value(portfolio, market_data, final_timestamp)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio, trades, initial_capital, start_date, end_date)
        
        # Create backtest result
        result = BacktestResult(
            id=backtest_id,
            strategy_id=strategy.id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=portfolio["equity"],
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            trades=trades if self.save_trades else [],
            equity_curve=portfolio["equity_curve"],
            metrics=metrics,
            parameters={p.name: p.value for p in strategy.parameters}
        )
        
        return result
    
    def _update_portfolio_value(self, portfolio: Dict[str, Any], market_data: List[MarketData],
                              timestamp: datetime) -> Dict[str, Any]:
        """Update portfolio value based on current market prices.
        
        Args:
            portfolio: Portfolio state
            market_data: Market data
            timestamp: Current timestamp
            
        Returns:
            Dict[str, Any]: Updated portfolio
        """
        # Get latest prices for each symbol
        positions_value = 0.0
        for symbol, quantity in portfolio["positions"].items():
            if quantity == 0:
                continue
            
            # Find latest price before or at timestamp
            symbol_data = [d for d in market_data if d.symbol == symbol and d.timestamp <= timestamp]
            if not symbol_data:
                continue
            
            latest_price = max(symbol_data, key=lambda x: x.timestamp).close
            position_value = quantity * latest_price
            positions_value += position_value
        
        # Update equity
        portfolio["equity"] = portfolio["cash"] + positions_value
        
        return portfolio
    
    async def _generate_signals_for_date(self, strategy: Strategy, market_data: List[MarketData],
                                       semantic_signals: List[Signal], timestamp: datetime) -> List[Signal]:
        """Generate signals for a specific date.
        
        Args:
            strategy: Strategy
            market_data: Market data
            semantic_signals: Semantic signals
            timestamp: Current timestamp
            
        Returns:
            List[Signal]: Generated signals
        """
        # Filter market data up to timestamp
        filtered_market_data = [d for d in market_data if d.timestamp <= timestamp]
        
        # Filter semantic signals up to timestamp
        filtered_semantic_signals = [s for s in semantic_signals if s.timestamp <= timestamp]
        
        # Generate signals using strategy generator
        date_signals = await strategy_generator.generate_signals(strategy, timestamp)
        
        # Add semantic signals for this date
        date_semantic_signals = [s for s in filtered_semantic_signals 
                               if s.timestamp.date() == timestamp.date()]
        
        # Combine signals
        all_signals = date_signals + date_semantic_signals
        
        return all_signals
    
    def _process_signals(self, signals: List[Signal], strategy: Strategy, portfolio: Dict[str, Any],
                       market_data: List[MarketData], timestamp: datetime) -> List[Trade]:
        """Process signals and execute trades.
        
        Args:
            signals: Signals to process
            strategy: Strategy
            portfolio: Portfolio state
            market_data: Market data
            timestamp: Current timestamp
            
        Returns:
            List[Trade]: Executed trades
        """
        trades = []
        
        # Process each signal
        for signal in signals:
            symbol = signal.symbol
            
            # Skip if symbol not in strategy
            if symbol not in strategy.symbols:
                continue
            
            # Get current position
            current_position = portfolio["positions"].get(symbol, 0)
            
            # Get latest price
            symbol_data = [d for d in market_data if d.symbol == symbol and d.timestamp <= timestamp]
            if not symbol_data:
                continue
            
            latest_data = max(symbol_data, key=lambda x: x.timestamp)
            latest_price = latest_data.close
            
            # Check entry conditions
            if signal.direction == "bullish" and current_position <= 0:
                # Calculate position size
                position_size = self._calculate_position_size(
                    strategy, portfolio, symbol, latest_price
                )
                
                if position_size > 0:
                    # Execute buy trade
                    trade = self._execute_trade(
                        strategy.id, symbol, timestamp, latest_price,
                        position_size, "buy", signal.id
                    )
                    
                    # Update portfolio
                    cost = position_size * latest_price * (1 + self.commission_rate + self.slippage_rate)
                    portfolio["cash"] -= cost
                    portfolio["positions"][symbol] += position_size
                    
                    trades.append(trade)
            
            # Check exit conditions
            elif signal.direction == "bearish" and current_position > 0:
                # Execute sell trade
                trade = self._execute_trade(
                    strategy.id, symbol, timestamp, latest_price,
                    current_position, "sell", signal.id
                )
                
                # Update portfolio
                proceeds = current_position * latest_price * (1 - self.commission_rate - self.slippage_rate)
                portfolio["cash"] += proceeds
                portfolio["positions"][symbol] = 0
                
                trades.append(trade)
        
        return trades
    
    def _calculate_position_size(self, strategy: Strategy, portfolio: Dict[str, Any],
                               symbol: str, price: float) -> float:
        """Calculate position size based on strategy and portfolio.
        
        Args:
            strategy: Strategy
            portfolio: Portfolio state
            symbol: Symbol
            price: Current price
            
        Returns:
            float: Position size (quantity)
        """
        # Get position sizing method
        position_sizing = strategy.position_sizing
        position_size_pct = strategy.position_size
        
        # Calculate position size
        if position_sizing == "fixed":
            # Fixed dollar amount
            return position_size_pct / price
        
        elif position_sizing == "percent_equity":
            # Percentage of equity
            equity = portfolio["equity"]
            dollar_amount = equity * position_size_pct
            return dollar_amount / price
        
        elif position_sizing == "volatility_adjusted":
            # Volatility-adjusted position sizing
            # This is a simplified implementation
            equity = portfolio["equity"]
            dollar_amount = equity * position_size_pct
            return dollar_amount / price
        
        else:
            # Default to percent of equity
            equity = portfolio["equity"]
            dollar_amount = equity * position_size_pct
            return dollar_amount / price
    
    def _execute_trade(self, strategy_id: str, symbol: str, timestamp: datetime,
                     price: float, quantity: float, side: str, signal_id: Optional[str] = None) -> Trade:
        """Execute a trade.
        
        Args:
            strategy_id: Strategy ID
            symbol: Symbol
            timestamp: Timestamp
            price: Execution price
            quantity: Quantity
            side: Order side (buy/sell)
            signal_id: Optional signal ID
            
        Returns:
            Trade: Executed trade
        """
        # Generate trade ID
        trade_id = f"trade_{uuid.uuid4().hex[:8]}"
        
        # Create trade
        trade = Trade(
            id=trade_id,
            strategy_id=strategy_id,
            symbol=symbol,
            entry_time=timestamp,
            entry_price=price,
            entry_type="market",
            quantity=quantity,
            side=side,
            status="filled",
            metadata={"signal_id": signal_id} if signal_id else {}
        )
        
        return trade
    
    def _close_all_positions(self, portfolio: Dict[str, Any], trades: List[Trade],
                           market_data: List[MarketData], timestamp: datetime) -> List[Trade]:
        """Close all open positions.
        
        Args:
            portfolio: Portfolio state
            trades: Existing trades
            market_data: Market data
            timestamp: Current timestamp
            
        Returns:
            List[Trade]: Closing trades
        """
        closing_trades = []
        
        # Find open positions
        for symbol, quantity in portfolio["positions"].items():
            if quantity <= 0:
                continue
            
            # Get latest price
            symbol_data = [d for d in market_data if d.symbol == symbol and d.timestamp <= timestamp]
            if not symbol_data:
                continue
            
            latest_data = max(symbol_data, key=lambda x: x.timestamp)
            latest_price = latest_data.close
            
            # Find the strategy ID from previous trades
            strategy_id = next((t.strategy_id for t in trades 
                              if t.symbol == symbol and t.side == "buy"), "unknown")
            
            # Execute sell trade
            trade = self._execute_trade(
                strategy_id, symbol, timestamp, latest_price,
                quantity, "sell"
            )
            
            # Update portfolio
            proceeds = quantity * latest_price * (1 - self.commission_rate - self.slippage_rate)
            portfolio["cash"] += proceeds
            portfolio["positions"][symbol] = 0
            
            closing_trades.append(trade)
        
        return closing_trades
    
    def _calculate_performance_metrics(self, portfolio: Dict[str, Any], trades: List[Trade],
                                     initial_capital: float, start_date: datetime,
                                     end_date: datetime) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            portfolio: Portfolio state
            trades: Executed trades
            initial_capital: Initial capital
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Extract equity curve
        equity_curve = pd.DataFrame(portfolio["equity_curve"])
        
        if len(equity_curve) < 2:
            # Not enough data for metrics
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "calmar_ratio": 0.0,
                "volatility": 0.0,
                "avg_trade_duration": 0.0
            }
        
        # Calculate returns
        equity_curve["return"] = equity_curve["equity"].pct_change()
        
        # Calculate total return
        final_equity = portfolio["equity"]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Calculate annual return
        days = (end_date - start_date).days
        annual_return = (1 + total_return) ** (365 / max(1, days)) - 1
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        daily_returns = equity_curve["return"].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0.0
        
        # Calculate Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / negative_returns.std()
        else:
            sortino_ratio = 0.0
        
        # Calculate maximum drawdown
        equity_curve["cummax"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["cummax"]) / equity_curve["cummax"]
        max_drawdown = abs(equity_curve["drawdown"].min())
        
        # Calculate win rate and profit factor
        closed_trades = [t for t in trades if t.exit_time is not None]
        winning_trades = [t for t in closed_trades if t.profit_loss_pct is not None and t.profit_loss_pct > 0]
        
        win_rate = len(winning_trades) / max(1, len(closed_trades))
        
        gross_profit = sum(t.profit_loss for t in winning_trades if t.profit_loss is not None)
        losing_trades = [t for t in closed_trades if t.profit_loss_pct is not None and t.profit_loss_pct <= 0]
        gross_loss = abs(sum(t.profit_loss for t in losing_trades if t.profit_loss is not None))
        
        profit_factor = gross_profit / max(0.01, gross_loss)
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / max(0.01, max_drawdown)
        
        # Calculate volatility
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Calculate average trade duration
        trade_durations = []
        for trade in closed_trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 86400  # days
                trade_durations.append(duration)
        
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0.0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar_ratio,
            "volatility": volatility,
            "avg_trade_duration": avg_trade_duration
        }

# Create backtester instance
backtester = Backtester()
