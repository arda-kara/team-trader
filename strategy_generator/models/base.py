"""
Base models for the strategy generator and optimizer.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator

# Enums
class StrategyType(str, Enum):
    """Strategy type enumeration."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    SENTIMENT_BASED = "sentiment_based"
    EVENT_DRIVEN = "event_driven"
    ML_BASED = "ml_based"

class TimeFrame(str, Enum):
    """Timeframe enumeration."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

class PositionSizingMethod(str, Enum):
    """Position sizing method enumeration."""
    FIXED = "fixed"
    PERCENT_EQUITY = "percent_equity"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"

class OptimizationMethod(str, Enum):
    """Optimization method enumeration."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"

class ObjectiveFunction(str, Enum):
    """Objective function enumeration."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TOTAL_RETURN = "total_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"

class MLModelType(str, Enum):
    """Machine learning model type enumeration."""
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class SignalType(str, Enum):
    """Signal type enumeration."""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    EVENT = "event"
    COMBINED = "combined"

class SignalDirection(str, Enum):
    """Signal direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

# Base Models
class MarketData(BaseModel):
    """Market data model."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "timestamp": "2023-01-01T00:00:00Z",
                "open": 150.0,
                "high": 152.5,
                "low": 149.5,
                "close": 152.0,
                "volume": 1000000,
                "timeframe": "1d"
            }
        }

class Signal(BaseModel):
    """Trading signal model."""
    id: str
    symbol: str
    timestamp: datetime
    type: SignalType
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    timeframe: TimeFrame
    source: str
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "signal_123",
                "symbol": "AAPL",
                "timestamp": "2023-01-01T00:00:00Z",
                "type": "technical",
                "direction": "bullish",
                "strength": 0.8,
                "timeframe": "1d",
                "source": "macd_crossover",
                "metadata": {"fast_period": 12, "slow_period": 26}
            }
        }

class StrategyParameter(BaseModel):
    """Strategy parameter model."""
    name: str
    value: Union[int, float, str, bool]
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    choices: Optional[List[Union[int, float, str, bool]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "fast_period",
                "value": 12,
                "min_value": 5,
                "max_value": 20,
                "step": 1
            }
        }

class Strategy(BaseModel):
    """Trading strategy model."""
    id: str
    name: str
    type: StrategyType
    description: str
    parameters: List[StrategyParameter]
    symbols: List[str]
    timeframe: TimeFrame
    position_sizing: PositionSizingMethod
    position_size: float  # Percentage of equity or fixed amount
    entry_rules: Dict[str, Any]
    exit_rules: Dict[str, Any]
    risk_management: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "strategy_123",
                "name": "MACD Crossover Strategy",
                "type": "trend_following",
                "description": "A trend following strategy using MACD crossovers",
                "parameters": [
                    {
                        "name": "fast_period",
                        "value": 12,
                        "min_value": 5,
                        "max_value": 20,
                        "step": 1
                    },
                    {
                        "name": "slow_period",
                        "value": 26,
                        "min_value": 15,
                        "max_value": 40,
                        "step": 1
                    }
                ],
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "timeframe": "1d",
                "position_sizing": "percent_equity",
                "position_size": 0.02,
                "entry_rules": {
                    "condition": "macd_crossover",
                    "direction": "bullish"
                },
                "exit_rules": {
                    "condition": "macd_crossover",
                    "direction": "bearish"
                },
                "risk_management": {
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                }
            }
        }

class Trade(BaseModel):
    """Trade model."""
    id: str
    strategy_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    entry_type: OrderType
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_type: Optional[OrderType] = None
    quantity: float
    side: OrderSide
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    status: OrderStatus
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "trade_123",
                "strategy_id": "strategy_123",
                "symbol": "AAPL",
                "entry_time": "2023-01-01T10:00:00Z",
                "entry_price": 150.0,
                "entry_type": "market",
                "exit_time": "2023-01-03T14:30:00Z",
                "exit_price": 155.0,
                "exit_type": "market",
                "quantity": 10,
                "side": "buy",
                "profit_loss": 50.0,
                "profit_loss_pct": 0.033,
                "status": "filled",
                "metadata": {"signal_id": "signal_123"}
            }
        }

class BacktestResult(BaseModel):
    """Backtest result model."""
    id: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: List[Dict[str, Union[datetime, float]]]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "id": "backtest_123",
                "strategy_id": "strategy_123",
                "start_date": "2022-01-01T00:00:00Z",
                "end_date": "2022-12-31T00:00:00Z",
                "initial_capital": 10000.0,
                "final_capital": 12500.0,
                "total_return": 0.25,
                "annual_return": 0.25,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "max_drawdown": 0.1,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "trades": [],  # Truncated for brevity
                "equity_curve": [],  # Truncated for brevity
                "metrics": {
                    "calmar_ratio": 2.5,
                    "volatility": 0.15,
                    "avg_trade_duration": 3.5
                },
                "parameters": {
                    "fast_period": 12,
                    "slow_period": 26
                }
            }
        }

class OptimizationResult(BaseModel):
    """Optimization result model."""
    id: str
    strategy_id: str
    method: OptimizationMethod
    objective: ObjectiveFunction
    start_date: datetime
    end_date: datetime
    best_parameters: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    execution_time: float  # seconds
    
    class Config:
        schema_extra = {
            "example": {
                "id": "optimization_123",
                "strategy_id": "strategy_123",
                "method": "bayesian",
                "objective": "sharpe_ratio",
                "start_date": "2022-01-01T00:00:00Z",
                "end_date": "2022-12-31T00:00:00Z",
                "best_parameters": {
                    "fast_period": 10,
                    "slow_period": 30
                },
                "best_score": 1.8,
                "all_results": [],  # Truncated for brevity
                "execution_time": 120.5
            }
        }

class MLModel(BaseModel):
    """Machine learning model metadata."""
    id: str
    name: str
    type: MLModelType
    features: List[str]
    target: str
    train_start_date: datetime
    train_end_date: datetime
    validation_score: float
    parameters: Dict[str, Any]
    file_path: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": "model_123",
                "name": "AAPL Price Predictor",
                "type": "gradient_boosting",
                "features": ["price_lag_1", "price_lag_2", "volume", "rsi", "sentiment"],
                "target": "price_direction",
                "train_start_date": "2020-01-01T00:00:00Z",
                "train_end_date": "2022-12-31T00:00:00Z",
                "validation_score": 0.65,
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1
                },
                "file_path": "/models/aapl_price_predictor.pkl"
            }
        }

# Request/Response Models
class StrategyRequest(BaseModel):
    """Request model for strategy creation."""
    name: str
    type: StrategyType
    description: str
    parameters: List[StrategyParameter]
    symbols: List[str]
    timeframe: TimeFrame
    position_sizing: PositionSizingMethod
    position_size: float
    entry_rules: Dict[str, Any]
    exit_rules: Dict[str, Any]
    risk_management: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "MACD Crossover Strategy",
                "type": "trend_following",
                "description": "A trend following strategy using MACD crossovers",
                "parameters": [
                    {
                        "name": "fast_period",
                        "value": 12,
                        "min_value": 5,
                        "max_value": 20,
                        "step": 1
                    },
                    {
                        "name": "slow_period",
                        "value": 26,
                        "min_value": 15,
                        "max_value": 40,
                        "step": 1
                    }
                ],
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "timeframe": "1d",
                "position_sizing": "percent_equity",
                "position_size": 0.02,
                "entry_rules": {
                    "condition": "macd_crossover",
                    "direction": "bullish"
                },
                "exit_rules": {
                    "condition": "macd_crossover",
                    "direction": "bearish"
                },
                "risk_management": {
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                }
            }
        }

class StrategyResponse(BaseModel):
    """Response model for strategy creation."""
    strategy: Strategy
    
    class Config:
        schema_extra = {
            "example": {
                "strategy": {
                    "id": "strategy_123",
                    "name": "MACD Crossover Strategy",
                    "type": "trend_following",
                    "description": "A trend following strategy using MACD crossovers",
                    "parameters": [
                        {
                            "name": "fast_period",
                            "value": 12,
                            "min_value": 5,
                            "max_value": 20,
                            "step": 1
                        },
                        {
                            "name": "slow_period",
                            "value": 26,
                            "min_value": 15,
                            "max_value": 40,
                            "step": 1
                        }
                    ],
                    "symbols": ["AAPL", "MSFT", "GOOGL"],
                    "timeframe": "1d",
                    "position_sizing": "percent_equity",
                    "position_size": 0.02,
                    "entry_rules": {
                        "condition": "macd_crossover",
                        "direction": "bullish"
                    },
                    "exit_rules": {
                        "condition": "macd_crossover",
                        "direction": "bearish"
                    },
                    "risk_management": {
                        "stop_loss": 0.02,
                        "take_profit": 0.04
                    }
                }
            }
        }

class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    parameters: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy_123",
                "start_date": "2022-01-01T00:00:00Z",
                "end_date": "2022-12-31T00:00:00Z",
                "initial_capital": 10000.0,
                "parameters": {
                    "fast_period": 10,
                    "slow_period": 30
                }
            }
        }

class BacktestResponse(BaseModel):
    """Response model for backtesting."""
    result: BacktestResult
    
    class Config:
        schema_extra = {
            "example": {
                "result": {
                    "id": "backtest_123",
                    "strategy_id": "strategy_123",
                    "start_date": "2022-01-01T00:00:00Z",
                    "end_date": "2022-12-31T00:00:00Z",
                    "initial_capital": 10000.0,
                    "final_capital": 12500.0,
                    "total_return": 0.25,
                    "annual_return": 0.25,
                    "sharpe_ratio": 1.5,
                    "sortino_ratio": 2.0,
                    "max_drawdown": 0.1,
                    "win_rate": 0.6,
                    "profit_factor": 1.8,
                    "trades": [],  # Truncated for brevity
                    "equity_curve": [],  # Truncated for brevity
                    "metrics": {
                        "calmar_ratio": 2.5,
                        "volatility": 0.15,
                        "avg_trade_duration": 3.5
                    },
                    "parameters": {
                        "fast_period": 10,
                        "slow_period": 30
                    }
                }
            }
        }

class OptimizationRequest(BaseModel):
    """Request model for strategy optimization."""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    method: OptimizationMethod
    objective: ObjectiveFunction
    parameters: Dict[str, Dict[str, Any]]
    constraints: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy_123",
                "start_date": "2022-01-01T00:00:00Z",
                "end_date": "2022-12-31T00:00:00Z",
                "initial_capital": 10000.0,
                "method": "bayesian",
                "objective": "sharpe_ratio",
                "parameters": {
                    "fast_period": {
                        "min": 5,
                        "max": 20,
                        "step": 1
                    },
                    "slow_period": {
                        "min": 15,
                        "max": 40,
                        "step": 1
                    }
                },
                "constraints": {
                    "max_drawdown": 0.25,
                    "min_trades": 20
                }
            }
        }

class OptimizationResponse(BaseModel):
    """Response model for strategy optimization."""
    result: OptimizationResult
    
    class Config:
        schema_extra = {
            "example": {
                "result": {
                    "id": "optimization_123",
                    "strategy_id": "strategy_123",
                    "method": "bayesian",
                    "objective": "sharpe_ratio",
                    "start_date": "2022-01-01T00:00:00Z",
                    "end_date": "2022-12-31T00:00:00Z",
                    "best_parameters": {
                        "fast_period": 10,
                        "slow_period": 30
                    },
                    "best_score": 1.8,
                    "all_results": [],  # Truncated for brevity
                    "execution_time": 120.5
                }
            }
        }

class SignalGenerationRequest(BaseModel):
    """Request model for signal generation."""
    strategy_id: str
    symbols: List[str]
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy_123",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }

class SignalGenerationResponse(BaseModel):
    """Response model for signal generation."""
    signals: List[Signal]
    
    class Config:
        schema_extra = {
            "example": {
                "signals": [
                    {
                        "id": "signal_123",
                        "symbol": "AAPL",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "type": "technical",
                        "direction": "bullish",
                        "strength": 0.8,
                        "timeframe": "1d",
                        "source": "macd_crossover",
                        "metadata": {"fast_period": 12, "slow_period": 26}
                    }
                ]
            }
        }

class MLModelTrainRequest(BaseModel):
    """Request model for ML model training."""
    name: str
    type: MLModelType
    symbols: List[str]
    features: List[str]
    target: str
    start_date: datetime
    end_date: datetime
    parameters: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "AAPL Price Predictor",
                "type": "gradient_boosting",
                "symbols": ["AAPL"],
                "features": ["price_lag_1", "price_lag_2", "volume", "rsi", "sentiment"],
                "target": "price_direction",
                "start_date": "2020-01-01T00:00:00Z",
                "end_date": "2022-12-31T00:00:00Z",
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1
                }
            }
        }

class MLModelTrainResponse(BaseModel):
    """Response model for ML model training."""
    model: MLModel
    
    class Config:
        schema_extra = {
            "example": {
                "model": {
                    "id": "model_123",
                    "name": "AAPL Price Predictor",
                    "type": "gradient_boosting",
                    "features": ["price_lag_1", "price_lag_2", "volume", "rsi", "sentiment"],
                    "target": "price_direction",
                    "train_start_date": "2020-01-01T00:00:00Z",
                    "train_end_date": "2022-12-31T00:00:00Z",
                    "validation_score": 0.65,
                    "parameters": {
                        "n_estimators": 100,
                        "max_depth": 5,
                        "learning_rate": 0.1
                    },
                    "file_path": "/models/aapl_price_predictor.pkl"
                }
            }
        }

class MLModelPredictRequest(BaseModel):
    """Request model for ML model prediction."""
    model_id: str
    data: Dict[str, List[Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "model_123",
                "data": {
                    "price_lag_1": [150.0, 151.0, 149.5],
                    "price_lag_2": [148.0, 150.0, 151.0],
                    "volume": [1000000, 1200000, 900000],
                    "rsi": [65.0, 68.0, 62.0],
                    "sentiment": [0.2, 0.5, 0.1]
                }
            }
        }

class MLModelPredictResponse(BaseModel):
    """Response model for ML model prediction."""
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [1, 1, 0],
                "probabilities": [
                    [0.3, 0.7],
                    [0.2, 0.8],
                    [0.6, 0.4]
                ]
            }
        }
