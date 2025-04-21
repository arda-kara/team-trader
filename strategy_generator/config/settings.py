"""
Configuration settings for the Strategy Generator and Optimizer.
"""

import os
from typing import Dict, List, Optional, Union
from pydantic import BaseSettings, Field, validator

class BacktestSettings(BaseSettings):
    """Backtesting settings."""
    # Time period settings
    default_start_date: str = Field("2020-01-01", env="BACKTEST_START_DATE")
    default_end_date: str = Field("2023-12-31", env="BACKTEST_END_DATE")
    
    # Data settings
    default_timeframe: str = Field("1d", env="BACKTEST_TIMEFRAME")
    available_timeframes: List[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    # Execution settings
    commission_rate: float = Field(0.001, env="BACKTEST_COMMISSION_RATE")  # 0.1%
    slippage_rate: float = Field(0.0005, env="BACKTEST_SLIPPAGE_RATE")  # 0.05%
    
    # Output settings
    save_trades: bool = Field(True, env="BACKTEST_SAVE_TRADES")
    plot_results: bool = Field(True, env="BACKTEST_PLOT_RESULTS")
    
    # Performance metrics
    metrics: List[str] = [
        "total_return", "annual_return", "sharpe_ratio", "sortino_ratio", 
        "max_drawdown", "win_rate", "profit_factor", "calmar_ratio"
    ]

class StrategySettings(BaseSettings):
    """Strategy generation settings."""
    # Strategy types
    available_types: List[str] = [
        "trend_following", "mean_reversion", "breakout", "momentum",
        "statistical_arbitrage", "sentiment_based", "event_driven", "ml_based"
    ]
    
    # Default parameters for each strategy type
    default_params: Dict[str, Dict[str, Union[int, float, str, bool]]] = {
        "trend_following": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "atr_period": 14,
            "atr_multiplier": 2.0
        },
        "mean_reversion": {
            "lookback_period": 20,
            "entry_zscore": 2.0,
            "exit_zscore": 0.0,
            "stop_loss_std": 3.0
        },
        "breakout": {
            "channel_period": 20,
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "min_volume_multiplier": 1.5
        },
        "momentum": {
            "momentum_period": 14,
            "signal_period": 3,
            "threshold": 0.0
        },
        "sentiment_based": {
            "sentiment_threshold": 0.5,
            "lookback_period": 3,
            "signal_weight": 0.7,
            "price_weight": 0.3
        },
        "event_driven": {
            "event_window": 5,
            "pre_event_window": 2,
            "post_event_window": 3,
            "min_confidence": 0.7
        },
        "ml_based": {
            "features": ["price", "volume", "sentiment", "volatility"],
            "lookback_period": 20,
            "prediction_horizon": 5,
            "train_test_split": 0.8
        }
    }
    
    # Signal integration weights
    signal_weights: Dict[str, float] = {
        "price": 0.4,
        "volume": 0.1,
        "technical": 0.2,
        "sentiment": 0.15,
        "event": 0.15
    }
    
    # Position sizing methods
    position_sizing_methods: List[str] = [
        "fixed", "percent_equity", "volatility_adjusted", "kelly", "optimal_f"
    ]
    
    # Default position sizing method
    default_position_sizing: str = Field("percent_equity", env="DEFAULT_POSITION_SIZING")
    
    # Default position size (as percentage of equity)
    default_position_size: float = Field(0.02, env="DEFAULT_POSITION_SIZE")  # 2%

class OptimizerSettings(BaseSettings):
    """Strategy optimization settings."""
    # Optimization methods
    available_methods: List[str] = [
        "grid_search", "random_search", "bayesian", "genetic", "particle_swarm"
    ]
    
    # Default optimization method
    default_method: str = Field("bayesian", env="DEFAULT_OPTIMIZATION_METHOD")
    
    # Optimization parameters
    max_trials: int = Field(100, env="OPTIMIZATION_MAX_TRIALS")
    timeout: int = Field(3600, env="OPTIMIZATION_TIMEOUT")  # seconds
    n_jobs: int = Field(-1, env="OPTIMIZATION_N_JOBS")  # -1 means use all cores
    
    # Cross-validation settings
    cv_method: str = Field("time_series", env="CV_METHOD")
    cv_folds: int = Field(5, env="CV_FOLDS")
    
    # Objective function
    default_objective: str = Field("sharpe_ratio", env="DEFAULT_OBJECTIVE")
    available_objectives: List[str] = [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "total_return",
        "risk_adjusted_return", "max_drawdown", "win_rate", "profit_factor"
    ]
    
    # Constraints
    max_drawdown_constraint: float = Field(0.25, env="MAX_DRAWDOWN_CONSTRAINT")
    min_trades_constraint: int = Field(20, env="MIN_TRADES_CONSTRAINT")
    min_win_rate_constraint: float = Field(0.4, env="MIN_WIN_RATE_CONSTRAINT")

class MLSettings(BaseSettings):
    """Machine learning settings."""
    # Model types
    available_models: List[str] = [
        "linear", "random_forest", "gradient_boosting", "neural_network", 
        "lstm", "transformer", "ensemble"
    ]
    
    # Default model
    default_model: str = Field("gradient_boosting", env="DEFAULT_ML_MODEL")
    
    # Feature engineering
    default_features: List[str] = [
        "price", "volume", "volatility", "rsi", "macd", "bollinger",
        "sentiment_score", "event_signals", "economic_indicators"
    ]
    
    # Training settings
    train_test_split: float = Field(0.8, env="ML_TRAIN_TEST_SPLIT")
    validation_size: float = Field(0.2, env="ML_VALIDATION_SIZE")
    
    # Hyperparameter tuning
    hp_tuning_method: str = Field("bayesian", env="ML_HP_TUNING_METHOD")
    hp_tuning_trials: int = Field(50, env="ML_HP_TUNING_TRIALS")
    
    # Model evaluation metrics
    evaluation_metrics: List[str] = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "mse", "mae"
    ]
    
    # Model persistence
    save_models: bool = Field(True, env="ML_SAVE_MODELS")
    model_dir: str = Field("models", env="ML_MODEL_DIR")

class APISettings(BaseSettings):
    """API settings."""
    # API host and port
    host: str = Field("0.0.0.0", env="STRATEGY_API_HOST")
    port: int = Field(8002, env="STRATEGY_API_PORT")
    
    # API rate limiting
    rate_limit_requests: int = Field(100, env="API_RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="API_RATE_LIMIT_PERIOD")  # seconds

class Settings(BaseSettings):
    """Main settings class for Strategy Generator and Optimizer."""
    # Environment settings
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    
    # Component settings
    backtest: BacktestSettings = BacktestSettings()
    strategy: StrategySettings = StrategySettings()
    optimizer: OptimizerSettings = OptimizerSettings()
    ml: MLSettings = MLSettings()
    api: APISettings = APISettings()
    
    # Redis settings (for queue and cache)
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(2, env="REDIS_DB")  # Use different DB than other components
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    # Data access settings
    data_api_url: str = Field("http://localhost:8000/api/data", env="DATA_API_URL")
    semantic_api_url: str = Field("http://localhost:8001/api/semantic", env="SEMANTIC_API_URL")
    
    # Processing settings
    batch_size: int = Field(10, env="PROCESSING_BATCH_SIZE")
    processing_interval: int = Field(60, env="PROCESSING_INTERVAL")  # seconds
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()
