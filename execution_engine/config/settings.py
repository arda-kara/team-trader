"""
Configuration settings for the execution engine.
"""

from typing import Dict, List, Any
from pydantic import BaseSettings, Field

class BrokerSettings(BaseSettings):
    """Settings for broker connections."""
    alpaca: Dict[str, Any] = {
        "api_key": "YOUR_ALPACA_API_KEY",
        "api_secret": "YOUR_ALPACA_API_SECRET",
        "base_url": "https://paper-api.alpaca.markets",  # Use paper trading by default
        "data_url": "https://data.alpaca.markets",
        "use_sandbox": True
    }
    
    interactive_brokers: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 7497,  # TWS paper trading port
        "client_id": 1,
        "timeout": 20,
        "use_sandbox": True
    }
    
    binance: Dict[str, Any] = {
        "api_key": "YOUR_BINANCE_API_KEY",
        "api_secret": "YOUR_BINANCE_API_SECRET",
        "testnet": True  # Use testnet by default
    }
    
    default_broker: str = "alpaca"
    enabled_brokers: List[str] = ["alpaca"]

class OrderSettings(BaseSettings):
    """Settings for order management."""
    default_order_type: str = "market"
    default_time_in_force: str = "day"
    max_order_size_percent: float = 0.1  # Maximum order size as percentage of portfolio
    min_order_size_usd: float = 100.0  # Minimum order size in USD
    max_order_size_usd: float = 100000.0  # Maximum order size in USD
    max_orders_per_second: int = 5  # Rate limiting
    retry_attempts: int = 3  # Number of retry attempts for failed orders
    retry_delay_seconds: int = 2  # Delay between retry attempts
    cancel_after_seconds: int = 300  # Auto-cancel orders after this time (5 minutes)

class ExecutionSettings(BaseSettings):
    """Settings for execution algorithms."""
    default_algorithm: str = "market"
    available_algorithms: List[str] = ["market", "limit", "twap", "vwap", "is"]
    twap_settings: Dict[str, Any] = {
        "interval_seconds": 300,  # 5 minutes
        "max_participation_rate": 0.3
    }
    vwap_settings: Dict[str, Any] = {
        "interval_seconds": 300,  # 5 minutes
        "max_participation_rate": 0.3,
        "historical_volume_days": 20
    }
    implementation_shortfall_settings: Dict[str, Any] = {
        "urgency": "medium",  # low, medium, high
        "max_participation_rate": 0.3
    }
    smart_routing: bool = True
    market_hours_only: bool = True

class PositionSettings(BaseSettings):
    """Settings for position management."""
    max_positions: int = 20  # Maximum number of concurrent positions
    position_size_limits: Dict[str, Dict[str, float]] = {
        "default": {
            "max_notional_usd": 100000.0,
            "max_portfolio_pct": 0.05
        },
        "high_volatility": {
            "max_notional_usd": 50000.0,
            "max_portfolio_pct": 0.03
        }
    }
    auto_hedge_enabled: bool = False
    auto_hedge_threshold: float = 0.1  # Auto-hedge when exposure exceeds this threshold
    rebalance_frequency: str = "daily"  # never, daily, weekly

class DatabaseSettings(BaseSettings):
    """Settings for execution database."""
    connection_string: str = "sqlite:///execution_engine.db"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    order_history_days: int = 90  # Days to keep order history
    execution_history_days: int = 90  # Days to keep execution history

class APISettings(BaseSettings):
    """Settings for execution engine API."""
    host: str = "0.0.0.0"
    port: int = 8002
    debug: bool = False
    reload: bool = False
    workers: int = 1
    timeout_keep_alive: int = 5
    access_log: bool = True
    cors_origins: List[str] = ["*"]
    api_keys: List[str] = ["development_api_key"]

class LoggingSettings(BaseSettings):
    """Settings for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "/var/log/trading_pipeline/execution_engine.log"
    console_output: bool = True
    log_orders: bool = True
    log_executions: bool = True
    log_positions: bool = True
    log_performance: bool = True

class SimulationSettings(BaseSettings):
    """Settings for simulation mode."""
    enabled: bool = True
    latency_ms: int = 100  # Simulated latency in milliseconds
    slippage_bps: int = 5  # Simulated slippage in basis points
    commission_bps: int = 1  # Simulated commission in basis points
    rejection_probability: float = 0.01  # Probability of order rejection
    partial_fill_probability: float = 0.2  # Probability of partial fill
    
class Settings(BaseSettings):
    """Main settings for execution engine."""
    environment: str = Field("development", env="TRADING_ENV")
    brokers: BrokerSettings = BrokerSettings()
    orders: OrderSettings = OrderSettings()
    execution: ExecutionSettings = ExecutionSettings()
    positions: PositionSettings = PositionSettings()
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    simulation: SimulationSettings = SimulationSettings()
    redis_url: str = "redis://localhost:6379/0"
    strategy_api_url: str = "http://localhost:8001/api"
    risk_api_url: str = "http://localhost:8003/api"
    
    class Config:
        env_prefix = "EXECUTION_"
        env_nested_delimiter = "__"

# Create settings instance
settings = Settings()
