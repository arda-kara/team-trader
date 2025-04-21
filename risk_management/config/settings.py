"""
Configuration settings for the risk management module.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseSettings, Field

class RiskLimitsSettings(BaseSettings):
    """Settings for risk limits."""
    max_position_size_pct: float = 0.05  # Maximum position size as percentage of portfolio
    max_strategy_allocation_pct: float = 0.25  # Maximum allocation to a single strategy
    max_sector_exposure_pct: float = 0.30  # Maximum exposure to a single sector
    max_asset_correlation: float = 0.7  # Maximum correlation between assets
    max_drawdown_pct: float = 0.15  # Maximum allowed drawdown
    max_daily_loss_pct: float = 0.03  # Maximum daily loss as percentage of portfolio
    max_var_pct: float = 0.05  # Maximum Value at Risk (95%, 1-day)
    max_leverage: float = 2.0  # Maximum leverage
    min_liquidity_ratio: float = 0.25  # Minimum liquidity ratio
    position_sizing_volatility_factor: float = 1.5  # Factor for volatility-based position sizing
    max_concentration_pct: Dict[str, float] = {
        "single_stock": 0.05,
        "sector": 0.30,
        "asset_class": 0.60,
        "strategy": 0.25
    }

class PortfolioOptimizationSettings(BaseSettings):
    """Settings for portfolio optimization."""
    optimization_method: str = "mean_variance"  # mean_variance, risk_parity, min_variance, max_sharpe
    risk_free_rate: float = 0.02  # Risk-free rate for Sharpe ratio calculation
    target_return: Optional[float] = None  # Target return for mean-variance optimization
    target_risk: Optional[float] = None  # Target risk for mean-variance optimization
    rebalance_threshold_pct: float = 0.05  # Threshold for rebalancing
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    optimization_lookback_days: int = 252  # Lookback period for optimization
    min_weight: float = 0.0  # Minimum weight for any asset
    max_weight: float = 0.25  # Maximum weight for any asset
    risk_aversion: float = 3.0  # Risk aversion parameter
    regularization_factor: float = 0.1  # Regularization factor for optimization
    use_constraints: bool = True  # Whether to use constraints in optimization
    constraints: Dict[str, Any] = {
        "sector_neutral": False,
        "long_only": True,
        "fully_invested": True
    }

class RiskModelsSettings(BaseSettings):
    """Settings for risk models."""
    var_confidence_level: float = 0.95  # Confidence level for VaR calculation
    var_lookback_days: int = 252  # Lookback period for VaR calculation
    var_method: str = "historical"  # historical, parametric, monte_carlo
    stress_test_scenarios: List[str] = ["2008_crisis", "covid_crash", "rate_hike", "custom"]
    volatility_lookback_days: int = 63  # Lookback period for volatility calculation
    volatility_method: str = "ewma"  # simple, ewma, garch
    volatility_lambda: float = 0.94  # Lambda parameter for EWMA volatility
    correlation_lookback_days: int = 126  # Lookback period for correlation calculation
    correlation_method: str = "pearson"  # pearson, spearman
    risk_factor_model: str = "pca"  # pca, fundamental
    num_risk_factors: int = 5  # Number of risk factors for PCA model
    custom_stress_scenarios: Dict[str, Dict[str, float]] = {
        "custom": {
            "equity_market": -0.15,
            "interest_rates": 0.01,
            "credit_spreads": 0.02,
            "volatility": 0.5
        }
    }

class ExposureManagementSettings(BaseSettings):
    """Settings for exposure management."""
    monitor_sectors: bool = True
    monitor_asset_classes: bool = True
    monitor_factors: bool = True
    monitor_geographies: bool = True
    monitor_currencies: bool = True
    sector_classification: str = "gics"  # gics, sic, custom
    factor_model: str = "barra"  # barra, fama_french, custom
    max_net_exposure: float = 1.0  # Maximum net exposure
    max_gross_exposure: float = 2.0  # Maximum gross exposure
    target_beta: float = 0.0  # Target portfolio beta
    beta_tolerance: float = 0.2  # Tolerance around target beta
    hedge_threshold_pct: float = 0.1  # Threshold for hedging
    auto_hedge: bool = False  # Whether to automatically hedge exposures
    exposure_limits: Dict[str, Dict[str, float]] = {
        "sectors": {
            "technology": 0.3,
            "financials": 0.3,
            "healthcare": 0.3,
            "consumer_discretionary": 0.3,
            "consumer_staples": 0.3,
            "industrials": 0.3,
            "energy": 0.2,
            "materials": 0.2,
            "utilities": 0.2,
            "real_estate": 0.2,
            "communication_services": 0.3
        },
        "factors": {
            "momentum": 0.3,
            "value": 0.3,
            "size": 0.3,
            "quality": 0.3,
            "volatility": 0.3
        },
        "asset_classes": {
            "equity": 1.0,
            "fixed_income": 0.5,
            "commodities": 0.3,
            "currencies": 0.3,
            "crypto": 0.2
        }
    }

class DrawdownProtectionSettings(BaseSettings):
    """Settings for drawdown protection."""
    enabled: bool = True
    max_drawdown_pct: float = 0.15  # Maximum allowed drawdown
    drawdown_calculation_method: str = "peak_to_trough"  # peak_to_trough, underwater
    action_threshold_pct: float = 0.1  # Threshold for taking action
    recovery_threshold_pct: float = 0.05  # Threshold for recovery
    reduction_method: str = "proportional"  # proportional, volatility_based, equal
    reduction_factor: float = 0.5  # Factor for reducing exposure
    stop_trading_threshold_pct: float = 0.15  # Threshold for stopping trading
    restart_trading_threshold_pct: float = 0.1  # Threshold for restarting trading
    use_time_based_recovery: bool = False  # Whether to use time-based recovery
    recovery_time_days: int = 20  # Days for time-based recovery

class DatabaseSettings(BaseSettings):
    """Settings for risk database."""
    connection_string: str = "sqlite:///risk_management.db"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    risk_history_days: int = 365  # Days to keep risk history

class APISettings(BaseSettings):
    """Settings for risk API."""
    host: str = "0.0.0.0"
    port: int = 8003
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
    file_path: str = "/var/log/trading_pipeline/risk_management.log"
    console_output: bool = True
    log_risk_checks: bool = True
    log_portfolio_changes: bool = True
    log_exposure_changes: bool = True
    log_drawdown_events: bool = True

class Settings(BaseSettings):
    """Main settings for risk management."""
    environment: str = Field("development", env="TRADING_ENV")
    risk_limits: RiskLimitsSettings = RiskLimitsSettings()
    portfolio_optimization: PortfolioOptimizationSettings = PortfolioOptimizationSettings()
    risk_models: RiskModelsSettings = RiskModelsSettings()
    exposure_management: ExposureManagementSettings = ExposureManagementSettings()
    drawdown_protection: DrawdownProtectionSettings = DrawdownProtectionSettings()
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    redis_url: str = "redis://localhost:6379/0"
    execution_api_url: str = "http://localhost:8002/api"
    strategy_api_url: str = "http://localhost:8001/api"
    
    class Config:
        env_prefix = "RISK_"
        env_nested_delimiter = "__"

# Create settings instance
settings = Settings()
