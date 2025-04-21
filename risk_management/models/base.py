"""
Base models for the risk management module.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import uuid

# Enums
class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(str, Enum):
    """Risk type enumeration."""
    MARKET = "market"
    CREDIT = "credit"
    LIQUIDITY = "liquidity"
    OPERATIONAL = "operational"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    EXPOSURE = "exposure"
    LEVERAGE = "leverage"

class RiskAction(str, Enum):
    """Risk action enumeration."""
    ALLOW = "allow"
    WARN = "warn"
    REDUCE = "reduce"
    BLOCK = "block"

class OptimizationMethod(str, Enum):
    """Optimization method enumeration."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    EQUAL_WEIGHT = "equal_weight"
    CUSTOM = "custom"

class RebalanceFrequency(str, Enum):
    """Rebalance frequency enumeration."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"

class VaRMethod(str, Enum):
    """VaR method enumeration."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"

class VolatilityMethod(str, Enum):
    """Volatility method enumeration."""
    SIMPLE = "simple"
    EWMA = "ewma"
    GARCH = "garch"

class CorrelationMethod(str, Enum):
    """Correlation method enumeration."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"

class SectorClassification(str, Enum):
    """Sector classification enumeration."""
    GICS = "gics"
    SIC = "sic"
    CUSTOM = "custom"

class FactorModel(str, Enum):
    """Factor model enumeration."""
    BARRA = "barra"
    FAMA_FRENCH = "fama_french"
    PCA = "pca"
    CUSTOM = "custom"

class DrawdownMethod(str, Enum):
    """Drawdown calculation method enumeration."""
    PEAK_TO_TROUGH = "peak_to_trough"
    UNDERWATER = "underwater"

class ReductionMethod(str, Enum):
    """Reduction method enumeration."""
    PROPORTIONAL = "proportional"
    VOLATILITY_BASED = "volatility_based"
    EQUAL = "equal"

# Base Models
class RiskLimit(BaseModel):
    """Risk limit model."""
    id: str = Field(default_factory=lambda: f"limit_{uuid.uuid4().hex[:8]}")
    name: str
    description: Optional[str] = None
    risk_type: RiskType
    threshold: float
    warning_threshold: Optional[float] = None
    action: RiskAction = RiskAction.WARN
    scope: str  # "portfolio", "strategy", "asset", "sector", etc.
    scope_id: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "limit_12345678",
                "name": "Max Position Size",
                "description": "Maximum position size as percentage of portfolio",
                "risk_type": "concentration",
                "threshold": 0.05,
                "warning_threshold": 0.04,
                "action": "warn",
                "scope": "portfolio",
                "is_active": True,
                "created_at": "2023-01-01T10:00:00Z",
                "updated_at": "2023-01-01T10:00:00Z"
            }
        }

class RiskCheck(BaseModel):
    """Risk check model."""
    id: str = Field(default_factory=lambda: f"check_{uuid.uuid4().hex[:8]}")
    limit_id: str
    value: float
    threshold: float
    is_breached: bool
    risk_level: RiskLevel
    action: RiskAction
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "check_12345678",
                "limit_id": "limit_12345678",
                "value": 0.06,
                "threshold": 0.05,
                "is_breached": True,
                "risk_level": "medium",
                "action": "warn",
                "timestamp": "2023-01-01T10:01:00Z",
                "context": {
                    "symbol": "AAPL",
                    "position_size": 0.06,
                    "portfolio_value": 100000.0
                }
            }
        }

class PortfolioAllocation(BaseModel):
    """Portfolio allocation model."""
    id: str = Field(default_factory=lambda: f"alloc_{uuid.uuid4().hex[:8]}")
    portfolio_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    allocations: Dict[str, float]  # asset_id -> weight
    expected_return: Optional[float] = None
    expected_risk: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    optimization_method: OptimizationMethod
    constraints: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "alloc_12345678",
                "portfolio_id": "portfolio_12345678",
                "timestamp": "2023-01-01T10:00:00Z",
                "allocations": {
                    "AAPL": 0.05,
                    "MSFT": 0.05,
                    "AMZN": 0.04,
                    "GOOGL": 0.04,
                    "BRK.B": 0.03
                },
                "expected_return": 0.12,
                "expected_risk": 0.18,
                "sharpe_ratio": 0.67,
                "optimization_method": "mean_variance",
                "constraints": {
                    "long_only": True,
                    "fully_invested": True
                }
            }
        }

class PortfolioRisk(BaseModel):
    """Portfolio risk model."""
    id: str = Field(default_factory=lambda: f"risk_{uuid.uuid4().hex[:8]}")
    portfolio_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_value: float
    cash: float
    invested: float
    leverage: float
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    expected_shortfall: float
    volatility: float
    beta: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: float
    correlation_matrix: Dict[str, Dict[str, float]] = {}  # asset_id -> asset_id -> correlation
    exposures: Dict[str, Dict[str, float]] = {}  # exposure_type -> exposure_name -> value
    stress_tests: Dict[str, float] = {}  # scenario_name -> portfolio_return
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "risk_12345678",
                "portfolio_id": "portfolio_12345678",
                "timestamp": "2023-01-01T10:00:00Z",
                "total_value": 100000.0,
                "cash": 20000.0,
                "invested": 80000.0,
                "leverage": 1.0,
                "var_95": 2000.0,
                "var_99": 3500.0,
                "expected_shortfall": 4000.0,
                "volatility": 0.15,
                "beta": 0.8,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.5,
                "max_drawdown": 0.12,
                "exposures": {
                    "sector": {
                        "technology": 0.25,
                        "financials": 0.15,
                        "healthcare": 0.10
                    },
                    "factor": {
                        "momentum": 0.2,
                        "value": 0.1,
                        "size": -0.05
                    }
                },
                "stress_tests": {
                    "2008_crisis": -0.25,
                    "covid_crash": -0.18,
                    "rate_hike": -0.08
                }
            }
        }

class ExposureAnalysis(BaseModel):
    """Exposure analysis model."""
    id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    portfolio_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    exposure_type: str  # "sector", "factor", "asset_class", "geography", "currency"
    exposures: Dict[str, float]  # exposure_name -> value
    net_exposure: float
    gross_exposure: float
    long_exposure: float
    short_exposure: float
    benchmark_exposures: Optional[Dict[str, float]] = None
    active_exposures: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "exp_12345678",
                "portfolio_id": "portfolio_12345678",
                "timestamp": "2023-01-01T10:00:00Z",
                "exposure_type": "sector",
                "exposures": {
                    "technology": 0.25,
                    "financials": 0.15,
                    "healthcare": 0.10,
                    "consumer_discretionary": 0.08,
                    "industrials": 0.07
                },
                "net_exposure": 0.65,
                "gross_exposure": 0.65,
                "long_exposure": 0.65,
                "short_exposure": 0.0,
                "benchmark_exposures": {
                    "technology": 0.20,
                    "financials": 0.15,
                    "healthcare": 0.12,
                    "consumer_discretionary": 0.10,
                    "industrials": 0.10
                },
                "active_exposures": {
                    "technology": 0.05,
                    "financials": 0.0,
                    "healthcare": -0.02,
                    "consumer_discretionary": -0.02,
                    "industrials": -0.03
                }
            }
        }

class DrawdownEvent(BaseModel):
    """Drawdown event model."""
    id: str = Field(default_factory=lambda: f"dd_{uuid.uuid4().hex[:8]}")
    portfolio_id: str
    start_date: datetime
    current_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    peak_value: float
    current_value: float
    drawdown_pct: float
    drawdown_duration_days: int
    is_active: bool = True
    actions_taken: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "dd_12345678",
                "portfolio_id": "portfolio_12345678",
                "start_date": "2023-01-01T10:00:00Z",
                "current_date": "2023-01-15T10:00:00Z",
                "peak_value": 100000.0,
                "current_value": 90000.0,
                "drawdown_pct": 0.10,
                "drawdown_duration_days": 14,
                "is_active": True,
                "actions_taken": [
                    {
                        "date": "2023-01-10T10:00:00Z",
                        "action": "reduce",
                        "reduction_pct": 0.25,
                        "description": "Reduced exposure by 25% due to 7.5% drawdown"
                    }
                ]
            }
        }

class RiskProfile(BaseModel):
    """Risk profile model."""
    id: str = Field(default_factory=lambda: f"profile_{uuid.uuid4().hex[:8]}")
    name: str
    description: Optional[str] = None
    risk_limits: List[RiskLimit] = []
    max_drawdown_pct: float
    max_leverage: float
    max_position_size_pct: float
    max_sector_exposure_pct: float
    var_limit_pct: float
    target_volatility: Optional[float] = None
    target_beta: Optional[float] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "profile_12345678",
                "name": "Conservative",
                "description": "Conservative risk profile with lower risk limits",
                "max_drawdown_pct": 0.10,
                "max_leverage": 1.5,
                "max_position_size_pct": 0.03,
                "max_sector_exposure_pct": 0.20,
                "var_limit_pct": 0.03,
                "target_volatility": 0.10,
                "target_beta": 0.7,
                "is_active": True,
                "created_at": "2023-01-01T10:00:00Z",
                "updated_at": "2023-01-01T10:00:00Z"
            }
        }

# Request/Response Models
class RiskCheckRequest(BaseModel):
    """Request model for risk check."""
    order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    portfolio_id: Optional[str] = None
    check_types: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "order_id": "order_12345678",
                "strategy_id": "strategy_12345678",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100.0,
                "price": 150.0,
                "portfolio_id": "portfolio_12345678",
                "check_types": ["position_size", "concentration", "var"]
            }
        }

class RiskCheckResponse(BaseModel):
    """Response model for risk check."""
    is_approved: bool
    checks: List[RiskCheck]
    message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "is_approved": True,
                "checks": [
                    {
                        "id": "check_12345678",
                        "limit_id": "limit_12345678",
                        "value": 0.04,
                        "threshold": 0.05,
                        "is_breached": False,
                        "risk_level": "low",
                        "action": "allow",
                        "timestamp": "2023-01-01T10:01:00Z",
                        "context": {
                            "symbol": "AAPL",
                            "position_size": 0.04,
                            "portfolio_value": 100000.0
                        }
                    }
                ],
                "message": "Order approved"
            }
        }

class OptimizePortfolioRequest(BaseModel):
    """Request model for portfolio optimization."""
    portfolio_id: str
    optimization_method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    constraints: Dict[str, Any] = {}
    assets: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "portfolio_id": "portfolio_12345678",
                "optimization_method": "mean_variance",
                "target_return": 0.12,
                "constraints": {
                    "long_only": True,
                    "fully_invested": True
                },
                "assets": ["AAPL", "MSFT", "AMZN", "GOOGL", "BRK.B"]
            }
        }

class OptimizePortfolioResponse(BaseModel):
    """Response model for portfolio optimization."""
    allocation: PortfolioAllocation
    
    class Config:
        schema_extra = {
            "example": {
                "allocation": {
                    "id": "alloc_12345678",
                    "portfolio_id": "portfolio_12345678",
                    "timestamp": "2023-01-01T10:00:00Z",
                    "allocations": {
                        "AAPL": 0.05,
                        "MSFT": 0.05,
                        "AMZN": 0.04,
                        "GOOGL": 0.04,
                        "BRK.B": 0.03
                    },
                    "expected_return": 0.12,
                    "expected_risk": 0.18,
                    "sharpe_ratio": 0.67,
                    "optimization_method": "mean_variance",
                    "constraints": {
                        "long_only": True,
                        "fully_invested": True
                    }
                }
            }
        }

class GetPortfolioRiskRequest(BaseModel):
    """Request model for getting portfolio risk."""
    portfolio_id: str
    include_stress_tests: bool = False
    include_exposures: bool = True
    include_correlations: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "portfolio_id": "portfolio_12345678",
                "include_stress_tests": True,
                "include_exposures": True,
                "include_correlations": False
            }
        }

class GetPortfolioRiskResponse(BaseModel):
    """Response model for getting portfolio risk."""
    risk: PortfolioRisk
    
    class Config:
        schema_extra = {
            "example": {
                "risk": {
                    "id": "risk_12345678",
                    "portfolio_id": "portfolio_12345678",
                    "timestamp": "2023-01-01T10:00:00Z",
                    "total_value": 100000.0,
                    "cash": 20000.0,
                    "invested": 80000.0,
                    "leverage": 1.0,
                    "var_95": 2000.0,
                    "var_99": 3500.0,
                    "expected_shortfall": 4000.0,
                    "volatility": 0.15,
                    "beta": 0.8,
                    "sharpe_ratio": 1.2,
                    "sortino_ratio": 1.5,
                    "max_drawdown": 0.12,
                    "exposures": {
                        "sector": {
                            "technology": 0.25,
                            "financials": 0.15,
                            "healthcare": 0.10
                        }
                    },
                    "stress_tests": {
                        "2008_crisis": -0.25,
                        "covid_crash": -0.18,
                        "rate_hike": -0.08
                    }
                }
            }
        }

class GetExposureAnalysisRequest(BaseModel):
    """Request model for getting exposure analysis."""
    portfolio_id: str
    exposure_type: str
    include_benchmark: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "portfolio_id": "portfolio_12345678",
                "exposure_type": "sector",
                "include_benchmark": True
            }
        }

class GetExposureAnalysisResponse(BaseModel):
    """Response model for getting exposure analysis."""
    exposure: ExposureAnalysis
    
    class Config:
        schema_extra = {
            "example": {
                "exposure": {
                    "id": "exp_12345678",
                    "portfolio_id": "portfolio_12345678",
                    "timestamp": "2023-01-01T10:00:00Z",
                    "exposure_type": "sector",
                    "exposures": {
                        "technology": 0.25,
                        "financials": 0.15,
                        "healthcare": 0.10,
                        "consumer_discretionary": 0.08,
                        "industrials": 0.07
                    },
                    "net_exposure": 0.65,
                    "gross_exposure": 0.65,
                    "long_exposure": 0.65,
                    "short_exposure": 0.0,
                    "benchmark_exposures": {
                        "technology": 0.20,
                        "financials": 0.15,
                        "healthcare": 0.12,
                        "consumer_discretionary": 0.10,
                        "industrials": 0.10
                    },
                    "active_exposures": {
                        "technology": 0.05,
                        "financials": 0.0,
                        "healthcare": -0.02,
                        "consumer_discretionary": -0.02,
                        "industrials": -0.03
                    }
                }
            }
        }

class GetDrawdownEventsRequest(BaseModel):
    """Request model for getting drawdown events."""
    portfolio_id: str
    include_active_only: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "portfolio_id": "portfolio_12345678",
                "include_active_only": True,
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2023-01-31T23:59:59Z"
            }
        }

class GetDrawdownEventsResponse(BaseModel):
    """Response model for getting drawdown events."""
    events: List[DrawdownEvent]
    
    class Config:
        schema_extra = {
            "example": {
                "events": [
                    {
                        "id": "dd_12345678",
                        "portfolio_id": "portfolio_12345678",
                        "start_date": "2023-01-01T10:00:00Z",
                        "current_date": "2023-01-15T10:00:00Z",
                        "peak_value": 100000.0,
                        "current_value": 90000.0,
                        "drawdown_pct": 0.10,
                        "drawdown_duration_days": 14,
                        "is_active": True,
                        "actions_taken": [
                            {
                                "date": "2023-01-10T10:00:00Z",
                                "action": "reduce",
                                "reduction_pct": 0.25,
                                "description": "Reduced exposure by 25% due to 7.5% drawdown"
                            }
                        ]
                    }
                ]
            }
        }

class CreateRiskProfileRequest(BaseModel):
    """Request model for creating risk profile."""
    name: str
    description: Optional[str] = None
    max_drawdown_pct: float
    max_leverage: float
    max_position_size_pct: float
    max_sector_exposure_pct: float
    var_limit_pct: float
    target_volatility: Optional[float] = None
    target_beta: Optional[float] = None
    risk_limits: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Conservative",
                "description": "Conservative risk profile with lower risk limits",
                "max_drawdown_pct": 0.10,
                "max_leverage": 1.5,
                "max_position_size_pct": 0.03,
                "max_sector_exposure_pct": 0.20,
                "var_limit_pct": 0.03,
                "target_volatility": 0.10,
                "target_beta": 0.7,
                "risk_limits": [
                    {
                        "name": "Max Position Size",
                        "risk_type": "concentration",
                        "threshold": 0.03,
                        "warning_threshold": 0.025,
                        "action": "warn",
                        "scope": "portfolio"
                    }
                ]
            }
        }

class CreateRiskProfileResponse(BaseModel):
    """Response model for creating risk profile."""
    profile: RiskProfile
    
    class Config:
        schema_extra = {
            "example": {
                "profile": {
                    "id": "profile_12345678",
                    "name": "Conservative",
                    "description": "Conservative risk profile with lower risk limits",
                    "max_drawdown_pct": 0.10,
                    "max_leverage": 1.5,
                    "max_position_size_pct": 0.03,
                    "max_sector_exposure_pct": 0.20,
                    "var_limit_pct": 0.03,
                    "target_volatility": 0.10,
                    "target_beta": 0.7,
                    "is_active": True,
                    "created_at": "2023-01-01T10:00:00Z",
                    "updated_at": "2023-01-01T10:00:00Z"
                }
            }
        }
