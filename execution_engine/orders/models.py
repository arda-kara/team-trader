"""
Base models for the execution engine.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import uuid

# Enums
class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good till date

class OrderStatus(str, Enum):
    """Order status enumeration."""
    NEW = "new"
    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionAlgorithm(str, Enum):
    """Execution algorithm enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    IS = "is"      # Implementation shortfall

class BrokerType(str, Enum):
    """Broker type enumeration."""
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    BINANCE = "binance"
    SIMULATION = "simulation"

class AssetClass(str, Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    FIXED_INCOME = "fixed_income"

class AssetType(str, Enum):
    """Asset type enumeration."""
    STOCK = "stock"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    BOND = "bond"

class Currency(str, Enum):
    """Currency enumeration."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"
    BTC = "BTC"
    ETH = "ETH"

# Base Models
class Asset(BaseModel):
    """Asset model."""
    symbol: str
    name: Optional[str] = None
    asset_class: AssetClass
    asset_type: AssetType
    exchange: Optional[str] = None
    currency: Currency = Currency.USD
    is_tradable: bool = True
    is_shortable: bool = True
    is_marginable: bool = True
    min_trade_size: Optional[float] = None
    price_increment: Optional[float] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "asset_class": "equity",
                "asset_type": "stock",
                "exchange": "NASDAQ",
                "currency": "USD",
                "is_tradable": True,
                "is_shortable": True,
                "is_marginable": True,
                "min_trade_size": 1.0,
                "price_increment": 0.01,
                "metadata": {"sector": "Technology", "industry": "Consumer Electronics"}
            }
        }

class Order(BaseModel):
    """Order model."""
    id: str = Field(default_factory=lambda: f"order_{uuid.uuid4().hex[:8]}")
    client_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW
    broker: BrokerType
    broker_order_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    filled_avg_price: Optional[float] = None
    commission: Optional[float] = None
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    algorithm_params: Dict[str, Any] = {}
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "order_12345678",
                "client_order_id": "client_order_123",
                "strategy_id": "strategy_123",
                "signal_id": "signal_123",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100.0,
                "order_type": "market",
                "time_in_force": "day",
                "status": "new",
                "broker": "alpaca",
                "created_at": "2023-01-01T10:00:00Z",
                "execution_algorithm": "market"
            }
        }
    
    @validator('limit_price')
    def validate_limit_price(cls, v, values):
        """Validate limit price based on order type."""
        order_type = values.get('order_type')
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError(f"Limit price is required for {order_type} orders")
        return v
    
    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        """Validate stop price based on order type."""
        order_type = values.get('order_type')
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and v is None:
            raise ValueError(f"Stop price is required for {order_type} orders")
        return v

class Execution(BaseModel):
    """Execution model."""
    id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:8]}")
    order_id: str
    broker_order_id: Optional[str] = None
    broker_execution_id: Optional[str] = None
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    commission: float = 0.0
    liquidity: Optional[str] = None  # "added", "removed", "routed"
    venue: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "exec_12345678",
                "order_id": "order_12345678",
                "broker_order_id": "broker_order_123",
                "broker_execution_id": "broker_exec_123",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100.0,
                "price": 150.25,
                "timestamp": "2023-01-01T10:01:15Z",
                "commission": 1.0,
                "liquidity": "removed",
                "venue": "NASDAQ"
            }
        }

class Position(BaseModel):
    """Position model."""
    id: str = Field(default_factory=lambda: f"pos_{uuid.uuid4().hex[:8]}")
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    cost_basis: float
    unrealized_pl: Optional[float] = None
    unrealized_pl_pct: Optional[float] = None
    realized_pl: float = 0.0
    realized_pl_pct: Optional[float] = None
    open_time: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    strategy_id: Optional[str] = None
    broker: BrokerType
    asset_class: AssetClass
    asset_type: AssetType
    currency: Currency = Currency.USD
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "pos_12345678",
                "symbol": "AAPL",
                "quantity": 100.0,
                "avg_entry_price": 150.25,
                "current_price": 155.50,
                "market_value": 15550.0,
                "cost_basis": 15025.0,
                "unrealized_pl": 525.0,
                "unrealized_pl_pct": 0.035,
                "realized_pl": 0.0,
                "open_time": "2023-01-01T10:01:15Z",
                "updated_at": "2023-01-01T16:00:00Z",
                "strategy_id": "strategy_123",
                "broker": "alpaca",
                "asset_class": "equity",
                "asset_type": "stock",
                "currency": "USD"
            }
        }

class Portfolio(BaseModel):
    """Portfolio model."""
    id: str = Field(default_factory=lambda: f"portfolio_{uuid.uuid4().hex[:8]}")
    name: str
    cash: float
    equity: float
    buying_power: float
    positions: List[Position] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    broker: BrokerType
    currency: Currency = Currency.USD
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "portfolio_12345678",
                "name": "Main Portfolio",
                "cash": 50000.0,
                "equity": 100000.0,
                "buying_power": 150000.0,
                "positions": [],  # Truncated for brevity
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T16:00:00Z",
                "broker": "alpaca",
                "currency": "USD"
            }
        }

class ExecutionReport(BaseModel):
    """Execution report model."""
    id: str = Field(default_factory=lambda: f"report_{uuid.uuid4().hex[:8]}")
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    filled_quantity: float
    avg_price: Optional[float] = None
    status: OrderStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    broker: BrokerType
    broker_order_id: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "report_12345678",
                "order_id": "order_12345678",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100.0,
                "filled_quantity": 100.0,
                "avg_price": 150.25,
                "status": "filled",
                "timestamp": "2023-01-01T10:01:15Z",
                "broker": "alpaca",
                "broker_order_id": "broker_order_123",
                "message": "Order filled successfully"
            }
        }

class MarketData(BaseModel):
    """Market data model."""
    symbol: str
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "timestamp": "2023-01-01T10:00:00Z",
                "bid": 150.0,
                "ask": 150.1,
                "last": 150.05,
                "volume": 1000,
                "bid_size": 500,
                "ask_size": 300
            }
        }

# Request/Response Models
class CreateOrderRequest(BaseModel):
    """Request model for order creation."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    broker: Optional[BrokerType] = None
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    algorithm_params: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100.0,
                "order_type": "market",
                "time_in_force": "day",
                "client_order_id": "client_order_123",
                "strategy_id": "strategy_123",
                "signal_id": "signal_123",
                "execution_algorithm": "market"
            }
        }

class CreateOrderResponse(BaseModel):
    """Response model for order creation."""
    order: Order
    
    class Config:
        schema_extra = {
            "example": {
                "order": {
                    "id": "order_12345678",
                    "client_order_id": "client_order_123",
                    "strategy_id": "strategy_123",
                    "signal_id": "signal_123",
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 100.0,
                    "order_type": "market",
                    "time_in_force": "day",
                    "status": "new",
                    "broker": "alpaca",
                    "created_at": "2023-01-01T10:00:00Z",
                    "execution_algorithm": "market"
                }
            }
        }

class CancelOrderRequest(BaseModel):
    """Request model for order cancellation."""
    order_id: str
    
    class Config:
        schema_extra = {
            "example": {
                "order_id": "order_12345678"
            }
        }

class CancelOrderResponse(BaseModel):
    """Response model for order cancellation."""
    success: bool
    order_id: str
    message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "order_id": "order_12345678",
                "message": "Order canceled successfully"
            }
        }

class GetOrderRequest(BaseModel):
    """Request model for getting order details."""
    order_id: str
    
    class Config:
        schema_extra = {
            "example": {
                "order_id": "order_12345678"
            }
        }

class GetOrderResponse(BaseModel):
    """Response model for getting order details."""
    order: Order
    
    class Config:
        schema_extra = {
            "example": {
                "order": {
                    "id": "order_12345678",
                    "client_order_id": "client_order_123",
                    "strategy_id": "strategy_123",
                    "signal_id": "signal_123",
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 100.0,
                    "order_type": "market",
                    "time_in_force": "day",
                    "status": "filled",
                    "broker": "alpaca",
                    "created_at": "2023-01-01T10:00:00Z",
                    "filled_at": "2023-01-01T10:01:15Z",
                    "filled_quantity": 100.0,
                    "filled_avg_price": 150.25,
                    "execution_algorithm": "market"
                }
            }
        }

class GetPositionsRequest(BaseModel):
    """Request model for getting positions."""
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy_123",
                "symbol": "AAPL"
            }
        }

class GetPositionsResponse(BaseModel):
    """Response model for getting positions."""
    positions: List[Position]
    
    class Config:
        schema_extra = {
            "example": {
                "positions": [
                    {
                        "id": "pos_12345678",
                        "symbol": "AAPL",
                        "quantity": 100.0,
                        "avg_entry_price": 150.25,
                        "current_price": 155.50,
                        "market_value": 15550.0,
                        "cost_basis": 15025.0,
                        "unrealized_pl": 525.0,
                        "unrealized_pl_pct": 0.035,
                        "realized_pl": 0.0,
                        "open_time": "2023-01-01T10:01:15Z",
                        "updated_at": "2023-01-01T16:00:00Z",
                        "strategy_id": "strategy_123",
                        "broker": "alpaca",
                        "asset_class": "equity",
                        "asset_type": "stock",
                        "currency": "USD"
                    }
                ]
            }
        }

class GetPortfolioRequest(BaseModel):
    """Request model for getting portfolio."""
    pass
    
    class Config:
        schema_extra = {
            "example": {}
        }

class GetPortfolioResponse(BaseModel):
    """Response model for getting portfolio."""
    portfolio: Portfolio
    
    class Config:
        schema_extra = {
            "example": {
                "portfolio": {
                    "id": "portfolio_12345678",
                    "name": "Main Portfolio",
                    "cash": 50000.0,
                    "equity": 100000.0,
                    "buying_power": 150000.0,
                    "positions": [],  # Truncated for brevity
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-01T16:00:00Z",
                    "broker": "alpaca",
                    "currency": "USD"
                }
            }
        }

class ExecuteStrategyRequest(BaseModel):
    """Request model for executing a strategy."""
    strategy_id: str
    signals: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy_123",
                "signals": [
                    {
                        "id": "signal_123",
                        "symbol": "AAPL",
                        "direction": "bullish",
                        "strength": 0.8,
                        "timestamp": "2023-01-01T10:00:00Z"
                    }
                ]
            }
        }

class ExecuteStrategyResponse(BaseModel):
    """Response model for executing a strategy."""
    orders: List[Order]
    
    class Config:
        schema_extra = {
            "example": {
                "orders": [
                    {
                        "id": "order_12345678",
                        "client_order_id": "client_order_123",
                        "strategy_id": "strategy_123",
                        "signal_id": "signal_123",
                        "symbol": "AAPL",
                        "side": "buy",
                        "quantity": 100.0,
                        "order_type": "market",
                        "time_in_force": "day",
                        "status": "new",
                        "broker": "alpaca",
                        "created_at": "2023-01-01T10:00:00Z",
                        "execution_algorithm": "market"
                    }
                ]
            }
        }
