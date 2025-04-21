"""
API models for the dashboard interface.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator

# User models
class UserBase(BaseModel):
    """Base user model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    
class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(char in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`' for char in v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[str] = Field(None, regex=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    is_active: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None

class UserResponse(UserBase):
    """User response model."""
    id: int
    is_active: bool
    is_superuser: bool
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    roles: List[str] = []
    
    class Config:
        orm_mode = True

# Authentication models
class Token(BaseModel):
    """Token model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: int  # Unix timestamp

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: List[str] = []
    exp: Optional[int] = None  # Expiration time

class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str

class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str

# Dashboard models
class WidgetPosition(BaseModel):
    """Widget position model."""
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., ge=1)
    h: int = Field(..., ge=1)

class WidgetBase(BaseModel):
    """Base widget model."""
    name: str = Field(..., min_length=1, max_length=100)
    widget_type: str = Field(..., min_length=1, max_length=50)
    config: Dict[str, Any] = Field(default_factory=dict)
    position: WidgetPosition

class WidgetCreate(WidgetBase):
    """Widget creation model."""
    dashboard_id: int

class WidgetUpdate(BaseModel):
    """Widget update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    config: Optional[Dict[str, Any]] = None
    position: Optional[WidgetPosition] = None

class WidgetResponse(WidgetBase):
    """Widget response model."""
    id: int
    dashboard_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class DashboardBase(BaseModel):
    """Base dashboard model."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=200)
    is_default: bool = False
    is_public: bool = False

class DashboardCreate(DashboardBase):
    """Dashboard creation model."""
    widgets: Optional[List[WidgetBase]] = None

class DashboardUpdate(BaseModel):
    """Dashboard update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=200)
    is_default: Optional[bool] = None
    is_public: Optional[bool] = None
    layout: Optional[Dict[str, Any]] = None

class DashboardResponse(DashboardBase):
    """Dashboard response model."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    widgets: List[WidgetResponse] = []
    
    class Config:
        orm_mode = True

# Alert models
class AlertBase(BaseModel):
    """Base alert model."""
    title: str = Field(..., min_length=1, max_length=100)
    message: Optional[str] = None
    severity: str = Field(..., regex=r"^(info|warning|error|critical)$")
    source: str = Field(..., min_length=1, max_length=50)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AlertCreate(AlertBase):
    """Alert creation model."""
    user_id: Optional[int] = None

class AlertUpdate(BaseModel):
    """Alert update model."""
    is_read: Optional[bool] = None
    is_acknowledged: Optional[bool] = None

class AlertResponse(AlertBase):
    """Alert response model."""
    id: int
    is_read: bool
    is_acknowledged: bool
    acknowledged_at: Optional[datetime] = None
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

# System status models
class SystemStatusBase(BaseModel):
    """Base system status model."""
    component: str = Field(..., min_length=1, max_length=50)
    status: str = Field(..., regex=r"^(online|offline|degraded)$")
    message: Optional[str] = Field(None, max_length=200)
    details: Dict[str, Any] = Field(default_factory=dict)

class SystemStatusCreate(SystemStatusBase):
    """System status creation model."""
    pass

class SystemStatusUpdate(BaseModel):
    """System status update model."""
    status: Optional[str] = Field(None, regex=r"^(online|offline|degraded)$")
    message: Optional[str] = Field(None, max_length=200)
    details: Optional[Dict[str, Any]] = None

class SystemStatusResponse(SystemStatusBase):
    """System status response model."""
    id: int
    last_check: datetime
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

# API key models
class ApiKeyBase(BaseModel):
    """Base API key model."""
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None

class ApiKeyCreate(ApiKeyBase):
    """API key creation model."""
    user_id: int

class ApiKeyResponse(ApiKeyBase):
    """API key response model."""
    id: int
    key: str
    user_id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_used: Optional[datetime] = None
    
    class Config:
        orm_mode = True

# Query models
class SavedQueryBase(BaseModel):
    """Base saved query model."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=200)
    query_type: str = Field(..., min_length=1, max_length=50)
    query_params: Dict[str, Any] = Field(default_factory=dict)
    is_public: bool = False

class SavedQueryCreate(SavedQueryBase):
    """Saved query creation model."""
    user_id: int

class SavedQueryUpdate(BaseModel):
    """Saved query update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=200)
    query_params: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None

class SavedQueryResponse(SavedQueryBase):
    """Saved query response model."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

# Notification models
class NotificationBase(BaseModel):
    """Base notification model."""
    title: str = Field(..., min_length=1, max_length=100)
    message: Optional[str] = None
    notification_type: str = Field(..., min_length=1, max_length=50)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NotificationCreate(NotificationBase):
    """Notification creation model."""
    user_id: int

class NotificationUpdate(BaseModel):
    """Notification update model."""
    is_read: Optional[bool] = None

class NotificationResponse(NotificationBase):
    """Notification response model."""
    id: int
    is_read: bool
    user_id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

# Scheduled task models
class ScheduledTaskBase(BaseModel):
    """Base scheduled task model."""
    name: str = Field(..., min_length=1, max_length=100)
    task_type: str = Field(..., min_length=1, max_length=50)
    schedule: str = Field(..., min_length=1, max_length=100)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

class ScheduledTaskCreate(ScheduledTaskBase):
    """Scheduled task creation model."""
    created_by: int

class ScheduledTaskUpdate(BaseModel):
    """Scheduled task update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    schedule: Optional[str] = Field(None, min_length=1, max_length=100)
    parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class ScheduledTaskResponse(ScheduledTaskBase):
    """Scheduled task response model."""
    id: int
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    created_by: int
    
    class Config:
        orm_mode = True

# Role models
class RoleBase(BaseModel):
    """Base role model."""
    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=200)
    permissions: List[str] = Field(default_factory=list)

class RoleCreate(RoleBase):
    """Role creation model."""
    pass

class RoleUpdate(BaseModel):
    """Role update model."""
    description: Optional[str] = Field(None, max_length=200)
    permissions: Optional[List[str]] = None

class RoleResponse(RoleBase):
    """Role response model."""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

# Audit log models
class AuditLogBase(BaseModel):
    """Base audit log model."""
    action: str = Field(..., min_length=1, max_length=50)
    entity_type: str = Field(..., min_length=1, max_length=50)
    entity_id: Optional[str] = Field(None, max_length=50)
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = Field(None, max_length=50)
    user_agent: Optional[str] = Field(None, max_length=200)

class AuditLogCreate(AuditLogBase):
    """Audit log creation model."""
    user_id: Optional[int] = None

class AuditLogResponse(AuditLogBase):
    """Audit log response model."""
    id: int
    user_id: Optional[int] = None
    created_at: datetime
    
    class Config:
        orm_mode = True

# Component data models
class MarketDataResponse(BaseModel):
    """Market data response model."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    close: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PortfolioPositionResponse(BaseModel):
    """Portfolio position response model."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    realized_pnl: float
    weight: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PortfolioSummaryResponse(BaseModel):
    """Portfolio summary response model."""
    total_value: float
    cash: float
    invested: float
    daily_pnl: float
    daily_pnl_percent: float
    total_pnl: float
    total_pnl_percent: float
    positions: List[PortfolioPositionResponse] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TradeResponse(BaseModel):
    """Trade response model."""
    id: str
    symbol: str
    side: str  # buy, sell
    quantity: float
    price: float
    timestamp: datetime
    status: str  # filled, partial, cancelled, pending
    order_id: str
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OrderResponse(BaseModel):
    """Order response model."""
    id: str
    symbol: str
    side: str  # buy, sell
    order_type: str  # market, limit, stop, etc.
    quantity: float
    price: Optional[float] = None
    status: str  # open, filled, cancelled, etc.
    created_at: datetime
    updated_at: datetime
    filled_quantity: float = 0
    average_fill_price: Optional[float] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StrategyResponse(BaseModel):
    """Strategy response model."""
    id: str
    name: str
    type: str
    status: str  # active, inactive, backtest
    performance: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SignalResponse(BaseModel):
    """Signal response model."""
    id: str
    symbol: str
    signal_type: str
    direction: str  # long, short, neutral
    strength: float
    timestamp: datetime
    source: str
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RiskMetricsResponse(BaseModel):
    """Risk metrics response model."""
    portfolio_var: float
    portfolio_cvar: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    volatility: float
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    var_by_position: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentActivityResponse(BaseModel):
    """Agent activity response model."""
    id: str
    agent_id: str
    agent_type: str
    action: str
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response model."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    best_trade: float
    worst_trade: float
    time_in_market: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
