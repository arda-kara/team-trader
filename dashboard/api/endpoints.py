"""
Backend API endpoints for the dashboard interface.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from ..config.settings import settings
from ..models.api import (
    UserCreate, UserUpdate, UserResponse, Token, TokenData, LoginRequest, RefreshTokenRequest,
    DashboardCreate, DashboardUpdate, DashboardResponse,
    WidgetCreate, WidgetUpdate, WidgetResponse,
    AlertCreate, AlertUpdate, AlertResponse,
    SystemStatusCreate, SystemStatusUpdate, SystemStatusResponse,
    ApiKeyCreate, ApiKeyResponse,
    SavedQueryCreate, SavedQueryUpdate, SavedQueryResponse,
    NotificationCreate, NotificationUpdate, NotificationResponse,
    RoleCreate, RoleUpdate, RoleResponse,
    AuditLogCreate, AuditLogResponse,
    MarketDataResponse, PortfolioSummaryResponse, TradeResponse, OrderResponse,
    StrategyResponse, SignalResponse, RiskMetricsResponse, AgentActivityResponse,
    PerformanceMetricsResponse
)
from ..models.repositories import (
    UserRepository, RoleRepository, DashboardRepository, WidgetRepository,
    AlertRepository, AuditLogRepository, SystemStatusRepository, ApiKeyRepository,
    SavedQueryRepository, NotificationRepository, UserSessionRepository, ScheduledTaskRepository
)
from ..utils.database import get_db
from ..utils.auth import (
    get_password_hash, verify_password, create_access_token, create_refresh_token,
    get_current_user, get_current_active_user, get_current_admin_user
)
from ..utils.audit import create_audit_log
from ..services.component_service import ComponentService

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Create component service
component_service = ComponentService()

# Authentication endpoints
@router.post("/auth/login", response_model=Token)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Login user.
    
    Args:
        login_data: Login data
        db: Database session
        
    Returns:
        Token: Authentication token
    """
    user_repo = UserRepository(db)
    user = user_repo.get_by_username(login_data.username)
    
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login
    user_repo.update_last_login(user.id)
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.auth.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token_expires = timedelta(days=settings.auth.refresh_token_expire_days)
    refresh_token = create_refresh_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=refresh_token_expires
    )
    
    # Create user session
    session_repo = UserSessionRepository(db)
    session_repo.create({
        "user_id": user.id,
        "session_id": access_token,
        "ip_address": "127.0.0.1",  # This would be extracted from request in a real system
        "user_agent": "Unknown",    # This would be extracted from request in a real system
        "expires_at": datetime.utcnow() + access_token_expires,
        "is_active": True
    })
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=user.id,
        action="login",
        entity_type="user",
        entity_id=str(user.id),
        details={"ip_address": "127.0.0.1"}  # This would be extracted from request in a real system
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_at": int((datetime.utcnow() + access_token_expires).timestamp())
    }

@router.post("/auth/refresh", response_model=Token)
async def refresh_token(refresh_data: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh authentication token.
    
    Args:
        refresh_data: Refresh token data
        db: Database session
        
    Returns:
        Token: New authentication token
    """
    try:
        # Decode refresh token
        payload = jwt.decode(
            refresh_data.refresh_token,
            settings.auth.secret_key,
            algorithms=[settings.auth.algorithm]
        )
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if username is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user
        user_repo = UserRepository(db)
        user = user_repo.get_by_id(user_id)
        
        if not user or user.username != username or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.auth.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=access_token_expires
        )
        
        # Create new refresh token
        refresh_token_expires = timedelta(days=settings.auth.refresh_token_expire_days)
        new_refresh_token = create_refresh_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=refresh_token_expires
        )
        
        # Update user session
        session_repo = UserSessionRepository(db)
        session_repo.create({
            "user_id": user.id,
            "session_id": access_token,
            "ip_address": "127.0.0.1",  # This would be extracted from request in a real system
            "user_agent": "Unknown",    # This would be extracted from request in a real system
            "expires_at": datetime.utcnow() + access_token_expires,
            "is_active": True
        })
        
        # Create audit log
        create_audit_log(
            db=db,
            user_id=user.id,
            action="token_refresh",
            entity_type="user",
            entity_id=str(user.id),
            details={"ip_address": "127.0.0.1"}  # This would be extracted from request in a real system
        )
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_at": int((datetime.utcnow() + access_token_expires).timestamp())
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/auth/logout")
async def logout(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Logout user.
    
    Args:
        db: Database session
        current_user: Current user
        
    Returns:
        Dict[str, Any]: Logout result
    """
    # Deactivate user session
    session_repo = UserSessionRepository(db)
    session_repo.deactivate_all_for_user(current_user.id)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="logout",
        entity_type="user",
        entity_id=str(current_user.id),
        details={"ip_address": "127.0.0.1"}  # This would be extracted from request in a real system
    )
    
    return {"message": "Successfully logged out"}

# User endpoints
@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_admin_user)
):
    """Create user.
    
    Args:
        user_data: User data
        db: Database session
        current_user: Current user (must be admin)
        
    Returns:
        UserResponse: Created user
    """
    user_repo = UserRepository(db)
    
    # Check if username already exists
    if user_repo.get_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if user_repo.get_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    user = user_repo.create({
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "is_active": True,
        "is_superuser": False,
        "preferences": {}
    })
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="create",
        entity_type="user",
        entity_id=str(user.id),
        details={"username": user.username}
    )
    
    return user

@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_active_user)):
    """Get current user info.
    
    Args:
        current_user: Current user
        
    Returns:
        UserResponse: Current user info
    """
    return current_user

@router.put("/users/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Update current user.
    
    Args:
        user_data: User data
        db: Database session
        current_user: Current user
        
    Returns:
        UserResponse: Updated user
    """
    user_repo = UserRepository(db)
    
    # Check if email already exists
    if user_data.email and user_data.email != current_user.email:
        existing_user = user_repo.get_by_email(user_data.email)
        if existing_user and existing_user.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Update user
    updated_user = user_repo.update(current_user.id, user_data.dict(exclude_unset=True))
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="update",
        entity_type="user",
        entity_id=str(current_user.id),
        details={"fields": list(user_data.dict(exclude_unset=True).keys())}
    )
    
    return updated_user

@router.get("/users", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_admin_user)
):
    """Get users.
    
    Args:
        skip: Number of users to skip
        limit: Maximum number of users to return
        db: Database session
        current_user: Current user (must be admin)
        
    Returns:
        List[UserResponse]: List of users
    """
    user_repo = UserRepository(db)
    return user_repo.get_all(skip=skip, limit=limit)

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int = Path(..., title="User ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_admin_user)
):
    """Get user.
    
    Args:
        user_id: User ID
        db: Database session
        current_user: Current user (must be admin)
        
    Returns:
        UserResponse: User
    """
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_data: UserUpdate,
    user_id: int = Path(..., title="User ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_admin_user)
):
    """Update user.
    
    Args:
        user_data: User data
        user_id: User ID
        db: Database session
        current_user: Current user (must be admin)
        
    Returns:
        UserResponse: Updated user
    """
    user_repo = UserRepository(db)
    
    # Check if user exists
    user = user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if email already exists
    if user_data.email and user_data.email != user.email:
        existing_user = user_repo.get_by_email(user_data.email)
        if existing_user and existing_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Update user
    updated_user = user_repo.update(user_id, user_data.dict(exclude_unset=True))
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="update",
        entity_type="user",
        entity_id=str(user_id),
        details={"fields": list(user_data.dict(exclude_unset=True).keys())}
    )
    
    return updated_user

# Dashboard endpoints
@router.post("/dashboards", response_model=DashboardResponse)
async def create_dashboard(
    dashboard_data: DashboardCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Create dashboard.
    
    Args:
        dashboard_data: Dashboard data
        db: Database session
        current_user: Current user
        
    Returns:
        DashboardResponse: Created dashboard
    """
    dashboard_repo = DashboardRepository(db)
    widget_repo = WidgetRepository(db)
    
    # Create dashboard
    dashboard = dashboard_repo.create({
        "name": dashboard_data.name,
        "description": dashboard_data.description,
        "is_default": dashboard_data.is_default,
        "is_public": dashboard_data.is_public,
        "user_id": current_user.id,
        "layout": {}
    })
    
    # Create widgets if provided
    if dashboard_data.widgets:
        for widget_data in dashboard_data.widgets:
            widget_repo.create({
                "name": widget_data.name,
                "widget_type": widget_data.widget_type,
                "config": widget_data.config,
                "position": widget_data.position.dict(),
                "dashboard_id": dashboard.id
            })
    
    # If this is the default dashboard, update other dashboards
    if dashboard_data.is_default:
        dashboard_repo.set_default(dashboard.id, current_user.id)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="create",
        entity_type="dashboard",
        entity_id=str(dashboard.id),
        details={"name": dashboard.name}
    )
    
    # Refresh dashboard to include widgets
    dashboard = dashboard_repo.get_by_id(dashboard.id)
    
    return dashboard

@router.get("/dashboards", response_model=List[DashboardResponse])
async def get_dashboards(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get dashboards.
    
    Args:
        skip: Number of dashboards to skip
        limit: Maximum number of dashboards to return
        db: Database session
        current_user: Current user
        
    Returns:
        List[DashboardResponse]: List of dashboards
    """
    dashboard_repo = DashboardRepository(db)
    
    # Get user's dashboards
    user_dashboards = dashboard_repo.get_by_user(current_user.id, skip=skip, limit=limit)
    
    # Get public dashboards
    public_dashboards = dashboard_repo.get_public_dashboards(skip=skip, limit=limit)
    
    # Combine and deduplicate
    dashboards = list(user_dashboards)
    for dashboard in public_dashboards:
        if dashboard.id not in [d.id for d in dashboards]:
            dashboards.append(dashboard)
    
    return dashboards[:limit]

@router.get("/dashboards/{dashboard_id}", response_model=DashboardResponse)
async def get_dashboard(
    dashboard_id: int = Path(..., title="Dashboard ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get dashboard.
    
    Args:
        dashboard_id: Dashboard ID
        db: Database session
        current_user: Current user
        
    Returns:
        DashboardResponse: Dashboard
    """
    dashboard_repo = DashboardRepository(db)
    dashboard = dashboard_repo.get_by_id(dashboard_id)
    
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )
    
    # Check if user has access to dashboard
    if dashboard.user_id != current_user.id and not dashboard.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this dashboard"
        )
    
    return dashboard

@router.put("/dashboards/{dashboard_id}", response_model=DashboardResponse)
async def update_dashboard(
    dashboard_data: DashboardUpdate,
    dashboard_id: int = Path(..., title="Dashboard ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Update dashboard.
    
    Args:
        dashboard_data: Dashboard data
        dashboard_id: Dashboard ID
        db: Database session
        current_user: Current user
        
    Returns:
        DashboardResponse: Updated dashboard
    """
    dashboard_repo = DashboardRepository(db)
    
    # Check if dashboard exists
    dashboard = dashboard_repo.get_by_id(dashboard_id)
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )
    
    # Check if user owns dashboard
    if dashboard.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this dashboard"
        )
    
    # Update dashboard
    updated_dashboard = dashboard_repo.update(dashboard_id, dashboard_data.dict(exclude_unset=True))
    
    # If this is the default dashboard, update other dashboards
    if dashboard_data.is_default and dashboard_data.is_default is True:
        dashboard_repo.set_default(dashboard_id, current_user.id)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="update",
        entity_type="dashboard",
        entity_id=str(dashboard_id),
        details={"fields": list(dashboard_data.dict(exclude_unset=True).keys())}
    )
    
    return updated_dashboard

@router.delete("/dashboards/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: int = Path(..., title="Dashboard ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Delete dashboard.
    
    Args:
        dashboard_id: Dashboard ID
        db: Database session
        current_user: Current user
        
    Returns:
        Dict[str, Any]: Delete result
    """
    dashboard_repo = DashboardRepository(db)
    
    # Check if dashboard exists
    dashboard = dashboard_repo.get_by_id(dashboard_id)
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )
    
    # Check if user owns dashboard
    if dashboard.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this dashboard"
        )
    
    # Delete dashboard
    dashboard_repo.delete(dashboard_id)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="delete",
        entity_type="dashboard",
        entity_id=str(dashboard_id),
        details={"name": dashboard.name}
    )
    
    return {"message": "Dashboard deleted successfully"}

# Widget endpoints
@router.post("/widgets", response_model=WidgetResponse)
async def create_widget(
    widget_data: WidgetCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Create widget.
    
    Args:
        widget_data: Widget data
        db: Database session
        current_user: Current user
        
    Returns:
        WidgetResponse: Created widget
    """
    dashboard_repo = DashboardRepository(db)
    widget_repo = WidgetRepository(db)
    
    # Check if dashboard exists
    dashboard = dashboard_repo.get_by_id(widget_data.dashboard_id)
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )
    
    # Check if user owns dashboard
    if dashboard.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to add widgets to this dashboard"
        )
    
    # Create widget
    widget = widget_repo.create({
        "name": widget_data.name,
        "widget_type": widget_data.widget_type,
        "config": widget_data.config,
        "position": widget_data.position.dict(),
        "dashboard_id": widget_data.dashboard_id
    })
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="create",
        entity_type="widget",
        entity_id=str(widget.id),
        details={"name": widget.name, "dashboard_id": widget.dashboard_id}
    )
    
    return widget

@router.get("/widgets/{widget_id}", response_model=WidgetResponse)
async def get_widget(
    widget_id: int = Path(..., title="Widget ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get widget.
    
    Args:
        widget_id: Widget ID
        db: Database session
        current_user: Current user
        
    Returns:
        WidgetResponse: Widget
    """
    widget_repo = WidgetRepository(db)
    dashboard_repo = DashboardRepository(db)
    
    # Get widget
    widget = widget_repo.get_by_id(widget_id)
    if not widget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Widget not found"
        )
    
    # Get dashboard
    dashboard = dashboard_repo.get_by_id(widget.dashboard_id)
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )
    
    # Check if user has access to dashboard
    if dashboard.user_id != current_user.id and not dashboard.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this widget"
        )
    
    return widget

@router.put("/widgets/{widget_id}", response_model=WidgetResponse)
async def update_widget(
    widget_data: WidgetUpdate,
    widget_id: int = Path(..., title="Widget ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Update widget.
    
    Args:
        widget_data: Widget data
        widget_id: Widget ID
        db: Database session
        current_user: Current user
        
    Returns:
        WidgetResponse: Updated widget
    """
    widget_repo = WidgetRepository(db)
    dashboard_repo = DashboardRepository(db)
    
    # Get widget
    widget = widget_repo.get_by_id(widget_id)
    if not widget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Widget not found"
        )
    
    # Get dashboard
    dashboard = dashboard_repo.get_by_id(widget.dashboard_id)
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )
    
    # Check if user owns dashboard
    if dashboard.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this widget"
        )
    
    # Update widget
    updated_widget = widget_repo.update(widget_id, widget_data.dict(exclude_unset=True))
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="update",
        entity_type="widget",
        entity_id=str(widget_id),
        details={"fields": list(widget_data.dict(exclude_unset=True).keys())}
    )
    
    return updated_widget

@router.delete("/widgets/{widget_id}")
async def delete_widget(
    widget_id: int = Path(..., title="Widget ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Delete widget.
    
    Args:
        widget_id: Widget ID
        db: Database session
        current_user: Current user
        
    Returns:
        Dict[str, Any]: Delete result
    """
    widget_repo = WidgetRepository(db)
    dashboard_repo = DashboardRepository(db)
    
    # Get widget
    widget = widget_repo.get_by_id(widget_id)
    if not widget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Widget not found"
        )
    
    # Get dashboard
    dashboard = dashboard_repo.get_by_id(widget.dashboard_id)
    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )
    
    # Check if user owns dashboard
    if dashboard.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this widget"
        )
    
    # Delete widget
    widget_repo.delete(widget_id)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="delete",
        entity_type="widget",
        entity_id=str(widget_id),
        details={"name": widget.name, "dashboard_id": widget.dashboard_id}
    )
    
    return {"message": "Widget deleted successfully"}

# Alert endpoints
@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    skip: int = 0,
    limit: int = 100,
    unread_only: bool = False,
    severity: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get alerts.
    
    Args:
        skip: Number of alerts to skip
        limit: Maximum number of alerts to return
        unread_only: Whether to return only unread alerts
        severity: Filter by severity
        db: Database session
        current_user: Current user
        
    Returns:
        List[AlertResponse]: List of alerts
    """
    alert_repo = AlertRepository(db)
    
    if unread_only:
        alerts = alert_repo.get_unread_by_user(current_user.id, skip=skip, limit=limit)
    elif severity:
        # Get alerts by severity for the current user
        all_alerts = alert_repo.get_by_severity(severity, skip=0, limit=1000)
        alerts = [alert for alert in all_alerts if alert.user_id == current_user.id]
        alerts = alerts[skip:skip+limit]
    else:
        alerts = alert_repo.get_by_user(current_user.id, skip=skip, limit=limit)
    
    return alerts

@router.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int = Path(..., title="Alert ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get alert.
    
    Args:
        alert_id: Alert ID
        db: Database session
        current_user: Current user
        
    Returns:
        AlertResponse: Alert
    """
    alert_repo = AlertRepository(db)
    
    # Get alert
    alert = alert_repo.get_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    # Check if user has access to alert
    if alert.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this alert"
        )
    
    return alert

@router.put("/alerts/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_data: AlertUpdate,
    alert_id: int = Path(..., title="Alert ID"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Update alert.
    
    Args:
        alert_data: Alert data
        alert_id: Alert ID
        db: Database session
        current_user: Current user
        
    Returns:
        AlertResponse: Updated alert
    """
    alert_repo = AlertRepository(db)
    
    # Get alert
    alert = alert_repo.get_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    # Check if user has access to alert
    if alert.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this alert"
        )
    
    # Update alert
    updated_alert = alert_repo.update(alert_id, alert_data.dict(exclude_unset=True))
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="update",
        entity_type="alert",
        entity_id=str(alert_id),
        details={"fields": list(alert_data.dict(exclude_unset=True).keys())}
    )
    
    return updated_alert

@router.post("/alerts/mark-all-read")
async def mark_all_alerts_read(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Mark all alerts as read.
    
    Args:
        db: Database session
        current_user: Current user
        
    Returns:
        Dict[str, Any]: Result
    """
    alert_repo = AlertRepository(db)
    
    # Mark all alerts as read
    count = alert_repo.mark_all_as_read(current_user.id)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="mark_all_read",
        entity_type="alert",
        entity_id=None,
        details={"count": count}
    )
    
    return {"message": f"Marked {count} alerts as read"}

# System status endpoints
@router.get("/system/status", response_model=List[SystemStatusResponse])
async def get_system_status(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get system status.
    
    Args:
        db: Database session
        current_user: Current user
        
    Returns:
        List[SystemStatusResponse]: System status
    """
    status_repo = SystemStatusRepository(db)
    
    # Get all system statuses
    statuses = status_repo.get_all(limit=100)
    
    return statuses

@router.get("/system/status/{component}", response_model=SystemStatusResponse)
async def get_component_status(
    component: str = Path(..., title="Component name"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get component status.
    
    Args:
        component: Component name
        db: Database session
        current_user: Current user
        
    Returns:
        SystemStatusResponse: Component status
    """
    status_repo = SystemStatusRepository(db)
    
    # Get component status
    status = status_repo.get_by_component(component)
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Status for component '{component}' not found"
        )
    
    return status

# Trading data endpoints
@router.get("/market-data", response_model=List[MarketDataResponse])
async def get_market_data(
    symbols: str = Query(..., title="Comma-separated list of symbols"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get market data.
    
    Args:
        symbols: Comma-separated list of symbols
        db: Database session
        current_user: Current user
        
    Returns:
        List[MarketDataResponse]: Market data
    """
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(",")]
    
    # Get market data from component service
    try:
        market_data = await component_service.get_market_data(symbol_list)
        return market_data
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting market data: {str(e)}"
        )

@router.get("/portfolio", response_model=PortfolioSummaryResponse)
async def get_portfolio(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get portfolio.
    
    Args:
        db: Database session
        current_user: Current user
        
    Returns:
        PortfolioSummaryResponse: Portfolio
    """
    # Get portfolio from component service
    try:
        portfolio = await component_service.get_portfolio()
        return portfolio
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting portfolio: {str(e)}"
        )

@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get trades.
    
    Args:
        limit: Maximum number of trades to return
        db: Database session
        current_user: Current user
        
    Returns:
        List[TradeResponse]: Trades
    """
    # Get trades from component service
    try:
        trades = await component_service.get_trades(limit)
        return trades
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting trades: {str(e)}"
        )

@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    status: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get orders.
    
    Args:
        status: Filter by status
        limit: Maximum number of orders to return
        db: Database session
        current_user: Current user
        
    Returns:
        List[OrderResponse]: Orders
    """
    # Get orders from component service
    try:
        orders = await component_service.get_orders(status, limit)
        return orders
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting orders: {str(e)}"
        )

@router.get("/strategies", response_model=List[StrategyResponse])
async def get_strategies(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get strategies.
    
    Args:
        db: Database session
        current_user: Current user
        
    Returns:
        List[StrategyResponse]: Strategies
    """
    # Get strategies from component service
    try:
        strategies = await component_service.get_strategies()
        return strategies
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting strategies: {str(e)}"
        )

@router.get("/signals", response_model=List[SignalResponse])
async def get_signals(
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get signals.
    
    Args:
        limit: Maximum number of signals to return
        db: Database session
        current_user: Current user
        
    Returns:
        List[SignalResponse]: Signals
    """
    # Get signals from component service
    try:
        signals = await component_service.get_signals(limit)
        return signals
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting signals: {str(e)}"
        )

@router.get("/risk-metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get risk metrics.
    
    Args:
        db: Database session
        current_user: Current user
        
    Returns:
        RiskMetricsResponse: Risk metrics
    """
    # Get risk metrics from component service
    try:
        risk_metrics = await component_service.get_risk_metrics()
        return risk_metrics
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting risk metrics: {str(e)}"
        )

@router.get("/agent-activity", response_model=List[AgentActivityResponse])
async def get_agent_activity(
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get agent activity.
    
    Args:
        limit: Maximum number of activities to return
        db: Database session
        current_user: Current user
        
    Returns:
        List[AgentActivityResponse]: Agent activities
    """
    # Get agent activity from component service
    try:
        activities = await component_service.get_agent_activity(limit)
        return activities
    except Exception as e:
        logger.error(f"Error getting agent activity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting agent activity: {str(e)}"
        )

@router.get("/performance-metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get performance metrics.
    
    Args:
        db: Database session
        current_user: Current user
        
    Returns:
        PerformanceMetricsResponse: Performance metrics
    """
    # Get performance metrics from component service
    try:
        performance_metrics = await component_service.get_performance_metrics()
        return performance_metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting performance metrics: {str(e)}"
        )
