"""
Database models for the dashboard interface.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# User-Role association table
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('role_id', Integer, ForeignKey('roles.id'))
)

class User(Base):
    """User model."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    preferences = Column(JSON, default=dict)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    dashboards = relationship("Dashboard", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

class Role(Base):
    """Role model."""
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(String(200))
    permissions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")

class Dashboard(Base):
    """Dashboard model."""
    __tablename__ = 'dashboards'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(200))
    layout = Column(JSON, default=dict)
    is_default = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="dashboards")
    widgets = relationship("Widget", back_populates="dashboard")

class Widget(Base):
    """Widget model."""
    __tablename__ = 'widgets'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    widget_type = Column(String(50), nullable=False)
    config = Column(JSON, default=dict)
    position = Column(JSON, default=dict)
    dashboard_id = Column(Integer, ForeignKey('dashboards.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dashboard = relationship("Dashboard", back_populates="widgets")

class Alert(Base):
    """Alert model."""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), nullable=False)
    message = Column(Text)
    severity = Column(String(20), nullable=False)  # info, warning, error, critical
    source = Column(String(50), nullable=False)
    is_read = Column(Boolean, default=False)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="alerts")

class AuditLog(Base):
    """Audit log model."""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(50), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(50))
    details = Column(JSON, default=dict)
    user_id = Column(Integer, ForeignKey('users.id'))
    ip_address = Column(String(50))
    user_agent = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")

class SystemStatus(Base):
    """System status model."""
    __tablename__ = 'system_status'
    
    id = Column(Integer, primary_key=True, index=True)
    component = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)  # online, offline, degraded
    message = Column(String(200))
    details = Column(JSON, default=dict)
    last_check = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ApiKey(Base):
    """API key model."""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    permissions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used = Column(DateTime)

class SavedQuery(Base):
    """Saved query model."""
    __tablename__ = 'saved_queries'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(200))
    query_type = Column(String(50), nullable=False)
    query_params = Column(JSON, default=dict)
    user_id = Column(Integer, ForeignKey('users.id'))
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Notification(Base):
    """Notification model."""
    __tablename__ = 'notifications'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100), nullable=False)
    message = Column(Text)
    notification_type = Column(String(50), nullable=False)
    is_read = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)

class UserSession(Base):
    """User session model."""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    ip_address = Column(String(50))
    user_agent = Column(String(200))
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ScheduledTask(Base):
    """Scheduled task model."""
    __tablename__ = 'scheduled_tasks'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)
    schedule = Column(String(100), nullable=False)  # Cron expression
    parameters = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'))
