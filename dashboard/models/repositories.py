"""
Database repositories for the dashboard interface.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_, not_

from ..models.database import (
    User, Role, Dashboard, Widget, Alert, AuditLog, 
    SystemStatus, ApiKey, SavedQuery, Notification,
    UserSession, ScheduledTask
)

logger = logging.getLogger(__name__)

class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, db: Session, model_class):
        """Initialize repository.
        
        Args:
            db: Database session
            model_class: SQLAlchemy model class
        """
        self.db = db
        self.model_class = model_class
    
    def get_by_id(self, id: int) -> Optional[Any]:
        """Get entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            Optional[Any]: Entity or None
        """
        return self.db.query(self.model_class).filter(self.model_class.id == id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[Any]:
        """Get all entities.
        
        Args:
            skip: Number of entities to skip
            limit: Maximum number of entities to return
            
        Returns:
            List[Any]: List of entities
        """
        return self.db.query(self.model_class).offset(skip).limit(limit).all()
    
    def create(self, data: Dict[str, Any]) -> Any:
        """Create entity.
        
        Args:
            data: Entity data
            
        Returns:
            Any: Created entity
        """
        entity = self.model_class(**data)
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity
    
    def update(self, id: int, data: Dict[str, Any]) -> Optional[Any]:
        """Update entity.
        
        Args:
            id: Entity ID
            data: Entity data
            
        Returns:
            Optional[Any]: Updated entity or None
        """
        entity = self.get_by_id(id)
        if not entity:
            return None
        
        for key, value in data.items():
            setattr(entity, key, value)
        
        self.db.commit()
        self.db.refresh(entity)
        return entity
    
    def delete(self, id: int) -> bool:
        """Delete entity.
        
        Args:
            id: Entity ID
            
        Returns:
            bool: Whether entity was deleted
        """
        entity = self.get_by_id(id)
        if not entity:
            return False
        
        self.db.delete(entity)
        self.db.commit()
        return True
    
    def count(self) -> int:
        """Count entities.
        
        Returns:
            int: Entity count
        """
        return self.db.query(self.model_class).count()

class UserRepository(BaseRepository):
    """User repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, User)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username.
        
        Args:
            username: Username
            
        Returns:
            Optional[User]: User or None
        """
        return self.db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email.
        
        Args:
            email: Email
            
        Returns:
            Optional[User]: User or None
        """
        return self.db.query(User).filter(User.email == email).first()
    
    def get_active_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get active users.
        
        Args:
            skip: Number of users to skip
            limit: Maximum number of users to return
            
        Returns:
            List[User]: List of active users
        """
        return self.db.query(User).filter(User.is_active == True).offset(skip).limit(limit).all()
    
    def get_by_role(self, role_name: str, skip: int = 0, limit: int = 100) -> List[User]:
        """Get users by role.
        
        Args:
            role_name: Role name
            skip: Number of users to skip
            limit: Maximum number of users to return
            
        Returns:
            List[User]: List of users with the specified role
        """
        return (
            self.db.query(User)
            .join(User.roles)
            .filter(Role.name == role_name)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def add_role(self, user_id: int, role_id: int) -> Optional[User]:
        """Add role to user.
        
        Args:
            user_id: User ID
            role_id: Role ID
            
        Returns:
            Optional[User]: Updated user or None
        """
        user = self.get_by_id(user_id)
        role = self.db.query(Role).filter(Role.id == role_id).first()
        
        if not user or not role:
            return None
        
        user.roles.append(role)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def remove_role(self, user_id: int, role_id: int) -> Optional[User]:
        """Remove role from user.
        
        Args:
            user_id: User ID
            role_id: Role ID
            
        Returns:
            Optional[User]: Updated user or None
        """
        user = self.get_by_id(user_id)
        role = self.db.query(Role).filter(Role.id == role_id).first()
        
        if not user or not role:
            return None
        
        user.roles.remove(role)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def update_last_login(self, user_id: int) -> Optional[User]:
        """Update user's last login timestamp.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[User]: Updated user or None
        """
        return self.update(user_id, {"last_login": datetime.utcnow()})

class RoleRepository(BaseRepository):
    """Role repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, Role)
    
    def get_by_name(self, name: str) -> Optional[Role]:
        """Get role by name.
        
        Args:
            name: Role name
            
        Returns:
            Optional[Role]: Role or None
        """
        return self.db.query(Role).filter(Role.name == name).first()
    
    def get_by_permission(self, permission: str, skip: int = 0, limit: int = 100) -> List[Role]:
        """Get roles by permission.
        
        Args:
            permission: Permission name
            skip: Number of roles to skip
            limit: Maximum number of roles to return
            
        Returns:
            List[Role]: List of roles with the specified permission
        """
        return (
            self.db.query(Role)
            .filter(Role.permissions.contains([permission]))
            .offset(skip)
            .limit(limit)
            .all()
        )

class DashboardRepository(BaseRepository):
    """Dashboard repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, Dashboard)
    
    def get_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Dashboard]:
        """Get dashboards by user.
        
        Args:
            user_id: User ID
            skip: Number of dashboards to skip
            limit: Maximum number of dashboards to return
            
        Returns:
            List[Dashboard]: List of dashboards for the specified user
        """
        return (
            self.db.query(Dashboard)
            .filter(Dashboard.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_default_for_user(self, user_id: int) -> Optional[Dashboard]:
        """Get default dashboard for user.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[Dashboard]: Default dashboard or None
        """
        return (
            self.db.query(Dashboard)
            .filter(Dashboard.user_id == user_id, Dashboard.is_default == True)
            .first()
        )
    
    def get_public_dashboards(self, skip: int = 0, limit: int = 100) -> List[Dashboard]:
        """Get public dashboards.
        
        Args:
            skip: Number of dashboards to skip
            limit: Maximum number of dashboards to return
            
        Returns:
            List[Dashboard]: List of public dashboards
        """
        return (
            self.db.query(Dashboard)
            .filter(Dashboard.is_public == True)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def set_default(self, dashboard_id: int, user_id: int) -> Optional[Dashboard]:
        """Set dashboard as default for user.
        
        Args:
            dashboard_id: Dashboard ID
            user_id: User ID
            
        Returns:
            Optional[Dashboard]: Updated dashboard or None
        """
        # Clear current default
        current_default = self.get_default_for_user(user_id)
        if current_default:
            current_default.is_default = False
            self.db.commit()
        
        # Set new default
        dashboard = self.get_by_id(dashboard_id)
        if not dashboard or dashboard.user_id != user_id:
            return None
        
        dashboard.is_default = True
        self.db.commit()
        self.db.refresh(dashboard)
        return dashboard

class WidgetRepository(BaseRepository):
    """Widget repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, Widget)
    
    def get_by_dashboard(self, dashboard_id: int, skip: int = 0, limit: int = 100) -> List[Widget]:
        """Get widgets by dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            skip: Number of widgets to skip
            limit: Maximum number of widgets to return
            
        Returns:
            List[Widget]: List of widgets for the specified dashboard
        """
        return (
            self.db.query(Widget)
            .filter(Widget.dashboard_id == dashboard_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_type(self, widget_type: str, skip: int = 0, limit: int = 100) -> List[Widget]:
        """Get widgets by type.
        
        Args:
            widget_type: Widget type
            skip: Number of widgets to skip
            limit: Maximum number of widgets to return
            
        Returns:
            List[Widget]: List of widgets of the specified type
        """
        return (
            self.db.query(Widget)
            .filter(Widget.widget_type == widget_type)
            .offset(skip)
            .limit(limit)
            .all()
        )

class AlertRepository(BaseRepository):
    """Alert repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, Alert)
    
    def get_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Alert]:
        """Get alerts by user.
        
        Args:
            user_id: User ID
            skip: Number of alerts to skip
            limit: Maximum number of alerts to return
            
        Returns:
            List[Alert]: List of alerts for the specified user
        """
        return (
            self.db.query(Alert)
            .filter(Alert.user_id == user_id)
            .order_by(desc(Alert.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_unread_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Alert]:
        """Get unread alerts by user.
        
        Args:
            user_id: User ID
            skip: Number of alerts to skip
            limit: Maximum number of alerts to return
            
        Returns:
            List[Alert]: List of unread alerts for the specified user
        """
        return (
            self.db.query(Alert)
            .filter(Alert.user_id == user_id, Alert.is_read == False)
            .order_by(desc(Alert.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_severity(self, severity: str, skip: int = 0, limit: int = 100) -> List[Alert]:
        """Get alerts by severity.
        
        Args:
            severity: Alert severity
            skip: Number of alerts to skip
            limit: Maximum number of alerts to return
            
        Returns:
            List[Alert]: List of alerts with the specified severity
        """
        return (
            self.db.query(Alert)
            .filter(Alert.severity == severity)
            .order_by(desc(Alert.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def mark_as_read(self, alert_id: int) -> Optional[Alert]:
        """Mark alert as read.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            Optional[Alert]: Updated alert or None
        """
        return self.update(alert_id, {"is_read": True})
    
    def mark_as_acknowledged(self, alert_id: int) -> Optional[Alert]:
        """Mark alert as acknowledged.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            Optional[Alert]: Updated alert or None
        """
        return self.update(
            alert_id, 
            {
                "is_acknowledged": True, 
                "acknowledged_at": datetime.utcnow()
            }
        )
    
    def mark_all_as_read(self, user_id: int) -> int:
        """Mark all alerts as read for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            int: Number of updated alerts
        """
        result = (
            self.db.query(Alert)
            .filter(Alert.user_id == user_id, Alert.is_read == False)
            .update({"is_read": True})
        )
        self.db.commit()
        return result

class AuditLogRepository(BaseRepository):
    """Audit log repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, AuditLog)
    
    def get_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """Get audit logs by user.
        
        Args:
            user_id: User ID
            skip: Number of audit logs to skip
            limit: Maximum number of audit logs to return
            
        Returns:
            List[AuditLog]: List of audit logs for the specified user
        """
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.user_id == user_id)
            .order_by(desc(AuditLog.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_action(self, action: str, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """Get audit logs by action.
        
        Args:
            action: Action
            skip: Number of audit logs to skip
            limit: Maximum number of audit logs to return
            
        Returns:
            List[AuditLog]: List of audit logs for the specified action
        """
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.action == action)
            .order_by(desc(AuditLog.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_entity(self, entity_type: str, entity_id: str, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """Get audit logs by entity.
        
        Args:
            entity_type: Entity type
            entity_id: Entity ID
            skip: Number of audit logs to skip
            limit: Maximum number of audit logs to return
            
        Returns:
            List[AuditLog]: List of audit logs for the specified entity
        """
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.entity_type == entity_type, AuditLog.entity_id == entity_id)
            .order_by(desc(AuditLog.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_date_range(self, start_date: datetime, end_date: datetime, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """Get audit logs by date range.
        
        Args:
            start_date: Start date
            end_date: End date
            skip: Number of audit logs to skip
            limit: Maximum number of audit logs to return
            
        Returns:
            List[AuditLog]: List of audit logs in the specified date range
        """
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.created_at >= start_date, AuditLog.created_at <= end_date)
            .order_by(desc(AuditLog.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

class SystemStatusRepository(BaseRepository):
    """System status repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, SystemStatus)
    
    def get_by_component(self, component: str) -> Optional[SystemStatus]:
        """Get system status by component.
        
        Args:
            component: Component name
            
        Returns:
            Optional[SystemStatus]: System status or None
        """
        return self.db.query(SystemStatus).filter(SystemStatus.component == component).first()
    
    def get_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List[SystemStatus]:
        """Get system statuses by status.
        
        Args:
            status: Status
            skip: Number of system statuses to skip
            limit: Maximum number of system statuses to return
            
        Returns:
            List[SystemStatus]: List of system statuses with the specified status
        """
        return (
            self.db.query(SystemStatus)
            .filter(SystemStatus.status == status)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update_status(self, component: str, status: str, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Optional[SystemStatus]:
        """Update system status.
        
        Args:
            component: Component name
            status: Status
            message: Message
            details: Details
            
        Returns:
            Optional[SystemStatus]: Updated system status or None
        """
        system_status = self.get_by_component(component)
        
        if not system_status:
            # Create new system status
            return self.create({
                "component": component,
                "status": status,
                "message": message,
                "details": details or {},
                "last_check": datetime.utcnow()
            })
        
        # Update existing system status
        update_data = {
            "status": status,
            "last_check": datetime.utcnow()
        }
        
        if message is not None:
            update_data["message"] = message
        
        if details is not None:
            update_data["details"] = details
        
        return self.update(system_status.id, update_data)

class ApiKeyRepository(BaseRepository):
    """API key repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, ApiKey)
    
    def get_by_key(self, key: str) -> Optional[ApiKey]:
        """Get API key by key.
        
        Args:
            key: API key
            
        Returns:
            Optional[ApiKey]: API key or None
        """
        return self.db.query(ApiKey).filter(ApiKey.key == key).first()
    
    def get_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[ApiKey]:
        """Get API keys by user.
        
        Args:
            user_id: User ID
            skip: Number of API keys to skip
            limit: Maximum number of API keys to return
            
        Returns:
            List[ApiKey]: List of API keys for the specified user
        """
        return (
            self.db.query(ApiKey)
            .filter(ApiKey.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_active_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[ApiKey]:
        """Get active API keys by user.
        
        Args:
            user_id: User ID
            skip: Number of API keys to skip
            limit: Maximum number of API keys to return
            
        Returns:
            List[ApiKey]: List of active API keys for the specified user
        """
        return (
            self.db.query(ApiKey)
            .filter(
                ApiKey.user_id == user_id,
                ApiKey.is_active == True,
                or_(
                    ApiKey.expires_at == None,
                    ApiKey.expires_at > datetime.utcnow()
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update_last_used(self, key: str) -> Optional[ApiKey]:
        """Update API key's last used timestamp.
        
        Args:
            key: API key
            
        Returns:
            Optional[ApiKey]: Updated API key or None
        """
        api_key = self.get_by_key(key)
        if not api_key:
            return None
        
        return self.update(api_key.id, {"last_used": datetime.utcnow()})
    
    def deactivate(self, key_id: int) -> Optional[ApiKey]:
        """Deactivate API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            Optional[ApiKey]: Updated API key or None
        """
        return self.update(key_id, {"is_active": False})

class SavedQueryRepository(BaseRepository):
    """Saved query repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, SavedQuery)
    
    def get_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[SavedQuery]:
        """Get saved queries by user.
        
        Args:
            user_id: User ID
            skip: Number of saved queries to skip
            limit: Maximum number of saved queries to return
            
        Returns:
            List[SavedQuery]: List of saved queries for the specified user
        """
        return (
            self.db.query(SavedQuery)
            .filter(SavedQuery.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_type(self, query_type: str, skip: int = 0, limit: int = 100) -> List[SavedQuery]:
        """Get saved queries by type.
        
        Args:
            query_type: Query type
            skip: Number of saved queries to skip
            limit: Maximum number of saved queries to return
            
        Returns:
            List[SavedQuery]: List of saved queries of the specified type
        """
        return (
            self.db.query(SavedQuery)
            .filter(SavedQuery.query_type == query_type)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_public_queries(self, skip: int = 0, limit: int = 100) -> List[SavedQuery]:
        """Get public saved queries.
        
        Args:
            skip: Number of saved queries to skip
            limit: Maximum number of saved queries to return
            
        Returns:
            List[SavedQuery]: List of public saved queries
        """
        return (
            self.db.query(SavedQuery)
            .filter(SavedQuery.is_public == True)
            .offset(skip)
            .limit(limit)
            .all()
        )

class NotificationRepository(BaseRepository):
    """Notification repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, Notification)
    
    def get_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Notification]:
        """Get notifications by user.
        
        Args:
            user_id: User ID
            skip: Number of notifications to skip
            limit: Maximum number of notifications to return
            
        Returns:
            List[Notification]: List of notifications for the specified user
        """
        return (
            self.db.query(Notification)
            .filter(Notification.user_id == user_id)
            .order_by(desc(Notification.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_unread_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Notification]:
        """Get unread notifications by user.
        
        Args:
            user_id: User ID
            skip: Number of notifications to skip
            limit: Maximum number of notifications to return
            
        Returns:
            List[Notification]: List of unread notifications for the specified user
        """
        return (
            self.db.query(Notification)
            .filter(Notification.user_id == user_id, Notification.is_read == False)
            .order_by(desc(Notification.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_type(self, notification_type: str, skip: int = 0, limit: int = 100) -> List[Notification]:
        """Get notifications by type.
        
        Args:
            notification_type: Notification type
            skip: Number of notifications to skip
            limit: Maximum number of notifications to return
            
        Returns:
            List[Notification]: List of notifications of the specified type
        """
        return (
            self.db.query(Notification)
            .filter(Notification.notification_type == notification_type)
            .order_by(desc(Notification.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def mark_as_read(self, notification_id: int) -> Optional[Notification]:
        """Mark notification as read.
        
        Args:
            notification_id: Notification ID
            
        Returns:
            Optional[Notification]: Updated notification or None
        """
        return self.update(notification_id, {"is_read": True})
    
    def mark_all_as_read(self, user_id: int) -> int:
        """Mark all notifications as read for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            int: Number of updated notifications
        """
        result = (
            self.db.query(Notification)
            .filter(Notification.user_id == user_id, Notification.is_read == False)
            .update({"is_read": True})
        )
        self.db.commit()
        return result

class UserSessionRepository(BaseRepository):
    """User session repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, UserSession)
    
    def get_by_session_id(self, session_id: str) -> Optional[UserSession]:
        """Get user session by session ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Optional[UserSession]: User session or None
        """
        return self.db.query(UserSession).filter(UserSession.session_id == session_id).first()
    
    def get_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[UserSession]:
        """Get user sessions by user.
        
        Args:
            user_id: User ID
            skip: Number of user sessions to skip
            limit: Maximum number of user sessions to return
            
        Returns:
            List[UserSession]: List of user sessions for the specified user
        """
        return (
            self.db.query(UserSession)
            .filter(UserSession.user_id == user_id)
            .order_by(desc(UserSession.last_activity))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_active_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[UserSession]:
        """Get active user sessions by user.
        
        Args:
            user_id: User ID
            skip: Number of user sessions to skip
            limit: Maximum number of user sessions to return
            
        Returns:
            List[UserSession]: List of active user sessions for the specified user
        """
        return (
            self.db.query(UserSession)
            .filter(
                UserSession.user_id == user_id,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            )
            .order_by(desc(UserSession.last_activity))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update_last_activity(self, session_id: str) -> Optional[UserSession]:
        """Update user session's last activity timestamp.
        
        Args:
            session_id: Session ID
            
        Returns:
            Optional[UserSession]: Updated user session or None
        """
        session = self.get_by_session_id(session_id)
        if not session:
            return None
        
        return self.update(session.id, {"last_activity": datetime.utcnow()})
    
    def deactivate(self, session_id: str) -> Optional[UserSession]:
        """Deactivate user session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Optional[UserSession]: Updated user session or None
        """
        session = self.get_by_session_id(session_id)
        if not session:
            return None
        
        return self.update(session.id, {"is_active": False})
    
    def deactivate_all_for_user(self, user_id: int) -> int:
        """Deactivate all user sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            int: Number of updated user sessions
        """
        result = (
            self.db.query(UserSession)
            .filter(UserSession.user_id == user_id, UserSession.is_active == True)
            .update({"is_active": False})
        )
        self.db.commit()
        return result
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired user sessions.
        
        Returns:
            int: Number of updated user sessions
        """
        result = (
            self.db.query(UserSession)
            .filter(
                UserSession.is_active == True,
                UserSession.expires_at <= datetime.utcnow()
            )
            .update({"is_active": False})
        )
        self.db.commit()
        return result

class ScheduledTaskRepository(BaseRepository):
    """Scheduled task repository."""
    
    def __init__(self, db: Session):
        """Initialize repository.
        
        Args:
            db: Database session
        """
        super().__init__(db, ScheduledTask)
    
    def get_by_type(self, task_type: str, skip: int = 0, limit: int = 100) -> List[ScheduledTask]:
        """Get scheduled tasks by type.
        
        Args:
            task_type: Task type
            skip: Number of scheduled tasks to skip
            limit: Maximum number of scheduled tasks to return
            
        Returns:
            List[ScheduledTask]: List of scheduled tasks of the specified type
        """
        return (
            self.db.query(ScheduledTask)
            .filter(ScheduledTask.task_type == task_type)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_active_tasks(self, skip: int = 0, limit: int = 100) -> List[ScheduledTask]:
        """Get active scheduled tasks.
        
        Args:
            skip: Number of scheduled tasks to skip
            limit: Maximum number of scheduled tasks to return
            
        Returns:
            List[ScheduledTask]: List of active scheduled tasks
        """
        return (
            self.db.query(ScheduledTask)
            .filter(ScheduledTask.is_active == True)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_due_tasks(self) -> List[ScheduledTask]:
        """Get due scheduled tasks.
        
        Returns:
            List[ScheduledTask]: List of due scheduled tasks
        """
        return (
            self.db.query(ScheduledTask)
            .filter(
                ScheduledTask.is_active == True,
                ScheduledTask.next_run <= datetime.utcnow()
            )
            .all()
        )
    
    def update_task_execution(self, task_id: int, next_run: datetime) -> Optional[ScheduledTask]:
        """Update scheduled task execution.
        
        Args:
            task_id: Task ID
            next_run: Next run timestamp
            
        Returns:
            Optional[ScheduledTask]: Updated scheduled task or None
        """
        return self.update(
            task_id,
            {
                "last_run": datetime.utcnow(),
                "next_run": next_run
            }
        )
