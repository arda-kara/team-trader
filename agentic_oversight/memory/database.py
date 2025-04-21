"""
Database models and utilities for the agentic oversight system.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, ForeignKey, Text, Enum, JSON, LargeBinary
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.pool import QueuePool

from ..config.settings import settings
from ..core.models import (
    AgentType, AgentStatus, TaskStatus, TaskPriority, AlertLevel,
    DecisionConfidence, ApprovalStatus, ReasoningFramework,
    CoordinationStrategy, ConflictResolutionStrategy, MemoryType
)

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.memory.relational_db_connection,
    pool_size=5,
    max_overflow=10,
    echo=False,
    poolclass=QueuePool
)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# Create base class for models
Base = declarative_base()

class AgentModel(Base):
    """Agent database model."""
    __tablename__ = "agents"
    
    id = Column(String(32), primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(Enum(AgentType), nullable=False)
    status = Column(Enum(AgentStatus), nullable=False, default=AgentStatus.IDLE)
    capabilities = Column(JSON, default=[])
    config = Column(JSON, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_active = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, default={})
    
    # Relationships
    tasks = relationship("TaskModel", back_populates="agent", foreign_keys="TaskModel.assigned_to")
    memories = relationship("MemoryModel", back_populates="agent")
    reasonings = relationship("ReasoningModel", back_populates="agent")
    decisions = relationship("DecisionModel", back_populates="agent")
    alerts = relationship("AlertModel", back_populates="agent")
    explanations = relationship("ExplanationModel", back_populates="agent")

class TaskModel(Base):
    """Task database model."""
    __tablename__ = "tasks"
    
    id = Column(String(32), primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    type = Column(String(64), nullable=False)
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING)
    priority = Column(Enum(TaskPriority), nullable=False, default=TaskPriority.MEDIUM)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    assigned_to = Column(String(32), ForeignKey("agents.id"))
    created_by = Column(String(32))
    due_by = Column(DateTime)
    completed_at = Column(DateTime)
    input_data = Column(JSON, default={})
    output_data = Column(JSON, default={})
    dependencies = Column(JSON, default=[])
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    
    # Relationships
    agent = relationship("AgentModel", back_populates="tasks", foreign_keys=[assigned_to])
    reasonings = relationship("ReasoningModel", back_populates="task")
    decisions = relationship("DecisionModel", back_populates="task")

class MemoryModel(Base):
    """Memory database model."""
    __tablename__ = "memories"
    
    id = Column(String(32), primary_key=True)
    agent_id = Column(String(32), ForeignKey("agents.id"), nullable=False)
    type = Column(Enum(MemoryType), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary)  # Store embedding as binary
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime)
    context = Column(JSON, default={})
    importance = Column(Float, default=0.5)
    metadata = Column(JSON, default={})
    
    # Relationships
    agent = relationship("AgentModel", back_populates="memories")

class ReasoningModel(Base):
    """Reasoning database model."""
    __tablename__ = "reasonings"
    
    id = Column(String(32), primary_key=True)
    agent_id = Column(String(32), ForeignKey("agents.id"), nullable=False)
    task_id = Column(String(32), ForeignKey("tasks.id"), nullable=False)
    framework = Column(Enum(ReasoningFramework), nullable=False)
    steps = Column(JSON, default=[])
    conclusion = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    alternatives = Column(JSON, default=[])
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    duration_ms = Column(Integer, nullable=False)
    metadata = Column(JSON, default={})
    
    # Relationships
    agent = relationship("AgentModel", back_populates="reasonings")
    task = relationship("TaskModel", back_populates="reasonings")
    decisions = relationship("DecisionModel", back_populates="reasoning")

class DecisionModel(Base):
    """Decision database model."""
    __tablename__ = "decisions"
    
    id = Column(String(32), primary_key=True)
    agent_id = Column(String(32), ForeignKey("agents.id"), nullable=False)
    task_id = Column(String(32), ForeignKey("tasks.id"), nullable=False)
    reasoning_id = Column(String(32), ForeignKey("reasonings.id"), nullable=False)
    decision = Column(Text, nullable=False)
    confidence = Column(Enum(DecisionConfidence), nullable=False)
    justification = Column(Text, nullable=False)
    alternatives = Column(JSON, default=[])
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    requires_approval = Column(Boolean, default=False)
    approval_status = Column(Enum(ApprovalStatus))
    approved_by = Column(String(32))
    approved_at = Column(DateTime)
    metadata = Column(JSON, default={})
    
    # Relationships
    agent = relationship("AgentModel", back_populates="decisions")
    task = relationship("TaskModel", back_populates="decisions")
    reasoning = relationship("ReasoningModel", back_populates="decisions")
    explanations = relationship("ExplanationModel", back_populates="decision")

class AlertModel(Base):
    """Alert database model."""
    __tablename__ = "alerts"
    
    id = Column(String(32), primary_key=True)
    agent_id = Column(String(32), ForeignKey("agents.id"), nullable=False)
    level = Column(Enum(AlertLevel), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(32))
    acknowledged_at = Column(DateTime)
    related_entity_type = Column(String(64))
    related_entity_id = Column(String(32))
    actions = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    
    # Relationships
    agent = relationship("AgentModel", back_populates="alerts")

class ExplanationModel(Base):
    """Explanation database model."""
    __tablename__ = "explanations"
    
    id = Column(String(32), primary_key=True)
    agent_id = Column(String(32), ForeignKey("agents.id"), nullable=False)
    decision_id = Column(String(32), ForeignKey("decisions.id"), nullable=False)
    explanation = Column(Text, nullable=False)
    detail_level = Column(String(32), nullable=False)
    audience = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    factors = Column(JSON, default=[])
    context = Column(JSON, default={})
    metadata = Column(JSON, default={})
    
    # Relationships
    agent = relationship("AgentModel", back_populates="explanations")
    decision = relationship("DecisionModel", back_populates="explanations")

class CoordinationModel(Base):
    """Coordination database model."""
    __tablename__ = "coordinations"
    
    id = Column(String(32), primary_key=True)
    strategy = Column(Enum(CoordinationStrategy), nullable=False)
    agents = Column(JSON, nullable=False)
    tasks = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    result = Column(JSON, default={})
    conflicts = Column(JSON, default=[])
    resolution_strategy = Column(Enum(ConflictResolutionStrategy), nullable=False)
    metadata = Column(JSON, default={})

class HumanInteractionModel(Base):
    """Human interaction database model."""
    __tablename__ = "human_interactions"
    
    id = Column(String(32), primary_key=True)
    type = Column(String(64), nullable=False)
    user_id = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    responded_at = Column(DateTime)
    response = Column(Text)
    related_entity_type = Column(String(64))
    related_entity_id = Column(String(32))
    metadata = Column(JSON, default={})

class SystemStatusModel(Base):
    """System status database model."""
    __tablename__ = "system_statuses"
    
    id = Column(String(32), primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    components = Column(JSON, nullable=False)
    agents = Column(JSON, nullable=False)
    tasks = Column(JSON, nullable=False)
    alerts = Column(JSON, nullable=False)
    performance_metrics = Column(JSON, nullable=False)
    overall_status = Column(String(32), nullable=False)
    metadata = Column(JSON, default={})

def init_db():
    """Initialize database by creating all tables."""
    Base.metadata.create_all(engine)
    logger.info("Database tables created")

def get_session():
    """Get a database session.
    
    Returns:
        Session: Database session
    """
    return Session()

def close_session(session):
    """Close a database session.
    
    Args:
        session: Database session
    """
    session.close()

def cleanup_old_data():
    """Clean up old data based on retention settings."""
    session = get_session()
    try:
        # Calculate cutoff dates
        log_cutoff = datetime.utcnow() - timedelta(days=settings.monitoring.log_retention_days)
        metrics_cutoff = datetime.utcnow() - timedelta(days=settings.monitoring.metrics_retention_days)
        
        # Delete old system statuses
        session.query(SystemStatusModel).filter(SystemStatusModel.timestamp < metrics_cutoff).delete()
        
        # Delete old alerts
        session.query(AlertModel).filter(
            AlertModel.acknowledged == True,
            AlertModel.created_at < log_cutoff
        ).delete()
        
        # Delete old human interactions
        session.query(HumanInteractionModel).filter(
            HumanInteractionModel.responded_at != None,
            HumanInteractionModel.created_at < log_cutoff
        ).delete()
        
        # Delete old memories that have expired
        session.query(MemoryModel).filter(
            MemoryModel.expires_at < datetime.utcnow()
        ).delete()
        
        session.commit()
        logger.info("Old data cleaned up")
    except Exception as e:
        session.rollback()
        logger.error(f"Error cleaning up old data: {e}")
    finally:
        close_session(session)

# Repository classes
class AgentRepository:
    """Repository for agent operations."""
    
    @staticmethod
    def create(agent_data):
        """Create a new agent.
        
        Args:
            agent_data: Agent data
            
        Returns:
            AgentModel: Created agent
        """
        session = get_session()
        try:
            agent = AgentModel(**agent_data)
            session.add(agent)
            session.commit()
            return agent
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating agent: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(agent_id):
        """Get agent by ID.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            AgentModel: Agent or None
        """
        session = get_session()
        try:
            return session.query(AgentModel).filter(AgentModel.id == agent_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_type(agent_type):
        """Get agents by type.
        
        Args:
            agent_type: Agent type
            
        Returns:
            List[AgentModel]: List of agents
        """
        session = get_session()
        try:
            return session.query(AgentModel).filter(AgentModel.type == agent_type).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_status(status):
        """Get agents by status.
        
        Args:
            status: Agent status
            
        Returns:
            List[AgentModel]: List of agents
        """
        session = get_session()
        try:
            return session.query(AgentModel).filter(AgentModel.status == status).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_all():
        """Get all agents.
        
        Returns:
            List[AgentModel]: List of all agents
        """
        session = get_session()
        try:
            return session.query(AgentModel).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(agent_id, agent_data):
        """Update agent.
        
        Args:
            agent_id: Agent ID
            agent_data: Agent data to update
            
        Returns:
            AgentModel: Updated agent or None
        """
        session = get_session()
        try:
            agent = session.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent:
                for key, value in agent_data.items():
                    setattr(agent, key, value)
                session.commit()
            return agent
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating agent: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def delete(agent_id):
        """Delete agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            bool: Success status
        """
        session = get_session()
        try:
            agent = session.query(AgentModel).filter(AgentModel.id == agent_id).first()
            if agent:
                session.delete(agent)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting agent: {e}")
            raise
        finally:
            close_session(session)

class TaskRepository:
    """Repository for task operations."""
    
    @staticmethod
    def create(task_data):
        """Create a new task.
        
        Args:
            task_data: Task data
            
        Returns:
            TaskModel: Created task
        """
        session = get_session()
        try:
            task = TaskModel(**task_data)
            session.add(task)
            session.commit()
            return task
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating task: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(task_id):
        """Get task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            TaskModel: Task or None
        """
        session = get_session()
        try:
            return session.query(TaskModel).filter(TaskModel.id == task_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_status(status):
        """Get tasks by status.
        
        Args:
            status: Task status
            
        Returns:
            List[TaskModel]: List of tasks
        """
        session = get_session()
        try:
            return session.query(TaskModel).filter(TaskModel.status == status).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_agent(agent_id):
        """Get tasks by agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List[TaskModel]: List of tasks
        """
        session = get_session()
        try:
            return session.query(TaskModel).filter(TaskModel.assigned_to == agent_id).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_priority(priority):
        """Get tasks by priority.
        
        Args:
            priority: Task priority
            
        Returns:
            List[TaskModel]: List of tasks
        """
        session = get_session()
        try:
            return session.query(TaskModel).filter(TaskModel.priority == priority).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_pending_tasks():
        """Get pending tasks.
        
        Returns:
            List[TaskModel]: List of pending tasks
        """
        session = get_session()
        try:
            return session.query(TaskModel).filter(
                TaskModel.status.in_([TaskStatus.PENDING, TaskStatus.ASSIGNED])
            ).order_by(TaskModel.priority.desc(), TaskModel.created_at.asc()).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(task_id, task_data):
        """Update task.
        
        Args:
            task_id: Task ID
            task_data: Task data to update
            
        Returns:
            TaskModel: Updated task or None
        """
        session = get_session()
        try:
            task = session.query(TaskModel).filter(TaskModel.id == task_id).first()
            if task:
                for key, value in task_data.items():
                    setattr(task, key, value)
                session.commit()
            return task
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating task: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def delete(task_id):
        """Delete task.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: Success status
        """
        session = get_session()
        try:
            task = session.query(TaskModel).filter(TaskModel.id == task_id).first()
            if task:
                session.delete(task)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting task: {e}")
            raise
        finally:
            close_session(session)

class MemoryRepository:
    """Repository for memory operations."""
    
    @staticmethod
    def create(memory_data):
        """Create a new memory.
        
        Args:
            memory_data: Memory data
            
        Returns:
            MemoryModel: Created memory
        """
        session = get_session()
        try:
            memory = MemoryModel(**memory_data)
            session.add(memory)
            session.commit()
            return memory
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating memory: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(memory_id):
        """Get memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            MemoryModel: Memory or None
        """
        session = get_session()
        try:
            return session.query(MemoryModel).filter(MemoryModel.id == memory_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_agent(agent_id, memory_type=None, limit=100):
        """Get memories by agent.
        
        Args:
            agent_id: Agent ID
            memory_type: Memory type
            limit: Maximum number of memories to return
            
        Returns:
            List[MemoryModel]: List of memories
        """
        session = get_session()
        try:
            query = session.query(MemoryModel).filter(MemoryModel.agent_id == agent_id)
            
            if memory_type:
                query = query.filter(MemoryModel.type == memory_type)
            
            return query.order_by(MemoryModel.created_at.desc()).limit(limit).all()
        finally:
            close_session(session)
    
    @staticmethod
    def delete_expired():
        """Delete expired memories.
        
        Returns:
            int: Number of deleted memories
        """
        session = get_session()
        try:
            count = session.query(MemoryModel).filter(
                MemoryModel.expires_at < datetime.utcnow()
            ).delete()
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting expired memories: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def delete(memory_id):
        """Delete memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            bool: Success status
        """
        session = get_session()
        try:
            memory = session.query(MemoryModel).filter(MemoryModel.id == memory_id).first()
            if memory:
                session.delete(memory)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting memory: {e}")
            raise
        finally:
            close_session(session)

class DecisionRepository:
    """Repository for decision operations."""
    
    @staticmethod
    def create(decision_data):
        """Create a new decision.
        
        Args:
            decision_data: Decision data
            
        Returns:
            DecisionModel: Created decision
        """
        session = get_session()
        try:
            decision = DecisionModel(**decision_data)
            session.add(decision)
            session.commit()
            return decision
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating decision: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(decision_id):
        """Get decision by ID.
        
        Args:
            decision_id: Decision ID
            
        Returns:
            DecisionModel: Decision or None
        """
        session = get_session()
        try:
            return session.query(DecisionModel).filter(DecisionModel.id == decision_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_task(task_id):
        """Get decisions by task.
        
        Args:
            task_id: Task ID
            
        Returns:
            List[DecisionModel]: List of decisions
        """
        session = get_session()
        try:
            return session.query(DecisionModel).filter(DecisionModel.task_id == task_id).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_agent(agent_id, limit=100):
        """Get decisions by agent.
        
        Args:
            agent_id: Agent ID
            limit: Maximum number of decisions to return
            
        Returns:
            List[DecisionModel]: List of decisions
        """
        session = get_session()
        try:
            return session.query(DecisionModel).filter(
                DecisionModel.agent_id == agent_id
            ).order_by(DecisionModel.created_at.desc()).limit(limit).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_pending_approvals():
        """Get decisions pending approval.
        
        Returns:
            List[DecisionModel]: List of decisions pending approval
        """
        session = get_session()
        try:
            return session.query(DecisionModel).filter(
                DecisionModel.requires_approval == True,
                DecisionModel.approval_status == ApprovalStatus.PENDING
            ).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(decision_id, decision_data):
        """Update decision.
        
        Args:
            decision_id: Decision ID
            decision_data: Decision data to update
            
        Returns:
            DecisionModel: Updated decision or None
        """
        session = get_session()
        try:
            decision = session.query(DecisionModel).filter(DecisionModel.id == decision_id).first()
            if decision:
                for key, value in decision_data.items():
                    setattr(decision, key, value)
                session.commit()
            return decision
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating decision: {e}")
            raise
        finally:
            close_session(session)

class AlertRepository:
    """Repository for alert operations."""
    
    @staticmethod
    def create(alert_data):
        """Create a new alert.
        
        Args:
            alert_data: Alert data
            
        Returns:
            AlertModel: Created alert
        """
        session = get_session()
        try:
            alert = AlertModel(**alert_data)
            session.add(alert)
            session.commit()
            return alert
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating alert: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(alert_id):
        """Get alert by ID.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            AlertModel: Alert or None
        """
        session = get_session()
        try:
            return session.query(AlertModel).filter(AlertModel.id == alert_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_active_alerts(level=None):
        """Get active alerts.
        
        Args:
            level: Alert level
            
        Returns:
            List[AlertModel]: List of active alerts
        """
        session = get_session()
        try:
            query = session.query(AlertModel).filter(
                AlertModel.acknowledged == False,
                (AlertModel.expires_at == None) | (AlertModel.expires_at > datetime.utcnow())
            )
            
            if level:
                query = query.filter(AlertModel.level == level)
            
            return query.order_by(AlertModel.created_at.desc()).all()
        finally:
            close_session(session)
    
    @staticmethod
    def acknowledge_alert(alert_id, user_id):
        """Acknowledge alert.
        
        Args:
            alert_id: Alert ID
            user_id: User ID
            
        Returns:
            AlertModel: Updated alert or None
        """
        session = get_session()
        try:
            alert = session.query(AlertModel).filter(AlertModel.id == alert_id).first()
            if alert:
                alert.acknowledged = True
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.utcnow()
                session.commit()
            return alert
        except Exception as e:
            session.rollback()
            logger.error(f"Error acknowledging alert: {e}")
            raise
        finally:
            close_session(session)

class ExplanationRepository:
    """Repository for explanation operations."""
    
    @staticmethod
    def create(explanation_data):
        """Create a new explanation.
        
        Args:
            explanation_data: Explanation data
            
        Returns:
            ExplanationModel: Created explanation
        """
        session = get_session()
        try:
            explanation = ExplanationModel(**explanation_data)
            session.add(explanation)
            session.commit()
            return explanation
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating explanation: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(explanation_id):
        """Get explanation by ID.
        
        Args:
            explanation_id: Explanation ID
            
        Returns:
            ExplanationModel: Explanation or None
        """
        session = get_session()
        try:
            return session.query(ExplanationModel).filter(ExplanationModel.id == explanation_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_decision(decision_id):
        """Get explanations by decision.
        
        Args:
            decision_id: Decision ID
            
        Returns:
            List[ExplanationModel]: List of explanations
        """
        session = get_session()
        try:
            return session.query(ExplanationModel).filter(ExplanationModel.decision_id == decision_id).all()
        finally:
            close_session(session)

class SystemStatusRepository:
    """Repository for system status operations."""
    
    @staticmethod
    def create(status_data):
        """Create a new system status.
        
        Args:
            status_data: System status data
            
        Returns:
            SystemStatusModel: Created system status
        """
        session = get_session()
        try:
            status = SystemStatusModel(**status_data)
            session.add(status)
            session.commit()
            return status
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating system status: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_latest():
        """Get latest system status.
        
        Returns:
            SystemStatusModel: Latest system status or None
        """
        session = get_session()
        try:
            return session.query(SystemStatusModel).order_by(SystemStatusModel.timestamp.desc()).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_history(hours=24):
        """Get system status history.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List[SystemStatusModel]: List of system statuses
        """
        session = get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return session.query(SystemStatusModel).filter(
                SystemStatusModel.timestamp >= cutoff
            ).order_by(SystemStatusModel.timestamp.asc()).all()
        finally:
            close_session(session)
