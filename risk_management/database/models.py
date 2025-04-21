"""
Database models and utilities for the risk management module.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, ForeignKey, Text, Enum, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.pool import QueuePool

from ..config.settings import settings
from ..models.base import (
    RiskLevel, RiskType, RiskAction, OptimizationMethod, RebalanceFrequency,
    VaRMethod, VolatilityMethod, CorrelationMethod, SectorClassification,
    FactorModel, DrawdownMethod, ReductionMethod
)

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database.connection_string,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    echo=settings.database.echo,
    poolclass=QueuePool
)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# Create base class for models
Base = declarative_base()

class RiskLimitModel(Base):
    """Risk limit database model."""
    __tablename__ = "risk_limits"
    
    id = Column(String(32), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    risk_type = Column(Enum(RiskType), nullable=False)
    threshold = Column(Float, nullable=False)
    warning_threshold = Column(Float)
    action = Column(Enum(RiskAction), nullable=False, default=RiskAction.WARN)
    scope = Column(String(32), nullable=False)
    scope_id = Column(String(32))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, default={})
    
    # Relationships
    checks = relationship("RiskCheckModel", back_populates="limit")
    profile_id = Column(String(32), ForeignKey("risk_profiles.id"))
    profile = relationship("RiskProfileModel", back_populates="limits")

class RiskCheckModel(Base):
    """Risk check database model."""
    __tablename__ = "risk_checks"
    
    id = Column(String(32), primary_key=True)
    limit_id = Column(String(32), ForeignKey("risk_limits.id"), nullable=False)
    value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    is_breached = Column(Boolean, nullable=False)
    risk_level = Column(Enum(RiskLevel), nullable=False)
    action = Column(Enum(RiskAction), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    context = Column(JSON, default={})
    
    # Relationships
    limit = relationship("RiskLimitModel", back_populates="checks")

class PortfolioAllocationModel(Base):
    """Portfolio allocation database model."""
    __tablename__ = "portfolio_allocations"
    
    id = Column(String(32), primary_key=True)
    portfolio_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    allocations = Column(JSON, nullable=False)
    expected_return = Column(Float)
    expected_risk = Column(Float)
    sharpe_ratio = Column(Float)
    optimization_method = Column(Enum(OptimizationMethod), nullable=False)
    constraints = Column(JSON, default={})
    metadata = Column(JSON, default={})

class PortfolioRiskModel(Base):
    """Portfolio risk database model."""
    __tablename__ = "portfolio_risks"
    
    id = Column(String(32), primary_key=True)
    portfolio_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    invested = Column(Float, nullable=False)
    leverage = Column(Float, nullable=False)
    var_95 = Column(Float, nullable=False)
    var_99 = Column(Float, nullable=False)
    expected_shortfall = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    beta = Column(Float, nullable=False)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float, nullable=False)
    correlation_matrix = Column(JSON, default={})
    exposures = Column(JSON, default={})
    stress_tests = Column(JSON, default={})
    metadata = Column(JSON, default={})

class ExposureAnalysisModel(Base):
    """Exposure analysis database model."""
    __tablename__ = "exposure_analyses"
    
    id = Column(String(32), primary_key=True)
    portfolio_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    exposure_type = Column(String(32), nullable=False)
    exposures = Column(JSON, nullable=False)
    net_exposure = Column(Float, nullable=False)
    gross_exposure = Column(Float, nullable=False)
    long_exposure = Column(Float, nullable=False)
    short_exposure = Column(Float, nullable=False)
    benchmark_exposures = Column(JSON)
    active_exposures = Column(JSON)
    metadata = Column(JSON, default={})

class DrawdownEventModel(Base):
    """Drawdown event database model."""
    __tablename__ = "drawdown_events"
    
    id = Column(String(32), primary_key=True)
    portfolio_id = Column(String(32), nullable=False, index=True)
    start_date = Column(DateTime, nullable=False)
    current_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_date = Column(DateTime)
    peak_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    drawdown_pct = Column(Float, nullable=False)
    drawdown_duration_days = Column(Integer, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    actions_taken = Column(JSON, default=[])
    metadata = Column(JSON, default={})

class RiskProfileModel(Base):
    """Risk profile database model."""
    __tablename__ = "risk_profiles"
    
    id = Column(String(32), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    max_drawdown_pct = Column(Float, nullable=False)
    max_leverage = Column(Float, nullable=False)
    max_position_size_pct = Column(Float, nullable=False)
    max_sector_exposure_pct = Column(Float, nullable=False)
    var_limit_pct = Column(Float, nullable=False)
    target_volatility = Column(Float)
    target_beta = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, default={})
    
    # Relationships
    limits = relationship("RiskLimitModel", back_populates="profile")

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
        # Calculate cutoff date
        risk_cutoff = datetime.utcnow() - timedelta(days=settings.database.risk_history_days)
        
        # Delete old risk checks
        session.query(RiskCheckModel).filter(RiskCheckModel.timestamp < risk_cutoff).delete()
        
        # Delete old portfolio risks
        session.query(PortfolioRiskModel).filter(PortfolioRiskModel.timestamp < risk_cutoff).delete()
        
        # Delete old exposure analyses
        session.query(ExposureAnalysisModel).filter(ExposureAnalysisModel.timestamp < risk_cutoff).delete()
        
        # Delete old portfolio allocations
        session.query(PortfolioAllocationModel).filter(PortfolioAllocationModel.timestamp < risk_cutoff).delete()
        
        # Delete old inactive drawdown events
        session.query(DrawdownEventModel).filter(
            DrawdownEventModel.is_active == False,
            DrawdownEventModel.end_date < risk_cutoff
        ).delete()
        
        session.commit()
        logger.info("Old risk data cleaned up")
    except Exception as e:
        session.rollback()
        logger.error(f"Error cleaning up old risk data: {e}")
    finally:
        close_session(session)

# Database repository classes
class RiskLimitRepository:
    """Repository for risk limit operations."""
    
    @staticmethod
    def create(limit_data):
        """Create a new risk limit.
        
        Args:
            limit_data: Risk limit data
            
        Returns:
            RiskLimitModel: Created risk limit
        """
        session = get_session()
        try:
            limit = RiskLimitModel(**limit_data)
            session.add(limit)
            session.commit()
            return limit
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating risk limit: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(limit_id):
        """Get risk limit by ID.
        
        Args:
            limit_id: Risk limit ID
            
        Returns:
            RiskLimitModel: Risk limit or None
        """
        session = get_session()
        try:
            return session.query(RiskLimitModel).filter(RiskLimitModel.id == limit_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_scope(scope, scope_id=None):
        """Get risk limits by scope.
        
        Args:
            scope: Scope
            scope_id: Scope ID
            
        Returns:
            List[RiskLimitModel]: List of risk limits
        """
        session = get_session()
        try:
            query = session.query(RiskLimitModel).filter(
                RiskLimitModel.scope == scope,
                RiskLimitModel.is_active == True
            )
            
            if scope_id:
                query = query.filter(RiskLimitModel.scope_id == scope_id)
            
            return query.all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_risk_type(risk_type):
        """Get risk limits by risk type.
        
        Args:
            risk_type: Risk type
            
        Returns:
            List[RiskLimitModel]: List of risk limits
        """
        session = get_session()
        try:
            return session.query(RiskLimitModel).filter(
                RiskLimitModel.risk_type == risk_type,
                RiskLimitModel.is_active == True
            ).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_active_limits():
        """Get all active risk limits.
        
        Returns:
            List[RiskLimitModel]: List of active risk limits
        """
        session = get_session()
        try:
            return session.query(RiskLimitModel).filter(RiskLimitModel.is_active == True).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(limit_id, limit_data):
        """Update risk limit.
        
        Args:
            limit_id: Risk limit ID
            limit_data: Risk limit data to update
            
        Returns:
            RiskLimitModel: Updated risk limit or None
        """
        session = get_session()
        try:
            limit = session.query(RiskLimitModel).filter(RiskLimitModel.id == limit_id).first()
            if limit:
                for key, value in limit_data.items():
                    setattr(limit, key, value)
                session.commit()
            return limit
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating risk limit: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def delete(limit_id):
        """Delete risk limit.
        
        Args:
            limit_id: Risk limit ID
            
        Returns:
            bool: Success status
        """
        session = get_session()
        try:
            limit = session.query(RiskLimitModel).filter(RiskLimitModel.id == limit_id).first()
            if limit:
                session.delete(limit)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting risk limit: {e}")
            raise
        finally:
            close_session(session)

class RiskCheckRepository:
    """Repository for risk check operations."""
    
    @staticmethod
    def create(check_data):
        """Create a new risk check.
        
        Args:
            check_data: Risk check data
            
        Returns:
            RiskCheckModel: Created risk check
        """
        session = get_session()
        try:
            check = RiskCheckModel(**check_data)
            session.add(check)
            session.commit()
            return check
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating risk check: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(check_id):
        """Get risk check by ID.
        
        Args:
            check_id: Risk check ID
            
        Returns:
            RiskCheckModel: Risk check or None
        """
        session = get_session()
        try:
            return session.query(RiskCheckModel).filter(RiskCheckModel.id == check_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_limit_id(limit_id, start_date=None, end_date=None):
        """Get risk checks by limit ID.
        
        Args:
            limit_id: Risk limit ID
            start_date: Start date
            end_date: End date
            
        Returns:
            List[RiskCheckModel]: List of risk checks
        """
        session = get_session()
        try:
            query = session.query(RiskCheckModel).filter(RiskCheckModel.limit_id == limit_id)
            
            if start_date:
                query = query.filter(RiskCheckModel.timestamp >= start_date)
            
            if end_date:
                query = query.filter(RiskCheckModel.timestamp <= end_date)
            
            return query.order_by(RiskCheckModel.timestamp.desc()).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_breached_checks(start_date=None, end_date=None):
        """Get breached risk checks.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List[RiskCheckModel]: List of breached risk checks
        """
        session = get_session()
        try:
            query = session.query(RiskCheckModel).filter(RiskCheckModel.is_breached == True)
            
            if start_date:
                query = query.filter(RiskCheckModel.timestamp >= start_date)
            
            if end_date:
                query = query.filter(RiskCheckModel.timestamp <= end_date)
            
            return query.order_by(RiskCheckModel.timestamp.desc()).all()
        finally:
            close_session(session)

class PortfolioAllocationRepository:
    """Repository for portfolio allocation operations."""
    
    @staticmethod
    def create(allocation_data):
        """Create a new portfolio allocation.
        
        Args:
            allocation_data: Portfolio allocation data
            
        Returns:
            PortfolioAllocationModel: Created portfolio allocation
        """
        session = get_session()
        try:
            allocation = PortfolioAllocationModel(**allocation_data)
            session.add(allocation)
            session.commit()
            return allocation
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating portfolio allocation: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(allocation_id):
        """Get portfolio allocation by ID.
        
        Args:
            allocation_id: Portfolio allocation ID
            
        Returns:
            PortfolioAllocationModel: Portfolio allocation or None
        """
        session = get_session()
        try:
            return session.query(PortfolioAllocationModel).filter(
                PortfolioAllocationModel.id == allocation_id
            ).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_latest_by_portfolio_id(portfolio_id):
        """Get latest portfolio allocation by portfolio ID.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            PortfolioAllocationModel: Latest portfolio allocation or None
        """
        session = get_session()
        try:
            return session.query(PortfolioAllocationModel).filter(
                PortfolioAllocationModel.portfolio_id == portfolio_id
            ).order_by(PortfolioAllocationModel.timestamp.desc()).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_portfolio_id(portfolio_id, start_date=None, end_date=None, limit=10):
        """Get portfolio allocations by portfolio ID.
        
        Args:
            portfolio_id: Portfolio ID
            start_date: Start date
            end_date: End date
            limit: Maximum number of allocations to return
            
        Returns:
            List[PortfolioAllocationModel]: List of portfolio allocations
        """
        session = get_session()
        try:
            query = session.query(PortfolioAllocationModel).filter(
                PortfolioAllocationModel.portfolio_id == portfolio_id
            )
            
            if start_date:
                query = query.filter(PortfolioAllocationModel.timestamp >= start_date)
            
            if end_date:
                query = query.filter(PortfolioAllocationModel.timestamp <= end_date)
            
            return query.order_by(PortfolioAllocationModel.timestamp.desc()).limit(limit).all()
        finally:
            close_session(session)

class PortfolioRiskRepository:
    """Repository for portfolio risk operations."""
    
    @staticmethod
    def create(risk_data):
        """Create a new portfolio risk.
        
        Args:
            risk_data: Portfolio risk data
            
        Returns:
            PortfolioRiskModel: Created portfolio risk
        """
        session = get_session()
        try:
            risk = PortfolioRiskModel(**risk_data)
            session.add(risk)
            session.commit()
            return risk
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating portfolio risk: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(risk_id):
        """Get portfolio risk by ID.
        
        Args:
            risk_id: Portfolio risk ID
            
        Returns:
            PortfolioRiskModel: Portfolio risk or None
        """
        session = get_session()
        try:
            return session.query(PortfolioRiskModel).filter(PortfolioRiskModel.id == risk_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_latest_by_portfolio_id(portfolio_id):
        """Get latest portfolio risk by portfolio ID.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            PortfolioRiskModel: Latest portfolio risk or None
        """
        session = get_session()
        try:
            return session.query(PortfolioRiskModel).filter(
                PortfolioRiskModel.portfolio_id == portfolio_id
            ).order_by(PortfolioRiskModel.timestamp.desc()).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_portfolio_id(portfolio_id, start_date=None, end_date=None, limit=10):
        """Get portfolio risks by portfolio ID.
        
        Args:
            portfolio_id: Portfolio ID
            start_date: Start date
            end_date: End date
            limit: Maximum number of risks to return
            
        Returns:
            List[PortfolioRiskModel]: List of portfolio risks
        """
        session = get_session()
        try:
            query = session.query(PortfolioRiskModel).filter(
                PortfolioRiskModel.portfolio_id == portfolio_id
            )
            
            if start_date:
                query = query.filter(PortfolioRiskModel.timestamp >= start_date)
            
            if end_date:
                query = query.filter(PortfolioRiskModel.timestamp <= end_date)
            
            return query.order_by(PortfolioRiskModel.timestamp.desc()).limit(limit).all()
        finally:
            close_session(session)

class ExposureAnalysisRepository:
    """Repository for exposure analysis operations."""
    
    @staticmethod
    def create(exposure_data):
        """Create a new exposure analysis.
        
        Args:
            exposure_data: Exposure analysis data
            
        Returns:
            ExposureAnalysisModel: Created exposure analysis
        """
        session = get_session()
        try:
            exposure = ExposureAnalysisModel(**exposure_data)
            session.add(exposure)
            session.commit()
            return exposure
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating exposure analysis: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(exposure_id):
        """Get exposure analysis by ID.
        
        Args:
            exposure_id: Exposure analysis ID
            
        Returns:
            ExposureAnalysisModel: Exposure analysis or None
        """
        session = get_session()
        try:
            return session.query(ExposureAnalysisModel).filter(
                ExposureAnalysisModel.id == exposure_id
            ).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_latest_by_portfolio_and_type(portfolio_id, exposure_type):
        """Get latest exposure analysis by portfolio ID and exposure type.
        
        Args:
            portfolio_id: Portfolio ID
            exposure_type: Exposure type
            
        Returns:
            ExposureAnalysisModel: Latest exposure analysis or None
        """
        session = get_session()
        try:
            return session.query(ExposureAnalysisModel).filter(
                ExposureAnalysisModel.portfolio_id == portfolio_id,
                ExposureAnalysisModel.exposure_type == exposure_type
            ).order_by(ExposureAnalysisModel.timestamp.desc()).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_portfolio_and_type(portfolio_id, exposure_type, start_date=None, end_date=None, limit=10):
        """Get exposure analyses by portfolio ID and exposure type.
        
        Args:
            portfolio_id: Portfolio ID
            exposure_type: Exposure type
            start_date: Start date
            end_date: End date
            limit: Maximum number of analyses to return
            
        Returns:
            List[ExposureAnalysisModel]: List of exposure analyses
        """
        session = get_session()
        try:
            query = session.query(ExposureAnalysisModel).filter(
                ExposureAnalysisModel.portfolio_id == portfolio_id,
                ExposureAnalysisModel.exposure_type == exposure_type
            )
            
            if start_date:
                query = query.filter(ExposureAnalysisModel.timestamp >= start_date)
            
            if end_date:
                query = query.filter(ExposureAnalysisModel.timestamp <= end_date)
            
            return query.order_by(ExposureAnalysisModel.timestamp.desc()).limit(limit).all()
        finally:
            close_session(session)

class DrawdownEventRepository:
    """Repository for drawdown event operations."""
    
    @staticmethod
    def create(event_data):
        """Create a new drawdown event.
        
        Args:
            event_data: Drawdown event data
            
        Returns:
            DrawdownEventModel: Created drawdown event
        """
        session = get_session()
        try:
            event = DrawdownEventModel(**event_data)
            session.add(event)
            session.commit()
            return event
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating drawdown event: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(event_id):
        """Get drawdown event by ID.
        
        Args:
            event_id: Drawdown event ID
            
        Returns:
            DrawdownEventModel: Drawdown event or None
        """
        session = get_session()
        try:
            return session.query(DrawdownEventModel).filter(DrawdownEventModel.id == event_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_active_by_portfolio_id(portfolio_id):
        """Get active drawdown events by portfolio ID.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            List[DrawdownEventModel]: List of active drawdown events
        """
        session = get_session()
        try:
            return session.query(DrawdownEventModel).filter(
                DrawdownEventModel.portfolio_id == portfolio_id,
                DrawdownEventModel.is_active == True
            ).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_portfolio_id(portfolio_id, include_active_only=False, start_date=None, end_date=None):
        """Get drawdown events by portfolio ID.
        
        Args:
            portfolio_id: Portfolio ID
            include_active_only: Whether to include only active events
            start_date: Start date
            end_date: End date
            
        Returns:
            List[DrawdownEventModel]: List of drawdown events
        """
        session = get_session()
        try:
            query = session.query(DrawdownEventModel).filter(
                DrawdownEventModel.portfolio_id == portfolio_id
            )
            
            if include_active_only:
                query = query.filter(DrawdownEventModel.is_active == True)
            
            if start_date:
                query = query.filter(DrawdownEventModel.start_date >= start_date)
            
            if end_date:
                query = query.filter(
                    (DrawdownEventModel.end_date <= end_date) | 
                    (DrawdownEventModel.end_date == None)
                )
            
            return query.order_by(DrawdownEventModel.start_date.desc()).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(event_id, event_data):
        """Update drawdown event.
        
        Args:
            event_id: Drawdown event ID
            event_data: Drawdown event data to update
            
        Returns:
            DrawdownEventModel: Updated drawdown event or None
        """
        session = get_session()
        try:
            event = session.query(DrawdownEventModel).filter(DrawdownEventModel.id == event_id).first()
            if event:
                for key, value in event_data.items():
                    setattr(event, key, value)
                session.commit()
            return event
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating drawdown event: {e}")
            raise
        finally:
            close_session(session)

class RiskProfileRepository:
    """Repository for risk profile operations."""
    
    @staticmethod
    def create(profile_data):
        """Create a new risk profile.
        
        Args:
            profile_data: Risk profile data
            
        Returns:
            RiskProfileModel: Created risk profile
        """
        session = get_session()
        try:
            profile = RiskProfileModel(**profile_data)
            session.add(profile)
            session.commit()
            return profile
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating risk profile: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(profile_id):
        """Get risk profile by ID.
        
        Args:
            profile_id: Risk profile ID
            
        Returns:
            RiskProfileModel: Risk profile or None
        """
        session = get_session()
        try:
            return session.query(RiskProfileModel).filter(RiskProfileModel.id == profile_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_name(name):
        """Get risk profile by name.
        
        Args:
            name: Risk profile name
            
        Returns:
            RiskProfileModel: Risk profile or None
        """
        session = get_session()
        try:
            return session.query(RiskProfileModel).filter(RiskProfileModel.name == name).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_active_profiles():
        """Get all active risk profiles.
        
        Returns:
            List[RiskProfileModel]: List of active risk profiles
        """
        session = get_session()
        try:
            return session.query(RiskProfileModel).filter(RiskProfileModel.is_active == True).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(profile_id, profile_data):
        """Update risk profile.
        
        Args:
            profile_id: Risk profile ID
            profile_data: Risk profile data to update
            
        Returns:
            RiskProfileModel: Updated risk profile or None
        """
        session = get_session()
        try:
            profile = session.query(RiskProfileModel).filter(RiskProfileModel.id == profile_id).first()
            if profile:
                for key, value in profile_data.items():
                    if key != 'limits':  # Handle limits separately
                        setattr(profile, key, value)
                session.commit()
            return profile
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating risk profile: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def delete(profile_id):
        """Delete risk profile.
        
        Args:
            profile_id: Risk profile ID
            
        Returns:
            bool: Success status
        """
        session = get_session()
        try:
            profile = session.query(RiskProfileModel).filter(RiskProfileModel.id == profile_id).first()
            if profile:
                session.delete(profile)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting risk profile: {e}")
            raise
        finally:
            close_session(session)
