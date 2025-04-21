"""
Database models and utilities for the execution engine.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, ForeignKey, Text, Enum, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.pool import QueuePool

from ..config.settings import settings
from ..orders.models import (
    OrderType, OrderSide, TimeInForce, OrderStatus, 
    ExecutionAlgorithm, BrokerType, AssetClass, AssetType, Currency
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

class AssetModel(Base):
    """Asset database model."""
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False, index=True, unique=True)
    name = Column(String(255))
    asset_class = Column(Enum(AssetClass), nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    exchange = Column(String(32))
    currency = Column(Enum(Currency), nullable=False, default=Currency.USD)
    is_tradable = Column(Boolean, default=True)
    is_shortable = Column(Boolean, default=True)
    is_marginable = Column(Boolean, default=True)
    min_trade_size = Column(Float)
    price_increment = Column(Float)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    orders = relationship("OrderModel", back_populates="asset")
    positions = relationship("PositionModel", back_populates="asset")

class OrderModel(Base):
    """Order database model."""
    __tablename__ = "orders"
    
    id = Column(String(32), primary_key=True)
    client_order_id = Column(String(64), index=True)
    strategy_id = Column(String(32), index=True)
    signal_id = Column(String(32), index=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(Enum(OrderSide), nullable=False)
    quantity = Column(Float, nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    time_in_force = Column(Enum(TimeInForce), nullable=False, default=TimeInForce.DAY)
    limit_price = Column(Float)
    stop_price = Column(Float)
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.NEW)
    broker = Column(Enum(BrokerType), nullable=False)
    broker_order_id = Column(String(64), index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    submitted_at = Column(DateTime)
    filled_at = Column(DateTime)
    canceled_at = Column(DateTime)
    expired_at = Column(DateTime)
    filled_quantity = Column(Float, default=0.0)
    filled_avg_price = Column(Float)
    commission = Column(Float, default=0.0)
    execution_algorithm = Column(Enum(ExecutionAlgorithm), nullable=False, default=ExecutionAlgorithm.MARKET)
    algorithm_params = Column(JSON, default={})
    parent_order_id = Column(String(32), ForeignKey("orders.id"))
    metadata = Column(JSON, default={})
    
    # Relationships
    asset = relationship("AssetModel", back_populates="orders")
    executions = relationship("ExecutionModel", back_populates="order")
    parent_order = relationship("OrderModel", remote_side=[id], backref="child_orders")

class ExecutionModel(Base):
    """Execution database model."""
    __tablename__ = "executions"
    
    id = Column(String(32), primary_key=True)
    order_id = Column(String(32), ForeignKey("orders.id"), nullable=False, index=True)
    broker_order_id = Column(String(64), index=True)
    broker_execution_id = Column(String(64), index=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(Enum(OrderSide), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    commission = Column(Float, default=0.0)
    liquidity = Column(String(32))
    venue = Column(String(32))
    metadata = Column(JSON, default={})
    
    # Relationships
    order = relationship("OrderModel", back_populates="executions")

class PositionModel(Base):
    """Position database model."""
    __tablename__ = "positions"
    
    id = Column(String(32), primary_key=True)
    symbol = Column(String(32), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    cost_basis = Column(Float, nullable=False)
    unrealized_pl = Column(Float)
    unrealized_pl_pct = Column(Float)
    realized_pl = Column(Float, default=0.0)
    realized_pl_pct = Column(Float)
    open_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    strategy_id = Column(String(32), index=True)
    broker = Column(Enum(BrokerType), nullable=False)
    asset_class = Column(Enum(AssetClass), nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    currency = Column(Enum(Currency), nullable=False, default=Currency.USD)
    metadata = Column(JSON, default={})
    
    # Relationships
    asset = relationship("AssetModel", back_populates="positions")
    portfolio = relationship("PortfolioModel", back_populates="positions")

class PortfolioModel(Base):
    """Portfolio database model."""
    __tablename__ = "portfolios"
    
    id = Column(String(32), primary_key=True)
    name = Column(String(255), nullable=False)
    cash = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    buying_power = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    broker = Column(Enum(BrokerType), nullable=False)
    currency = Column(Enum(Currency), nullable=False, default=Currency.USD)
    metadata = Column(JSON, default={})
    
    # Relationships
    positions = relationship("PositionModel", back_populates="portfolio")

class ExecutionReportModel(Base):
    """Execution report database model."""
    __tablename__ = "execution_reports"
    
    id = Column(String(32), primary_key=True)
    order_id = Column(String(32), ForeignKey("orders.id"), nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(Enum(OrderSide), nullable=False)
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, nullable=False)
    avg_price = Column(Float)
    status = Column(Enum(OrderStatus), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    broker = Column(Enum(BrokerType), nullable=False)
    broker_order_id = Column(String(64), index=True)
    message = Column(Text)
    metadata = Column(JSON, default={})
    
    # Relationships
    order = relationship("OrderModel")

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
        order_cutoff = datetime.utcnow() - timedelta(days=settings.database.order_history_days)
        execution_cutoff = datetime.utcnow() - timedelta(days=settings.database.execution_history_days)
        
        # Delete old orders
        session.query(OrderModel).filter(OrderModel.created_at < order_cutoff).delete()
        
        # Delete old executions
        session.query(ExecutionModel).filter(ExecutionModel.timestamp < execution_cutoff).delete()
        
        # Delete old execution reports
        session.query(ExecutionReportModel).filter(ExecutionReportModel.timestamp < execution_cutoff).delete()
        
        session.commit()
        logger.info("Old data cleaned up")
    except Exception as e:
        session.rollback()
        logger.error(f"Error cleaning up old data: {e}")
    finally:
        close_session(session)

# Database repository classes
class AssetRepository:
    """Repository for asset operations."""
    
    @staticmethod
    def create(asset_data):
        """Create a new asset.
        
        Args:
            asset_data: Asset data
            
        Returns:
            AssetModel: Created asset
        """
        session = get_session()
        try:
            asset = AssetModel(**asset_data)
            session.add(asset)
            session.commit()
            return asset
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating asset: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_symbol(symbol):
        """Get asset by symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            AssetModel: Asset or None
        """
        session = get_session()
        try:
            return session.query(AssetModel).filter(AssetModel.symbol == symbol).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_all():
        """Get all assets.
        
        Returns:
            List[AssetModel]: List of assets
        """
        session = get_session()
        try:
            return session.query(AssetModel).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(symbol, asset_data):
        """Update asset.
        
        Args:
            symbol: Asset symbol
            asset_data: Asset data to update
            
        Returns:
            AssetModel: Updated asset or None
        """
        session = get_session()
        try:
            asset = session.query(AssetModel).filter(AssetModel.symbol == symbol).first()
            if asset:
                for key, value in asset_data.items():
                    setattr(asset, key, value)
                session.commit()
            return asset
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating asset: {e}")
            raise
        finally:
            close_session(session)

class OrderRepository:
    """Repository for order operations."""
    
    @staticmethod
    def create(order_data):
        """Create a new order.
        
        Args:
            order_data: Order data
            
        Returns:
            OrderModel: Created order
        """
        session = get_session()
        try:
            order = OrderModel(**order_data)
            session.add(order)
            session.commit()
            return order
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating order: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(order_id):
        """Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderModel: Order or None
        """
        session = get_session()
        try:
            return session.query(OrderModel).filter(OrderModel.id == order_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_client_order_id(client_order_id):
        """Get order by client order ID.
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            OrderModel: Order or None
        """
        session = get_session()
        try:
            return session.query(OrderModel).filter(OrderModel.client_order_id == client_order_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_broker_order_id(broker_order_id):
        """Get order by broker order ID.
        
        Args:
            broker_order_id: Broker order ID
            
        Returns:
            OrderModel: Order or None
        """
        session = get_session()
        try:
            return session.query(OrderModel).filter(OrderModel.broker_order_id == broker_order_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_strategy_id(strategy_id):
        """Get orders by strategy ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            List[OrderModel]: List of orders
        """
        session = get_session()
        try:
            return session.query(OrderModel).filter(OrderModel.strategy_id == strategy_id).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_active_orders():
        """Get active orders.
        
        Returns:
            List[OrderModel]: List of active orders
        """
        session = get_session()
        try:
            return session.query(OrderModel).filter(
                OrderModel.status.in_([
                    OrderStatus.NEW, 
                    OrderStatus.PENDING, 
                    OrderStatus.ACCEPTED, 
                    OrderStatus.PARTIALLY_FILLED
                ])
            ).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(order_id, order_data):
        """Update order.
        
        Args:
            order_id: Order ID
            order_data: Order data to update
            
        Returns:
            OrderModel: Updated order or None
        """
        session = get_session()
        try:
            order = session.query(OrderModel).filter(OrderModel.id == order_id).first()
            if order:
                for key, value in order_data.items():
                    setattr(order, key, value)
                session.commit()
            return order
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating order: {e}")
            raise
        finally:
            close_session(session)

class ExecutionRepository:
    """Repository for execution operations."""
    
    @staticmethod
    def create(execution_data):
        """Create a new execution.
        
        Args:
            execution_data: Execution data
            
        Returns:
            ExecutionModel: Created execution
        """
        session = get_session()
        try:
            execution = ExecutionModel(**execution_data)
            session.add(execution)
            session.commit()
            return execution
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating execution: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(execution_id):
        """Get execution by ID.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            ExecutionModel: Execution or None
        """
        session = get_session()
        try:
            return session.query(ExecutionModel).filter(ExecutionModel.id == execution_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_order_id(order_id):
        """Get executions by order ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            List[ExecutionModel]: List of executions
        """
        session = get_session()
        try:
            return session.query(ExecutionModel).filter(ExecutionModel.order_id == order_id).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_broker_execution_id(broker_execution_id):
        """Get execution by broker execution ID.
        
        Args:
            broker_execution_id: Broker execution ID
            
        Returns:
            ExecutionModel: Execution or None
        """
        session = get_session()
        try:
            return session.query(ExecutionModel).filter(
                ExecutionModel.broker_execution_id == broker_execution_id
            ).first()
        finally:
            close_session(session)

class PositionRepository:
    """Repository for position operations."""
    
    @staticmethod
    def create(position_data):
        """Create a new position.
        
        Args:
            position_data: Position data
            
        Returns:
            PositionModel: Created position
        """
        session = get_session()
        try:
            position = PositionModel(**position_data)
            session.add(position)
            session.commit()
            return position
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating position: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(position_id):
        """Get position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            PositionModel: Position or None
        """
        session = get_session()
        try:
            return session.query(PositionModel).filter(PositionModel.id == position_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_symbol(symbol):
        """Get position by symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            PositionModel: Position or None
        """
        session = get_session()
        try:
            return session.query(PositionModel).filter(PositionModel.symbol == symbol).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_strategy_id(strategy_id):
        """Get positions by strategy ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            List[PositionModel]: List of positions
        """
        session = get_session()
        try:
            return session.query(PositionModel).filter(PositionModel.strategy_id == strategy_id).all()
        finally:
            close_session(session)
    
    @staticmethod
    def get_all():
        """Get all positions.
        
        Returns:
            List[PositionModel]: List of positions
        """
        session = get_session()
        try:
            return session.query(PositionModel).all()
        finally:
            close_session(session)
    
    @staticmethod
    def update(position_id, position_data):
        """Update position.
        
        Args:
            position_id: Position ID
            position_data: Position data to update
            
        Returns:
            PositionModel: Updated position or None
        """
        session = get_session()
        try:
            position = session.query(PositionModel).filter(PositionModel.id == position_id).first()
            if position:
                for key, value in position_data.items():
                    setattr(position, key, value)
                session.commit()
            return position
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating position: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def delete(position_id):
        """Delete position.
        
        Args:
            position_id: Position ID
            
        Returns:
            bool: Success status
        """
        session = get_session()
        try:
            position = session.query(PositionModel).filter(PositionModel.id == position_id).first()
            if position:
                session.delete(position)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting position: {e}")
            raise
        finally:
            close_session(session)

class PortfolioRepository:
    """Repository for portfolio operations."""
    
    @staticmethod
    def create(portfolio_data):
        """Create a new portfolio.
        
        Args:
            portfolio_data: Portfolio data
            
        Returns:
            PortfolioModel: Created portfolio
        """
        session = get_session()
        try:
            portfolio = PortfolioModel(**portfolio_data)
            session.add(portfolio)
            session.commit()
            return portfolio
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating portfolio: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(portfolio_id):
        """Get portfolio by ID.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            PortfolioModel: Portfolio or None
        """
        session = get_session()
        try:
            return session.query(PortfolioModel).filter(PortfolioModel.id == portfolio_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_broker(broker):
        """Get portfolio by broker.
        
        Args:
            broker: Broker type
            
        Returns:
            PortfolioModel: Portfolio or None
        """
        session = get_session()
        try:
            return session.query(PortfolioModel).filter(PortfolioModel.broker == broker).first()
        finally:
            close_session(session)
    
    @staticmethod
    def update(portfolio_id, portfolio_data):
        """Update portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            portfolio_data: Portfolio data to update
            
        Returns:
            PortfolioModel: Updated portfolio or None
        """
        session = get_session()
        try:
            portfolio = session.query(PortfolioModel).filter(PortfolioModel.id == portfolio_id).first()
            if portfolio:
                for key, value in portfolio_data.items():
                    setattr(portfolio, key, value)
                session.commit()
            return portfolio
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating portfolio: {e}")
            raise
        finally:
            close_session(session)

class ExecutionReportRepository:
    """Repository for execution report operations."""
    
    @staticmethod
    def create(report_data):
        """Create a new execution report.
        
        Args:
            report_data: Execution report data
            
        Returns:
            ExecutionReportModel: Created execution report
        """
        session = get_session()
        try:
            report = ExecutionReportModel(**report_data)
            session.add(report)
            session.commit()
            return report
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating execution report: {e}")
            raise
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_id(report_id):
        """Get execution report by ID.
        
        Args:
            report_id: Execution report ID
            
        Returns:
            ExecutionReportModel: Execution report or None
        """
        session = get_session()
        try:
            return session.query(ExecutionReportModel).filter(ExecutionReportModel.id == report_id).first()
        finally:
            close_session(session)
    
    @staticmethod
    def get_by_order_id(order_id):
        """Get execution reports by order ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            List[ExecutionReportModel]: List of execution reports
        """
        session = get_session()
        try:
            return session.query(ExecutionReportModel).filter(
                ExecutionReportModel.order_id == order_id
            ).all()
        finally:
            close_session(session)
