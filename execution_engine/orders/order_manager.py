"""
Order manager for handling order lifecycle.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from ..config.settings import settings
from ..orders.models import (
    Order, OrderType, OrderSide, TimeInForce, OrderStatus, 
    ExecutionAlgorithm, BrokerType, ExecutionReport
)
from ..database.models import (
    OrderRepository, ExecutionRepository, ExecutionReportRepository
)
from ..brokers.broker_factory import BrokerFactory

logger = logging.getLogger(__name__)

class OrderManager:
    """Manager for order lifecycle."""
    
    def __init__(self):
        """Initialize order manager."""
        self.broker_factory = BrokerFactory()
        self.default_broker = settings.brokers.default_broker
        self.max_order_size_percent = settings.orders.max_order_size_percent
        self.min_order_size_usd = settings.orders.min_order_size_usd
        self.max_order_size_usd = settings.orders.max_order_size_usd
        self.retry_attempts = settings.orders.retry_attempts
        self.retry_delay_seconds = settings.orders.retry_delay_seconds
    
    async def create_order(self, 
                         symbol: str, 
                         side: str, 
                         quantity: float, 
                         order_type: str = "market",
                         time_in_force: str = "day",
                         limit_price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         client_order_id: Optional[str] = None,
                         strategy_id: Optional[str] = None,
                         signal_id: Optional[str] = None,
                         broker: Optional[str] = None,
                         execution_algorithm: str = "market",
                         algorithm_params: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Order:
        """Create a new order.
        
        Args:
            symbol: Symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type
            time_in_force: Time in force
            limit_price: Limit price (required for limit and stop-limit orders)
            stop_price: Stop price (required for stop and stop-limit orders)
            client_order_id: Client order ID
            strategy_id: Strategy ID
            signal_id: Signal ID
            broker: Broker type
            execution_algorithm: Execution algorithm
            algorithm_params: Algorithm parameters
            metadata: Order metadata
            
        Returns:
            Order: Created order
        """
        # Validate inputs
        self._validate_order_inputs(
            symbol, side, quantity, order_type, 
            time_in_force, limit_price, stop_price
        )
        
        # Set defaults
        if broker is None:
            broker = self.default_broker
        
        if client_order_id is None:
            client_order_id = f"client_{uuid.uuid4().hex[:8]}"
        
        if algorithm_params is None:
            algorithm_params = {}
        
        if metadata is None:
            metadata = {}
        
        # Create order object
        order = Order(
            id=f"order_{uuid.uuid4().hex[:8]}",
            client_order_id=client_order_id,
            strategy_id=strategy_id,
            signal_id=signal_id,
            symbol=symbol,
            side=OrderSide(side),
            quantity=quantity,
            order_type=OrderType(order_type),
            time_in_force=TimeInForce(time_in_force),
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.NEW,
            broker=BrokerType(broker),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            execution_algorithm=ExecutionAlgorithm(execution_algorithm),
            algorithm_params=algorithm_params,
            metadata=metadata
        )
        
        # Save order to database
        OrderRepository.create(order.dict())
        
        # Log order creation
        logger.info(f"Order created: {order.id} - {symbol} {side} {quantity} {order_type}")
        
        return order
    
    def _validate_order_inputs(self, 
                             symbol: str, 
                             side: str, 
                             quantity: float, 
                             order_type: str,
                             time_in_force: str, 
                             limit_price: Optional[float],
                             stop_price: Optional[float]) -> None:
        """Validate order inputs.
        
        Args:
            symbol: Symbol
            side: Order side
            quantity: Order quantity
            order_type: Order type
            time_in_force: Time in force
            limit_price: Limit price
            stop_price: Stop price
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check symbol
        if not symbol:
            raise ValueError("Symbol is required")
        
        # Check side
        if side not in [s.value for s in OrderSide]:
            raise ValueError(f"Invalid order side: {side}")
        
        # Check quantity
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")
        
        # Check order type
        if order_type not in [t.value for t in OrderType]:
            raise ValueError(f"Invalid order type: {order_type}")
        
        # Check time in force
        if time_in_force not in [t.value for t in TimeInForce]:
            raise ValueError(f"Invalid time in force: {time_in_force}")
        
        # Check limit price for limit orders
        if order_type in [OrderType.LIMIT.value, OrderType.STOP_LIMIT.value] and limit_price is None:
            raise ValueError(f"Limit price is required for {order_type} orders")
        
        # Check stop price for stop orders
        if order_type in [OrderType.STOP.value, OrderType.STOP_LIMIT.value, OrderType.TRAILING_STOP.value] and stop_price is None:
            raise ValueError(f"Stop price is required for {order_type} orders")
    
    async def submit_order(self, order_id: str) -> Order:
        """Submit order to broker.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order: Updated order
            
        Raises:
            ValueError: If order not found
            RuntimeError: If order submission fails
        """
        # Get order from database
        order_data = OrderRepository.get_by_id(order_id)
        if not order_data:
            raise ValueError(f"Order not found: {order_id}")
        
        # Convert to Order object
        order = Order(**{k: v for k, v in order_data.__dict__.items() if not k.startswith('_')})
        
        # Check if order can be submitted
        if order.status != OrderStatus.NEW:
            raise ValueError(f"Order cannot be submitted: {order.status}")
        
        # Update order status
        order.status = OrderStatus.PENDING
        order.updated_at = datetime.utcnow()
        OrderRepository.update(order.id, {"status": order.status.value, "updated_at": order.updated_at})
        
        # Get broker client
        broker_client = self.broker_factory.get_broker(order.broker.value)
        
        # Submit order to broker
        try:
            broker_order = await broker_client.submit_order(order)
            
            # Update order with broker information
            order.broker_order_id = broker_order.broker_order_id
            order.status = broker_order.status
            order.submitted_at = datetime.utcnow()
            order.updated_at = datetime.utcnow()
            
            # Update order in database
            OrderRepository.update(order.id, {
                "broker_order_id": order.broker_order_id,
                "status": order.status.value,
                "submitted_at": order.submitted_at,
                "updated_at": order.updated_at
            })
            
            # Create execution report
            self._create_execution_report(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                filled_quantity=0.0,
                status=order.status.value,
                broker=order.broker.value,
                broker_order_id=order.broker_order_id,
                message="Order submitted to broker"
            )
            
            # Log order submission
            logger.info(f"Order submitted: {order.id} - {order.symbol} {order.side.value} {order.quantity} {order.order_type.value}")
            
            return order
        
        except Exception as e:
            # Update order status to rejected
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.utcnow()
            OrderRepository.update(order.id, {"status": order.status.value, "updated_at": order.updated_at})
            
            # Create execution report
            self._create_execution_report(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                filled_quantity=0.0,
                status=order.status.value,
                broker=order.broker.value,
                message=f"Order submission failed: {str(e)}"
            )
            
            # Log error
            logger.error(f"Order submission failed: {order.id} - {str(e)}")
            
            raise RuntimeError(f"Order submission failed: {str(e)}")
    
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order: Updated order
            
        Raises:
            ValueError: If order not found
            RuntimeError: If order cancellation fails
        """
        # Get order from database
        order_data = OrderRepository.get_by_id(order_id)
        if not order_data:
            raise ValueError(f"Order not found: {order_id}")
        
        # Convert to Order object
        order = Order(**{k: v for k, v in order_data.__dict__.items() if not k.startswith('_')})
        
        # Check if order can be canceled
        if order.status not in [OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]:
            raise ValueError(f"Order cannot be canceled: {order.status}")
        
        # If order is NEW and not submitted to broker yet, just mark as canceled
        if order.status == OrderStatus.NEW:
            order.status = OrderStatus.CANCELED
            order.canceled_at = datetime.utcnow()
            order.updated_at = datetime.utcnow()
            
            # Update order in database
            OrderRepository.update(order.id, {
                "status": order.status.value,
                "canceled_at": order.canceled_at,
                "updated_at": order.updated_at
            })
            
            # Create execution report
            self._create_execution_report(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                filled_quantity=order.filled_quantity,
                status=order.status.value,
                broker=order.broker.value,
                broker_order_id=order.broker_order_id,
                message="Order canceled before submission"
            )
            
            # Log order cancellation
            logger.info(f"Order canceled before submission: {order.id}")
            
            return order
        
        # Get broker client
        broker_client = self.broker_factory.get_broker(order.broker.value)
        
        # Cancel order with broker
        try:
            canceled_order = await broker_client.cancel_order(order)
            
            # Update order status
            order.status = canceled_order.status
            order.canceled_at = datetime.utcnow()
            order.updated_at = datetime.utcnow()
            
            # Update order in database
            OrderRepository.update(order.id, {
                "status": order.status.value,
                "canceled_at": order.canceled_at,
                "updated_at": order.updated_at
            })
            
            # Create execution report
            self._create_execution_report(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                filled_quantity=order.filled_quantity,
                status=order.status.value,
                broker=order.broker.value,
                broker_order_id=order.broker_order_id,
                message="Order canceled"
            )
            
            # Log order cancellation
            logger.info(f"Order canceled: {order.id}")
            
            return order
        
        except Exception as e:
            # Create execution report
            self._create_execution_report(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                filled_quantity=order.filled_quantity,
                status=order.status.value,
                broker=order.broker.value,
                broker_order_id=order.broker_order_id,
                message=f"Order cancellation failed: {str(e)}"
            )
            
            # Log error
            logger.error(f"Order cancellation failed: {order.id} - {str(e)}")
            
            raise RuntimeError(f"Order cancellation failed: {str(e)}")
    
    async def get_order(self, order_id: str) -> Order:
        """Get order details.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order: Order details
            
        Raises:
            ValueError: If order not found
        """
        # Get order from database
        order_data = OrderRepository.get_by_id(order_id)
        if not order_data:
            raise ValueError(f"Order not found: {order_id}")
        
        # Convert to Order object
        order = Order(**{k: v for k, v in order_data.__dict__.items() if not k.startswith('_')})
        
        # If order is in a final state, return it
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            return order
        
        # Otherwise, get latest status from broker
        if order.broker_order_id:
            try:
                # Get broker client
                broker_client = self.broker_factory.get_broker(order.broker.value)
                
                # Get order from broker
                broker_order = await broker_client.get_order(order.broker_order_id)
                
                # Update order with broker information
                if broker_order.status != order.status:
                    order.status = broker_order.status
                    order.updated_at = datetime.utcnow()
                    
                    # Update filled information if available
                    if broker_order.filled_quantity is not None:
                        order.filled_quantity = broker_order.filled_quantity
                    
                    if broker_order.filled_avg_price is not None:
                        order.filled_avg_price = broker_order.filled_avg_price
                    
                    # If order is filled, set filled_at
                    if order.status == OrderStatus.FILLED and not order.filled_at:
                        order.filled_at = datetime.utcnow()
                    
                    # Update order in database
                    update_data = {
                        "status": order.status.value,
                        "updated_at": order.updated_at,
                        "filled_quantity": order.filled_quantity
                    }
                    
                    if order.filled_avg_price is not None:
                        update_data["filled_avg_price"] = order.filled_avg_price
                    
                    if order.filled_at:
                        update_data["filled_at"] = order.filled_at
                    
                    OrderRepository.update(order.id, update_data)
                    
                    # Create execution report
                    self._create_execution_report(
                        order_id=order.id,
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        filled_quantity=order.filled_quantity,
                        avg_price=order.filled_avg_price,
                        status=order.status.value,
                        broker=order.broker.value,
                        broker_order_id=order.broker_order_id,
                        message=f"Order status updated: {order.status.value}"
                    )
            
            except Exception as e:
                # Log error but don't fail the request
                logger.error(f"Error getting order from broker: {order.id} - {str(e)}")
        
        return order
    
    async def get_orders(self, 
                       strategy_id: Optional[str] = None, 
                       status: Optional[str] = None,
                       symbol: Optional[str] = None) -> List[Order]:
        """Get orders with optional filtering.
        
        Args:
            strategy_id: Filter by strategy ID
            status: Filter by order status
            symbol: Filter by symbol
            
        Returns:
            List[Order]: List of orders
        """
        # Get orders from database
        if strategy_id:
            orders_data = OrderRepository.get_by_strategy_id(strategy_id)
        else:
            # This is a simplified implementation
            # In a real system, you would have more sophisticated querying
            orders_data = OrderRepository.get_active_orders()
        
        # Convert to Order objects
        orders = []
        for order_data in orders_data:
            order = Order(**{k: v for k, v in order_data.__dict__.items() if not k.startswith('_')})
            
            # Apply filters
            if status and order.status.value != status:
                continue
            
            if symbol and order.symbol != symbol:
                continue
            
            orders.append(order)
        
        return orders
    
    def _create_execution_report(self,
                               order_id: str,
                               symbol: str,
                               side: str,
                               quantity: float,
                               filled_quantity: float,
                               status: str,
                               broker: str,
                               broker_order_id: Optional[str] = None,
                               avg_price: Optional[float] = None,
                               message: Optional[str] = None) -> None:
        """Create execution report.
        
        Args:
            order_id: Order ID
            symbol: Symbol
            side: Order side
            quantity: Order quantity
            filled_quantity: Filled quantity
            status: Order status
            broker: Broker type
            broker_order_id: Broker order ID
            avg_price: Average fill price
            message: Report message
        """
        report = ExecutionReport(
            id=f"report_{uuid.uuid4().hex[:8]}",
            order_id=order_id,
            symbol=symbol,
            side=OrderSide(side),
            quantity=quantity,
            filled_quantity=filled_quantity,
            avg_price=avg_price,
            status=OrderStatus(status),
            timestamp=datetime.utcnow(),
            broker=BrokerType(broker),
            broker_order_id=broker_order_id,
            message=message
        )
        
        # Save report to database
        ExecutionReportRepository.create(report.dict())
        
        # Log report creation if logging is enabled
        if settings.logging.log_executions:
            logger.info(f"Execution report: {report.id} - {symbol} {side} {status} {message}")

# Create order manager instance
order_manager = OrderManager()
