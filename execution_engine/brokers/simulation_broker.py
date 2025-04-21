"""
Simulation broker implementation for testing.
"""

import logging
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from ..config.settings import settings
from ..orders.models import Order, OrderStatus, OrderType, OrderSide
from ..brokers.base_broker import BaseBroker

logger = logging.getLogger(__name__)

class SimulationBroker(BaseBroker):
    """Simulation broker for testing."""
    
    def __init__(self):
        """Initialize simulation broker."""
        self.orders = {}  # broker_order_id -> order
        self.positions = {}  # symbol -> position
        self.portfolio = {
            "cash": 100000.0,
            "equity": 100000.0,
            "buying_power": 200000.0,
            "positions": []
        }
        self.market_data = {}  # symbol -> market data
        
        # Simulation settings
        self.latency_ms = settings.simulation.latency_ms
        self.slippage_bps = settings.simulation.slippage_bps
        self.commission_bps = settings.simulation.commission_bps
        self.rejection_probability = settings.simulation.rejection_probability
        self.partial_fill_probability = settings.simulation.partial_fill_probability
        
        logger.info("Simulation broker initialized")
    
    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker.
        
        Args:
            order: Order to submit
            
        Returns:
            Order: Updated order with broker information
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Generate broker order ID
        broker_order_id = f"sim_{order.id}"
        
        # Simulate random rejection
        if random.random() < self.rejection_probability:
            order.status = OrderStatus.REJECTED
            order.broker_order_id = broker_order_id
            logger.info(f"Simulation rejected order: {order.id}")
            return order
        
        # Update order with broker information
        order.broker_order_id = broker_order_id
        order.status = OrderStatus.ACCEPTED
        
        # Store order
        self.orders[broker_order_id] = order.copy(deep=True)
        
        # Simulate order execution
        asyncio.create_task(self._simulate_execution(broker_order_id))
        
        logger.info(f"Simulation accepted order: {order.id}")
        return order
    
    async def cancel_order(self, order: Order) -> Order:
        """Cancel order.
        
        Args:
            order: Order to cancel
            
        Returns:
            Order: Updated order
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Check if order exists
        if order.broker_order_id not in self.orders:
            raise ValueError(f"Order not found: {order.broker_order_id}")
        
        # Get stored order
        stored_order = self.orders[order.broker_order_id]
        
        # Check if order can be canceled
        if stored_order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            raise ValueError(f"Order cannot be canceled: {stored_order.status}")
        
        # Update order status
        stored_order.status = OrderStatus.CANCELED
        stored_order.canceled_at = datetime.utcnow()
        
        # Update original order
        order.status = OrderStatus.CANCELED
        
        logger.info(f"Simulation canceled order: {order.id}")
        return order
    
    async def get_order(self, broker_order_id: str) -> Order:
        """Get order details from broker.
        
        Args:
            broker_order_id: Broker order ID
            
        Returns:
            Order: Order details
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Check if order exists
        if broker_order_id not in self.orders:
            raise ValueError(f"Order not found: {broker_order_id}")
        
        # Return stored order
        return self.orders[broker_order_id]
    
    async def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """Get orders from broker.
        
        Args:
            status: Filter by order status
            
        Returns:
            List[Order]: List of orders
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Filter orders by status if provided
        if status:
            return [order for order in self.orders.values() if order.status.value == status]
        
        # Return all orders
        return list(self.orders.values())
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from broker.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Return positions
        return list(self.positions.values())
    
    async def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio from broker.
        
        Returns:
            Dict[str, Any]: Portfolio information
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Return portfolio
        return self.portfolio
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from broker.
        
        Args:
            symbol: Symbol
            
        Returns:
            Dict[str, Any]: Market data
        """
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Generate random market data if not available
        if symbol not in self.market_data:
            price = random.uniform(10.0, 1000.0)
            self.market_data[symbol] = {
                "symbol": symbol,
                "timestamp": datetime.utcnow(),
                "bid": price - random.uniform(0.01, 0.1),
                "ask": price + random.uniform(0.01, 0.1),
                "last": price,
                "volume": random.randint(100, 10000),
                "bid_size": random.randint(100, 1000),
                "ask_size": random.randint(100, 1000)
            }
        
        # Return market data
        return self.market_data[symbol]
    
    async def _simulate_execution(self, broker_order_id: str) -> None:
        """Simulate order execution.
        
        Args:
            broker_order_id: Broker order ID
        """
        # Get order
        order = self.orders[broker_order_id]
        
        # Simulate execution delay (1-5 seconds)
        await asyncio.sleep(random.uniform(1.0, 5.0))
        
        # Skip if order was canceled
        if order.status == OrderStatus.CANCELED:
            return
        
        # Determine if partial fill
        is_partial = random.random() < self.partial_fill_probability
        
        # Get market data for price
        market_data = await self.get_market_data(order.symbol)
        
        # Determine execution price with slippage
        if order.order_type == OrderType.MARKET:
            # Market order
            base_price = market_data["ask"] if order.side == OrderSide.BUY else market_data["bid"]
            slippage_factor = 1.0 + (self.slippage_bps / 10000.0) if order.side == OrderSide.BUY else 1.0 - (self.slippage_bps / 10000.0)
            execution_price = base_price * slippage_factor
        elif order.order_type == OrderType.LIMIT:
            # Limit order
            if order.side == OrderSide.BUY and market_data["ask"] > order.limit_price:
                # Price too high, don't execute
                return
            elif order.side == OrderSide.SELL and market_data["bid"] < order.limit_price:
                # Price too low, don't execute
                return
            execution_price = order.limit_price
        else:
            # Other order types not fully implemented in simulation
            execution_price = market_data["last"]
        
        # Determine fill quantity
        if is_partial:
            fill_quantity = order.quantity * random.uniform(0.1, 0.9)
        else:
            fill_quantity = order.quantity
        
        # Update order
        order.filled_quantity = fill_quantity
        order.filled_avg_price = execution_price
        
        # Calculate commission
        order.commission = fill_quantity * execution_price * (self.commission_bps / 10000.0)
        
        # Update order status
        if is_partial:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()
        
        # Update position
        self._update_position(order)
        
        logger.info(f"Simulation executed order: {order.id} - {order.symbol} {order.side.value} {fill_quantity} @ {execution_price}")
    
    def _update_position(self, order: Order) -> None:
        """Update position based on order execution.
        
        Args:
            order: Executed order
        """
        symbol = order.symbol
        quantity = order.filled_quantity
        price = order.filled_avg_price
        
        # Calculate trade value
        trade_value = quantity * price
        commission = order.commission
        
        # Update portfolio cash
        if order.side == OrderSide.BUY:
            self.portfolio["cash"] -= (trade_value + commission)
        else:
            self.portfolio["cash"] += (trade_value - commission)
        
        # Update position
        if symbol not in self.positions:
            # New position
            if order.side == OrderSide.BUY:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_price": price,
                    "cost_basis": trade_value,
                    "market_value": trade_value,
                    "unrealized_pl": 0.0,
                    "realized_pl": 0.0
                }
            else:
                # Short position (simplified)
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": -quantity,
                    "avg_price": price,
                    "cost_basis": trade_value,
                    "market_value": trade_value,
                    "unrealized_pl": 0.0,
                    "realized_pl": 0.0
                }
        else:
            # Existing position
            position = self.positions[symbol]
            
            if order.side == OrderSide.BUY:
                # Add to position
                new_quantity = position["quantity"] + quantity
                new_cost = position["cost_basis"] + trade_value
                
                if new_quantity > 0:
                    # Long position
                    position["avg_price"] = new_cost / new_quantity
                    position["quantity"] = new_quantity
                    position["cost_basis"] = new_cost
                    position["market_value"] = new_quantity * price
                else:
                    # Position closed or flipped
                    realized_pl = (price - position["avg_price"]) * min(quantity, abs(position["quantity"]))
                    position["realized_pl"] += realized_pl
                    
                    if new_quantity == 0:
                        # Position closed
                        del self.positions[symbol]
                    else:
                        # Position flipped to short
                        position["quantity"] = new_quantity
                        position["avg_price"] = price
                        position["cost_basis"] = abs(new_quantity) * price
                        position["market_value"] = abs(new_quantity) * price
            else:
                # Sell from position
                new_quantity = position["quantity"] - quantity
                
                if position["quantity"] > 0:
                    # Selling from long position
                    realized_pl = (price - position["avg_price"]) * min(quantity, position["quantity"])
                    position["realized_pl"] += realized_pl
                    
                    if new_quantity >= 0:
                        # Still long or flat
                        position["quantity"] = new_quantity
                        position["market_value"] = new_quantity * price
                        
                        if new_quantity == 0:
                            # Position closed
                            del self.positions[symbol]
                    else:
                        # Flipped to short
                        position["quantity"] = new_quantity
                        position["avg_price"] = price
                        position["cost_basis"] = abs(new_quantity) * price
                        position["market_value"] = abs(new_quantity) * price
                else:
                    # Adding to short position
                    new_cost = position["cost_basis"] + trade_value
                    position["avg_price"] = new_cost / abs(new_quantity)
                    position["quantity"] = new_quantity
                    position["cost_basis"] = new_cost
                    position["market_value"] = abs(new_quantity) * price
        
        # Update portfolio equity
        total_position_value = sum(p["market_value"] for p in self.positions.values())
        self.portfolio["equity"] = self.portfolio["cash"] + total_position_value
        self.portfolio["buying_power"] = self.portfolio["cash"] * 2  # Simplified margin calculation
        
        # Update portfolio positions
        self.portfolio["positions"] = list(self.positions.values())
