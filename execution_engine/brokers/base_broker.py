"""
Base broker interface for all broker implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

from ..orders.models import Order, OrderStatus, ExecutionReport

logger = logging.getLogger(__name__)

class BaseBroker(ABC):
    """Base broker interface."""
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker.
        
        Args:
            order: Order to submit
            
        Returns:
            Order: Updated order with broker information
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order: Order) -> Order:
        """Cancel order.
        
        Args:
            order: Order to cancel
            
        Returns:
            Order: Updated order
        """
        pass
    
    @abstractmethod
    async def get_order(self, broker_order_id: str) -> Order:
        """Get order details from broker.
        
        Args:
            broker_order_id: Broker order ID
            
        Returns:
            Order: Order details
        """
        pass
    
    @abstractmethod
    async def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """Get orders from broker.
        
        Args:
            status: Filter by order status
            
        Returns:
            List[Order]: List of orders
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from broker.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        pass
    
    @abstractmethod
    async def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio from broker.
        
        Returns:
            Dict[str, Any]: Portfolio information
        """
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from broker.
        
        Args:
            symbol: Symbol
            
        Returns:
            Dict[str, Any]: Market data
        """
        pass
