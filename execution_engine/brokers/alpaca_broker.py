"""
Alpaca broker implementation.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

from ..config.settings import settings
from ..orders.models import Order, OrderStatus, OrderType, OrderSide, TimeInForce
from ..brokers.base_broker import BaseBroker

logger = logging.getLogger(__name__)

class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation."""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str, data_url: str, use_sandbox: bool = True):
        """Initialize Alpaca broker.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Alpaca API base URL
            data_url: Alpaca data API URL
            use_sandbox: Whether to use sandbox environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.data_url = data_url
        self.use_sandbox = use_sandbox
        
        # Initialize API client
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url,
            data_url=data_url,
            api_version='v2'
        )
        
        logger.info(f"Alpaca broker initialized with {'sandbox' if use_sandbox else 'live'} environment")
    
    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker.
        
        Args:
            order: Order to submit
            
        Returns:
            Order: Updated order with broker information
        """
        try:
            # Map order type
            alpaca_order_type = self._map_order_type(order.order_type.value)
            
            # Map time in force
            alpaca_time_in_force = self._map_time_in_force(order.time_in_force.value)
            
            # Submit order to Alpaca
            alpaca_order = await asyncio.to_thread(
                self.api.submit_order,
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side.value,
                type=alpaca_order_type,
                time_in_force=alpaca_time_in_force,
                limit_price=order.limit_price if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] else None,
                stop_price=order.stop_price if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] else None,
                client_order_id=order.client_order_id
            )
            
            # Update order with broker information
            order.broker_order_id = alpaca_order.id
            order.status = self._map_order_status(alpaca_order.status)
            
            logger.info(f"Order submitted to Alpaca: {order.id} - {alpaca_order.id}")
            
            return order
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            order.status = OrderStatus.REJECTED
            raise RuntimeError(f"Alpaca order submission failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error submitting order to Alpaca: {e}")
            order.status = OrderStatus.REJECTED
            raise RuntimeError(f"Alpaca order submission failed: {str(e)}")
    
    async def cancel_order(self, order: Order) -> Order:
        """Cancel order.
        
        Args:
            order: Order to cancel
            
        Returns:
            Order: Updated order
        """
        try:
            # Cancel order with Alpaca
            await asyncio.to_thread(
                self.api.cancel_order,
                order_id=order.broker_order_id
            )
            
            # Update order status
            order.status = OrderStatus.CANCELED
            
            logger.info(f"Order canceled with Alpaca: {order.id} - {order.broker_order_id}")
            
            return order
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            
            # Check if order was already filled or canceled
            if "order not found" in str(e).lower():
                # Get latest order status
                try:
                    alpaca_order = await asyncio.to_thread(
                        self.api.get_order,
                        order_id=order.broker_order_id
                    )
                    order.status = self._map_order_status(alpaca_order.status)
                except:
                    # If we can't get the order, assume it's canceled
                    order.status = OrderStatus.CANCELED
            
            raise RuntimeError(f"Alpaca order cancellation failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error canceling order with Alpaca: {e}")
            raise RuntimeError(f"Alpaca order cancellation failed: {str(e)}")
    
    async def get_order(self, broker_order_id: str) -> Order:
        """Get order details from broker.
        
        Args:
            broker_order_id: Broker order ID
            
        Returns:
            Order: Order details
        """
        try:
            # Get order from Alpaca
            alpaca_order = await asyncio.to_thread(
                self.api.get_order,
                order_id=broker_order_id
            )
            
            # Create order object
            order = Order(
                id="unknown",  # Will be updated by caller
                broker_order_id=alpaca_order.id,
                client_order_id=alpaca_order.client_order_id,
                symbol=alpaca_order.symbol,
                side=OrderSide(alpaca_order.side),
                quantity=float(alpaca_order.qty),
                order_type=self._map_alpaca_order_type(alpaca_order.type),
                time_in_force=self._map_alpaca_time_in_force(alpaca_order.time_in_force),
                limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                status=self._map_order_status(alpaca_order.status),
                broker=BrokerType.ALPACA,
                filled_quantity=float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0.0,
                filled_avg_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                created_at=datetime.fromisoformat(alpaca_order.created_at.replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(alpaca_order.updated_at.replace('Z', '+00:00')),
                submitted_at=datetime.fromisoformat(alpaca_order.submitted_at.replace('Z', '+00:00')) if alpaca_order.submitted_at else None,
                filled_at=datetime.fromisoformat(alpaca_order.filled_at.replace('Z', '+00:00')) if alpaca_order.filled_at else None,
                canceled_at=datetime.fromisoformat(alpaca_order.canceled_at.replace('Z', '+00:00')) if alpaca_order.canceled_at else None,
                expired_at=datetime.fromisoformat(alpaca_order.expired_at.replace('Z', '+00:00')) if alpaca_order.expired_at else None
            )
            
            return order
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            raise RuntimeError(f"Alpaca get order failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error getting order from Alpaca: {e}")
            raise RuntimeError(f"Alpaca get order failed: {str(e)}")
    
    async def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """Get orders from broker.
        
        Args:
            status: Filter by order status
            
        Returns:
            List[Order]: List of orders
        """
        try:
            # Map status
            alpaca_status = None
            if status:
                if status == OrderStatus.NEW.value or status == OrderStatus.PENDING.value:
                    alpaca_status = "new"
                elif status == OrderStatus.FILLED.value:
                    alpaca_status = "filled"
                elif status == OrderStatus.CANCELED.value:
                    alpaca_status = "canceled"
                elif status == OrderStatus.EXPIRED.value:
                    alpaca_status = "expired"
                elif status == OrderStatus.REJECTED.value:
                    alpaca_status = "rejected"
                elif status == OrderStatus.PARTIALLY_FILLED.value:
                    alpaca_status = "partially_filled"
            
            # Get orders from Alpaca
            alpaca_orders = await asyncio.to_thread(
                self.api.list_orders,
                status=alpaca_status,
                limit=100
            )
            
            # Convert to Order objects
            orders = []
            for alpaca_order in alpaca_orders:
                order = Order(
                    id="unknown",  # Will be updated by caller
                    broker_order_id=alpaca_order.id,
                    client_order_id=alpaca_order.client_order_id,
                    symbol=alpaca_order.symbol,
                    side=OrderSide(alpaca_order.side),
                    quantity=float(alpaca_order.qty),
                    order_type=self._map_alpaca_order_type(alpaca_order.type),
                    time_in_force=self._map_alpaca_time_in_force(alpaca_order.time_in_force),
                    limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                    stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                    status=self._map_order_status(alpaca_order.status),
                    broker=BrokerType.ALPACA,
                    filled_quantity=float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0.0,
                    filled_avg_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                    created_at=datetime.fromisoformat(alpaca_order.created_at.replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(alpaca_order.updated_at.replace('Z', '+00:00')),
                    submitted_at=datetime.fromisoformat(alpaca_order.submitted_at.replace('Z', '+00:00')) if alpaca_order.submitted_at else None,
                    filled_at=datetime.fromisoformat(alpaca_order.filled_at.replace('Z', '+00:00')) if alpaca_order.filled_at else None,
                    canceled_at=datetime.fromisoformat(alpaca_order.canceled_at.replace('Z', '+00:00')) if alpaca_order.canceled_at else None,
                    expired_at=datetime.fromisoformat(alpaca_order.expired_at.replace('Z', '+00:00')) if alpaca_order.expired_at else None
                )
                orders.append(order)
            
            return orders
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            raise RuntimeError(f"Alpaca get orders failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error getting orders from Alpaca: {e}")
            raise RuntimeError(f"Alpaca get orders failed: {str(e)}")
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from broker.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        try:
            # Get positions from Alpaca
            alpaca_positions = await asyncio.to_thread(
                self.api.list_positions
            )
            
            # Convert to position dictionaries
            positions = []
            for alpaca_position in alpaca_positions:
                position = {
                    "symbol": alpaca_position.symbol,
                    "quantity": float(alpaca_position.qty),
                    "avg_entry_price": float(alpaca_position.avg_entry_price),
                    "current_price": float(alpaca_position.current_price),
                    "market_value": float(alpaca_position.market_value),
                    "cost_basis": float(alpaca_position.cost_basis),
                    "unrealized_pl": float(alpaca_position.unrealized_pl),
                    "unrealized_pl_pct": float(alpaca_position.unrealized_plpc),
                    "asset_class": alpaca_position.asset_class,
                    "asset_type": "stock",  # Alpaca doesn't provide this directly
                    "currency": "USD"  # Alpaca uses USD
                }
                positions.append(position)
            
            return positions
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            raise RuntimeError(f"Alpaca get positions failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error getting positions from Alpaca: {e}")
            raise RuntimeError(f"Alpaca get positions failed: {str(e)}")
    
    async def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio from broker.
        
        Returns:
            Dict[str, Any]: Portfolio information
        """
        try:
            # Get account from Alpaca
            alpaca_account = await asyncio.to_thread(
                self.api.get_account
            )
            
            # Convert to portfolio dictionary
            portfolio = {
                "cash": float(alpaca_account.cash),
                "equity": float(alpaca_account.equity),
                "buying_power": float(alpaca_account.buying_power),
                "currency": "USD",
                "positions": await self.get_positions()
            }
            
            return portfolio
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            raise RuntimeError(f"Alpaca get portfolio failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error getting portfolio from Alpaca: {e}")
            raise RuntimeError(f"Alpaca get portfolio failed: {str(e)}")
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from broker.
        
        Args:
            symbol: Symbol
            
        Returns:
            Dict[str, Any]: Market data
        """
        try:
            # Get last quote from Alpaca
            alpaca_quote = await asyncio.to_thread(
                self.api.get_latest_quote,
                symbol=symbol
            )
            
            # Get last trade from Alpaca
            alpaca_trade = await asyncio.to_thread(
                self.api.get_latest_trade,
                symbol=symbol
            )
            
            # Convert to market data dictionary
            market_data = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "bid": float(alpaca_quote.bp),
                "ask": float(alpaca_quote.ap),
                "bid_size": int(alpaca_quote.bs),
                "ask_size": int(alpaca_quote.as_),
                "last": float(alpaca_trade.price),
                "volume": int(alpaca_trade.volume)
            }
            
            return market_data
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            raise RuntimeError(f"Alpaca get market data failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error getting market data from Alpaca: {e}")
            raise RuntimeError(f"Alpaca get market data failed: {str(e)}")
    
    def _map_order_type(self, order_type: str) -> str:
        """Map order type to Alpaca order type.
        
        Args:
            order_type: Order type
            
        Returns:
            str: Alpaca order type
        """
        if order_type == OrderType.MARKET.value:
            return "market"
        elif order_type == OrderType.LIMIT.value:
            return "limit"
        elif order_type == OrderType.STOP.value:
            return "stop"
        elif order_type == OrderType.STOP_LIMIT.value:
            return "stop_limit"
        elif order_type == OrderType.TRAILING_STOP.value:
            return "trailing_stop"
        else:
            return "market"
    
    def _map_alpaca_order_type(self, alpaca_order_type: str) -> OrderType:
        """Map Alpaca order type to order type.
        
        Args:
            alpaca_order_type: Alpaca order type
            
        Returns:
            OrderType: Order type
        """
        if alpaca_order_type == "market":
            return OrderType.MARKET
        elif alpaca_order_type == "limit":
            return OrderType.LIMIT
        elif alpaca_order_type == "stop":
            return OrderType.STOP
        elif alpaca_order_type == "stop_limit":
            return OrderType.STOP_LIMIT
        elif alpaca_order_type == "trailing_stop":
            return OrderType.TRAILING_STOP
        else:
            return OrderType.MARKET
    
    def _map_time_in_force(self, time_in_force: str) -> str:
        """Map time in force to Alpaca time in force.
        
        Args:
            time_in_force: Time in force
            
        Returns:
            str: Alpaca time in force
        """
        if time_in_force == TimeInForce.DAY.value:
            return "day"
        elif time_in_force == TimeInForce.GTC.value:
            return "gtc"
        elif time_in_force == TimeInForce.IOC.value:
            return "ioc"
        elif time_in_force == TimeInForce.FOK.value:
            return "fok"
        else:
            return "day"
    
    def _map_alpaca_time_in_force(self, alpaca_time_in_force: str) -> TimeInForce:
        """Map Alpaca time in force to time in force.
        
        Args:
            alpaca_time_in_force: Alpaca time in force
            
        Returns:
            TimeInForce: Time in force
        """
        if alpaca_time_in_force == "day":
            return TimeInForce.DAY
        elif alpaca_time_in_force == "gtc":
            return TimeInForce.GTC
        elif alpaca_time_in_force == "ioc":
            return TimeInForce.IOC
        elif alpaca_time_in_force == "fok":
            return TimeInForce.FOK
        else:
            return TimeInForce.DAY
    
    def _map_order_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to order status.
        
        Args:
            alpaca_status: Alpaca order status
            
        Returns:
            OrderStatus: Order status
        """
        if alpaca_status == "new":
            return OrderStatus.ACCEPTED
        elif alpaca_status == "filled":
            return OrderStatus.FILLED
        elif alpaca_status == "partially_filled":
            return OrderStatus.PARTIALLY_FILLED
        elif alpaca_status == "canceled":
            return OrderStatus.CANCELED
        elif alpaca_status == "expired":
            return OrderStatus.EXPIRED
        elif alpaca_status == "rejected":
            return OrderStatus.REJECTED
        elif alpaca_status == "pending_new":
            return OrderStatus.PENDING
        elif alpaca_status == "accepted":
            return OrderStatus.ACCEPTED
        else:
            return OrderStatus.NEW
