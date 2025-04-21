"""
Broker factory for creating broker clients.
"""

import logging
from typing import Dict, Any

from ..config.settings import settings
from ..brokers.alpaca_broker import AlpacaBroker
from ..brokers.interactive_brokers import InteractiveBrokersBroker
from ..brokers.binance_broker import BinanceBroker
from ..brokers.simulation_broker import SimulationBroker

logger = logging.getLogger(__name__)

class BrokerFactory:
    """Factory for creating broker clients."""
    
    def __init__(self):
        """Initialize broker factory."""
        self.brokers = {}
        self.broker_settings = settings.brokers
        self.simulation_enabled = settings.simulation.enabled
    
    def get_broker(self, broker_type: str) -> Any:
        """Get broker client.
        
        Args:
            broker_type: Broker type
            
        Returns:
            Any: Broker client
            
        Raises:
            ValueError: If broker type is not supported
        """
        # Check if broker is already initialized
        if broker_type in self.brokers:
            return self.brokers[broker_type]
        
        # Check if simulation mode is enabled
        if self.simulation_enabled:
            logger.info(f"Using simulation broker for {broker_type}")
            broker = SimulationBroker()
            self.brokers[broker_type] = broker
            return broker
        
        # Create broker based on type
        if broker_type == "alpaca":
            broker = self._create_alpaca_broker()
        elif broker_type == "interactive_brokers":
            broker = self._create_interactive_brokers_broker()
        elif broker_type == "binance":
            broker = self._create_binance_broker()
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
        
        # Cache broker
        self.brokers[broker_type] = broker
        
        return broker
    
    def _create_alpaca_broker(self) -> AlpacaBroker:
        """Create Alpaca broker client.
        
        Returns:
            AlpacaBroker: Alpaca broker client
        """
        config = self.broker_settings.alpaca
        
        broker = AlpacaBroker(
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            base_url=config["base_url"],
            data_url=config["data_url"],
            use_sandbox=config["use_sandbox"]
        )
        
        logger.info("Alpaca broker client created")
        
        return broker
    
    def _create_interactive_brokers_broker(self) -> InteractiveBrokersBroker:
        """Create Interactive Brokers broker client.
        
        Returns:
            InteractiveBrokersBroker: Interactive Brokers broker client
        """
        config = self.broker_settings.interactive_brokers
        
        broker = InteractiveBrokersBroker(
            host=config["host"],
            port=config["port"],
            client_id=config["client_id"],
            timeout=config["timeout"],
            use_sandbox=config["use_sandbox"]
        )
        
        logger.info("Interactive Brokers broker client created")
        
        return broker
    
    def _create_binance_broker(self) -> BinanceBroker:
        """Create Binance broker client.
        
        Returns:
            BinanceBroker: Binance broker client
        """
        config = self.broker_settings.binance
        
        broker = BinanceBroker(
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            testnet=config["testnet"]
        )
        
        logger.info("Binance broker client created")
        
        return broker
