"""
Component base class for the AI-Augmented Full-Stack Algorithmic Trading Pipeline.
This module provides a base class for all pipeline components with common functionality.
"""

import os
import sys
import logging
import asyncio
import signal
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..utils.component_communication import ComponentCommunicator
from ..utils.data_flow import DataFlowManager

logger = logging.getLogger('component_base')

class ComponentHealth(BaseModel):
    """Component health response model."""
    status: str
    version: str
    uptime: str
    component: str
    details: Dict[str, Any]

class ComponentBase:
    """
    Base class for all pipeline components.
    
    This class:
    1. Provides common functionality for all components
    2. Handles component lifecycle (initialization, startup, shutdown)
    3. Sets up API endpoints for health checks and component communication
    4. Manages configuration and logging
    5. Provides utilities for data flow and component communication
    """
    
    def __init__(
        self, 
        component_name: str,
        config_path: str = '/home/ubuntu/trading_pipeline/config/pipeline.yaml',
        host: str = None,
        port: int = None,
        log_level: str = None
    ):
        """
        Initialize the component.
        
        Args:
            component_name: Name of the component
            config_path: Path to the pipeline configuration file
            host: Host to bind to (overrides config)
            port: Port to bind to (overrides config)
            log_level: Logging level (overrides config)
        """
        self.component_name = component_name
        self.config_path = config_path
        self.config = self._load_config()
        
        # Override config with command line arguments
        if host:
            self.config['components'][component_name]['host'] = host
        if port:
            self.config['components'][component_name]['port'] = port
        if log_level:
            self.config['general']['log_level'] = log_level
        
        # Set up logging
        self._setup_logging()
        
        # Initialize component state
        self.running = False
        self.start_time = None
        self.app = self._create_app()
        self.communicator = ComponentCommunicator({
            'component_name': component_name,
            'components': self.config['components'],
            'integration': self.config['integration']
        })
        self.data_flow = DataFlowManager(self.config, component_name)
        
        logger.info(f"Component {component_name} initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the pipeline configuration from the YAML file.
        
        Returns:
            Dict[str, Any]: Pipeline configuration
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration
            return {
                'general': {
                    'environment': 'development',
                    'log_level': 'INFO',
                    'data_dir': '/home/ubuntu/trading_pipeline/data',
                },
                'components': {
                    self.component_name: {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 8000,
                        'api_prefix': '/api/v1'
                    }
                },
                'integration': {
                    'health_check_interval': 30,
                    'restart_attempts': 3,
                    'restart_delay': 5,
                    'data_flow': []
                }
            }
    
    def _setup_logging(self):
        """
        Set up logging for the component.
        """
        log_level = getattr(logging, self.config['general']['log_level'])
        log_dir = '/home/ubuntu/trading_pipeline/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{log_dir}/{self.component_name}.log')
            ]
        )
    
    def _create_app(self) -> FastAPI:
        """
        Create the FastAPI application.
        
        Returns:
            FastAPI: FastAPI application
        """
        app = FastAPI(
            title=f"{self.component_name.replace('_', ' ').title()} API",
            description=f"API for the {self.component_name.replace('_', ' ').title()} component of the AI-Augmented Full-Stack Algorithmic Trading Pipeline",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add health check endpoint
        @app.get("/health", response_model=ComponentHealth)
        async def health_check():
            """
            Health check endpoint.
            
            Returns:
                ComponentHealth: Component health status
            """
            return {
                "status": "healthy" if self.running else "stopped",
                "version": "1.0.0",
                "uptime": str(datetime.now() - self.start_time) if self.start_time else "0",
                "component": self.component_name,
                "details": self.get_health_details()
            }
        
        # Add message endpoint
        @app.post("/messages")
        async def receive_message(message: Dict[str, Any]):
            """
            Receive a message from another component.
            
            Args:
                message: Message data
                
            Returns:
                Dict[str, Any]: Message processing result
            """
            try:
                message_type = message.get('type')
                data = message.get('data', {})
                metadata = message.get('metadata', {})
                source = metadata.get('source', 'unknown')
                
                logger.info(f"Received message of type {message_type} from {source}")
                
                # Handle data messages
                if message_type.startswith('data_'):
                    data_type = message_type.replace('data_', '')
                    return await self.data_flow.receive_data(source, data_type, data)
                
                # Handle other message types
                return await self.handle_message(message_type, data, metadata)
            
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        # Add data types endpoint
        @app.get("/data/types")
        async def get_data_types():
            """
            Get available data types.
            
            Returns:
                Dict[str, List[str]]: Available data types
            """
            return {
                "data_types": self.get_available_data_types()
            }
        
        # Add data schema endpoint
        @app.get("/data/schema/{data_type}")
        async def get_data_schema(data_type: str):
            """
            Get schema for a specific data type.
            
            Args:
                data_type: Data type
                
            Returns:
                Dict[str, Any]: Data schema
            """
            schema = self.get_data_schema(data_type)
            if not schema:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Schema for data type {data_type} not found"
                )
            
            return {
                "data_type": data_type,
                "schema": schema
            }
        
        return app
    
    async def start(self):
        """
        Start the component.
        """
        if self.running:
            logger.warning(f"Component {self.component_name} is already running")
            return
        
        logger.info(f"Starting component {self.component_name}...")
        self.running = True
        self.start_time = datetime.now()
        
        # Start the communicator and data flow manager
        await self.communicator.start()
        await self.data_flow.start()
        
        # Perform component-specific initialization
        await self.initialize()
        
        logger.info(f"Component {self.component_name} started successfully")
    
    async def stop(self):
        """
        Stop the component.
        """
        if not self.running:
            logger.warning(f"Component {self.component_name} is not running")
            return
        
        logger.info(f"Stopping component {self.component_name}...")
        self.running = False
        
        # Perform component-specific cleanup
        await self.cleanup()
        
        # Stop the data flow manager and communicator
        await self.data_flow.stop()
        await self.communicator.stop()
        
        logger.info(f"Component {self.component_name} stopped successfully")
    
    async def run(self):
        """
        Run the component.
        """
        import uvicorn
        
        # Start the component
        await self.start()
        
        # Get component configuration
        component_config = self.config['components'][self.component_name]
        host = component_config['host']
        port = component_config['port']
        
        # Run the FastAPI application
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level=self.config['general']['log_level'].lower(),
            reload=self.config['general']['environment'] == 'development'
        )
        server = uvicorn.Server(config)
        
        # Handle signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)
        
        # Run the server
        await server.serve()
    
    def _handle_signal(self, signum, frame):
        """
        Handle system signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())
    
    async def initialize(self):
        """
        Initialize the component.
        This method should be overridden by subclasses.
        """
        pass
    
    async def cleanup(self):
        """
        Clean up the component.
        This method should be overridden by subclasses.
        """
        pass
    
    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information.
        This method should be overridden by subclasses.
        
        Returns:
            Dict[str, Any]: Health details
        """
        return {}
    
    async def handle_message(self, message_type: str, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a message from another component.
        This method should be overridden by subclasses.
        
        Args:
            message_type: Type of message
            data: Message data
            metadata: Message metadata
            
        Returns:
            Dict[str, Any]: Message handling result
        """
        return {
            "success": False,
            "error": f"Message type {message_type} not supported"
        }
    
    def get_available_data_types(self) -> List[str]:
        """
        Get available data types.
        This method should be overridden by subclasses.
        
        Returns:
            List[str]: Available data types
        """
        return []
    
    def get_data_schema(self, data_type: str) -> Dict[str, Any]:
        """
        Get schema for a specific data type.
        This method should be overridden by subclasses.
        
        Args:
            data_type: Data type
            
        Returns:
            Dict[str, Any]: Data schema
        """
        return {}
