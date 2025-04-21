"""
Main integration module for the AI-Augmented Full-Stack Algorithmic Trading Pipeline.
This module connects all components of the pipeline and manages their interactions.
"""

import os
import sys
import logging
import asyncio
import signal
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import yaml
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/ubuntu/trading_pipeline/logs/pipeline.log')
    ]
)

logger = logging.getLogger('pipeline_integrator')

class PipelineIntegrator:
    """
    Main class for integrating all components of the trading pipeline.
    
    This class:
    1. Initializes all pipeline components
    2. Manages communication between components
    3. Coordinates the overall pipeline workflow
    4. Handles error recovery and fault tolerance
    5. Provides monitoring and status reporting
    """
    
    def __init__(self, config_path: str = '/home/ubuntu/trading_pipeline/config/pipeline.yaml'):
        """
        Initialize the pipeline integrator.
        
        Args:
            config_path: Path to the pipeline configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.components = {}
        self.component_processes = {}
        self.message_queues = {}
        self.status = {
            'data_ingestion': 'stopped',
            'semantic_signal': 'stopped',
            'strategy_generator': 'stopped',
            'execution_engine': 'stopped',
            'risk_management': 'stopped',
            'agentic_oversight': 'stopped',
            'dashboard': 'stopped'
        }
        self.metrics = {
            'start_time': None,
            'processed_data_points': 0,
            'generated_signals': 0,
            'executed_trades': 0,
            'active_strategies': 0,
            'errors': 0,
            'warnings': 0
        }
        self.event_loop = None
        
        # Create message queues for inter-component communication
        self._setup_message_queues()
        
        logger.info("Pipeline integrator initialized")
    
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
                    'data_ingestion': {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 8001,
                        'api_prefix': '/api/v1',
                        'polling_interval': 60,
                        'sources': ['alpaca', 'yahoo_finance', 'newsapi', 'fred']
                    },
                    'semantic_signal': {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 8002,
                        'api_prefix': '/api/v1',
                        'processing_interval': 300
                    },
                    'strategy_generator': {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 8003,
                        'api_prefix': '/api/v1',
                        'optimization_interval': 3600
                    },
                    'execution_engine': {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 8004,
                        'api_prefix': '/api/v1',
                        'simulation_mode': True
                    },
                    'risk_management': {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 8005,
                        'api_prefix': '/api/v1',
                        'risk_check_interval': 60
                    },
                    'agentic_oversight': {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 8006,
                        'api_prefix': '/api/v1',
                        'monitoring_interval': 60
                    },
                    'dashboard': {
                        'enabled': True,
                        'host': 'localhost',
                        'port': 8000,
                        'api_prefix': '/api'
                    }
                },
                'integration': {
                    'startup_sequence': [
                        'data_ingestion',
                        'semantic_signal',
                        'strategy_generator',
                        'risk_management',
                        'execution_engine',
                        'agentic_oversight',
                        'dashboard'
                    ],
                    'shutdown_sequence': [
                        'execution_engine',
                        'strategy_generator',
                        'risk_management',
                        'semantic_signal',
                        'data_ingestion',
                        'agentic_oversight',
                        'dashboard'
                    ],
                    'health_check_interval': 30,
                    'restart_attempts': 3,
                    'restart_delay': 5
                }
            }
    
    def _setup_message_queues(self):
        """
        Set up message queues for inter-component communication.
        """
        # Create a message queue for each component
        for component in self.config['components']:
            self.message_queues[component] = {
                'in': queue.Queue(),
                'out': queue.Queue()
            }
        
        logger.info("Message queues set up for all components")
    
    async def start(self):
        """
        Start the pipeline and all its components.
        """
        if self.running:
            logger.warning("Pipeline is already running")
            return
        
        logger.info("Starting pipeline...")
        self.running = True
        self.metrics['start_time'] = datetime.now()
        
        # Create data directories if they don't exist
        os.makedirs(self.config['general']['data_dir'], exist_ok=True)
        os.makedirs('/home/ubuntu/trading_pipeline/logs', exist_ok=True)
        
        # Start components in the specified order
        for component_name in self.config['integration']['startup_sequence']:
            if self.config['components'][component_name]['enabled']:
                await self._start_component(component_name)
            else:
                logger.info(f"Component {component_name} is disabled, skipping")
        
        # Start the health check loop
        self.event_loop = asyncio.get_event_loop()
        self.event_loop.create_task(self._health_check_loop())
        
        logger.info("Pipeline started successfully")
    
    async def _start_component(self, component_name: str):
        """
        Start a specific component.
        
        Args:
            component_name: Name of the component to start
        """
        logger.info(f"Starting component: {component_name}")
        
        try:
            # Import the component module
            module_path = f"/home/ubuntu/trading_pipeline/{component_name}/main.py"
            
            # Check if the module exists
            if not os.path.exists(module_path):
                logger.error(f"Component module not found: {module_path}")
                self.status[component_name] = 'error'
                return
            
            # Start the component as a subprocess
            component_config = self.config['components'][component_name]
            cmd = [
                sys.executable,
                module_path,
                '--host', component_config['host'],
                '--port', str(component_config['port']),
                '--log-level', self.config['general']['log_level']
            ]
            
            # Add component-specific arguments
            if component_name == 'execution_engine' and 'simulation_mode' in component_config:
                cmd.extend(['--simulation', str(component_config['simulation_mode']).lower()])
            
            # Start the process
            import subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.component_processes[component_name] = process
            
            # Wait for the component to start
            await asyncio.sleep(2)
            
            # Check if the process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Component {component_name} failed to start: {stderr}")
                self.status[component_name] = 'error'
                return
            
            # Start a thread to read the process output
            threading.Thread(
                target=self._read_process_output,
                args=(component_name, process),
                daemon=True
            ).start()
            
            self.status[component_name] = 'running'
            logger.info(f"Component {component_name} started successfully")
            
        except Exception as e:
            logger.error(f"Error starting component {component_name}: {e}")
            self.status[component_name] = 'error'
    
    def _read_process_output(self, component_name: str, process: Any):
        """
        Read and log the output from a component process.
        
        Args:
            component_name: Name of the component
            process: Component process
        """
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"[{component_name}] {line.strip()}")
        
        for line in iter(process.stderr.readline, ''):
            if line:
                logger.error(f"[{component_name}] {line.strip()}")
                self.metrics['errors'] += 1
    
    async def _health_check_loop(self):
        """
        Periodically check the health of all components and restart if necessary.
        """
        while self.running:
            try:
                for component_name, process in self.component_processes.items():
                    # Check if the process is still running
                    if process.poll() is not None:
                        logger.warning(f"Component {component_name} has stopped unexpectedly")
                        self.status[component_name] = 'stopped'
                        
                        # Attempt to restart the component
                        restart_attempts = self.config['integration']['restart_attempts']
                        restart_delay = self.config['integration']['restart_delay']
                        
                        for attempt in range(restart_attempts):
                            logger.info(f"Attempting to restart {component_name} (attempt {attempt + 1}/{restart_attempts})")
                            await asyncio.sleep(restart_delay)
                            await self._start_component(component_name)
                            
                            # Check if restart was successful
                            if self.status[component_name] == 'running':
                                logger.info(f"Component {component_name} restarted successfully")
                                break
                        
                        if self.status[component_name] != 'running':
                            logger.error(f"Failed to restart component {component_name} after {restart_attempts} attempts")
                
                # Check component health via API
                await self._check_component_health()
                
                # Wait for the next health check
                await asyncio.sleep(self.config['integration']['health_check_interval'])
            
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.config['integration']['health_check_interval'])
    
    async def _check_component_health(self):
        """
        Check the health of all components via their health check API endpoints.
        """
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            for component_name, component_config in self.config['components'].items():
                if not component_config['enabled'] or self.status[component_name] != 'running':
                    continue
                
                try:
                    url = f"http://{component_config['host']}:{component_config['port']}{component_config['api_prefix']}/health"
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            logger.debug(f"Component {component_name} health check: {health_data}")
                            
                            # Update component status based on health check
                            if health_data.get('status') == 'healthy':
                                self.status[component_name] = 'running'
                            else:
                                self.status[component_name] = 'degraded'
                                logger.warning(f"Component {component_name} is in degraded state: {health_data.get('message')}")
                        else:
                            logger.warning(f"Component {component_name} health check failed with status {response.status}")
                            self.status[component_name] = 'degraded'
                
                except Exception as e:
                    logger.warning(f"Error checking health of component {component_name}: {e}")
                    # Don't change status based on a single failed health check
    
    async def stop(self):
        """
        Stop the pipeline and all its components.
        """
        if not self.running:
            logger.warning("Pipeline is not running")
            return
        
        logger.info("Stopping pipeline...")
        self.running = False
        
        # Stop components in the specified order
        for component_name in self.config['integration']['shutdown_sequence']:
            if component_name in self.component_processes:
                await self._stop_component(component_name)
        
        logger.info("Pipeline stopped successfully")
    
    async def _stop_component(self, component_name: str):
        """
        Stop a specific component.
        
        Args:
            component_name: Name of the component to stop
        """
        logger.info(f"Stopping component: {component_name}")
        
        try:
            process = self.component_processes.get(component_name)
            if process:
                # Try to terminate gracefully
                process.terminate()
                
                # Wait for the process to terminate
                try:
                    await asyncio.wait_for(
                        asyncio.create_subprocess_exec(
                            *['wait', str(process.pid)]
                        ),
                        timeout=5
                    )
                except asyncio.TimeoutError:
                    # Force kill if it doesn't terminate
                    logger.warning(f"Component {component_name} did not terminate gracefully, forcing kill")
                    process.kill()
                
                self.status[component_name] = 'stopped'
                logger.info(f"Component {component_name} stopped successfully")
            else:
                logger.warning(f"Component {component_name} is not running")
        
        except Exception as e:
            logger.error(f"Error stopping component {component_name}: {e}")
    
    async def restart(self):
        """
        Restart the pipeline and all its components.
        """
        logger.info("Restarting pipeline...")
        await self.stop()
        await asyncio.sleep(5)  # Wait for all components to fully stop
        await self.start()
        logger.info("Pipeline restarted successfully")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the pipeline and all its components.
        
        Returns:
            Dict[str, Any]: Pipeline status
        """
        return {
            'running': self.running,
            'components': self.status,
            'metrics': {
                **self.metrics,
                'uptime': str(datetime.now() - self.metrics['start_time']) if self.metrics['start_time'] else None
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def send_message(self, source: str, target: str, message: Dict[str, Any]):
        """
        Send a message from one component to another.
        
        Args:
            source: Source component name
            target: Target component name
            message: Message to send
        """
        if target not in self.message_queues:
            logger.error(f"Invalid target component: {target}")
            return
        
        # Add metadata to the message
        message['_metadata'] = {
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'message_id': f"{source}-{target}-{int(time.time() * 1000)}"
        }
        
        # Put the message in the target component's input queue
        self.message_queues[target]['in'].put(message)
        logger.debug(f"Message sent from {source} to {target}: {message}")
    
    async def receive_message(self, target: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Receive a message for a specific component.
        
        Args:
            target: Target component name
            timeout: Timeout in seconds
            
        Returns:
            Optional[Dict[str, Any]]: Received message or None if timeout
        """
        if target not in self.message_queues:
            logger.error(f"Invalid target component: {target}")
            return None
        
        try:
            # Get a message from the component's input queue
            message = self.message_queues[target]['in'].get(timeout=timeout)
            logger.debug(f"Message received by {target}: {message}")
            return message
        except queue.Empty:
            return None
    
    def handle_signal(self, signum, frame):
        """
        Handle system signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        if signum == signal.SIGINT or signum == signal.SIGTERM:
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.stop())

# Main entry point
async def main():
    """
    Main entry point for the pipeline integrator.
    """
    # Create the pipeline integrator
    integrator = PipelineIntegrator()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, integrator.handle_signal)
    signal.signal(signal.SIGTERM, integrator.handle_signal)
    
    # Start the pipeline
    await integrator.start()
    
    try:
        # Keep the main thread running
        while integrator.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        await integrator.stop()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
