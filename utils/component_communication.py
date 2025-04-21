"""
Component communication module for the AI-Augmented Full-Stack Algorithmic Trading Pipeline.
This module provides utilities for inter-component communication.
"""

import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger('component_communication')

class ComponentCommunicator:
    """
    Utility class for communication between pipeline components.
    
    This class:
    1. Provides methods for sending and receiving messages between components
    2. Handles API calls between components
    3. Implements retry logic and error handling
    4. Provides serialization and deserialization of messages
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the component communicator.
        
        Args:
            config: Component configuration
        """
        self.config = config
        self.component_name = config.get('component_name', 'unknown')
        self.session = None
        logger.info(f"Component communicator initialized for {self.component_name}")
    
    async def start(self):
        """
        Start the component communicator.
        """
        self.session = aiohttp.ClientSession()
        logger.info(f"Component communicator started for {self.component_name}")
    
    async def stop(self):
        """
        Stop the component communicator.
        """
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Component communicator stopped for {self.component_name}")
    
    async def call_component_api(
        self, 
        target_component: str, 
        endpoint: str, 
        method: str = 'GET', 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Call an API endpoint on another component.
        
        Args:
            target_component: Name of the target component
            endpoint: API endpoint path
            method: HTTP method (GET, POST, PUT, DELETE)
            data: Request data (for POST, PUT)
            params: Query parameters (for GET)
            headers: HTTP headers
            timeout: Request timeout in seconds
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            Exception: If the API call fails after all retries
        """
        if not self.session:
            await self.start()
        
        # Get target component configuration
        if target_component not in self.config.get('components', {}):
            raise ValueError(f"Unknown target component: {target_component}")
        
        target_config = self.config['components'][target_component]
        
        # Build the URL
        url = f"http://{target_config['host']}:{target_config['port']}{target_config['api_prefix']}{endpoint}"
        
        # Set default headers
        if headers is None:
            headers = {}
        headers['Content-Type'] = 'application/json'
        headers['X-Source-Component'] = self.component_name
        
        # Set default timeout
        if timeout is None:
            timeout = self.config.get('integration', {}).get('message_timeout', 10)
        
        # Retry logic
        for attempt in range(retries):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=headers,
                    timeout=timeout
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"API call to {target_component} failed with status {response.status}: {error_text}"
                            f" (attempt {attempt + 1}/{retries})"
                        )
                        
                        if attempt + 1 < retries:
                            await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        else:
                            raise Exception(
                                f"API call to {target_component} failed with status {response.status}: {error_text}"
                            )
            
            except asyncio.TimeoutError:
                logger.warning(
                    f"API call to {target_component} timed out (attempt {attempt + 1}/{retries})"
                )
                
                if attempt + 1 < retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise Exception(f"API call to {target_component} timed out after {retries} attempts")
            
            except Exception as e:
                logger.warning(
                    f"API call to {target_component} failed: {str(e)} (attempt {attempt + 1}/{retries})"
                )
                
                if attempt + 1 < retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
    
    async def send_message(
        self, 
        target_component: str, 
        message_type: str, 
        data: Dict[str, Any],
        priority: str = 'normal'
    ) -> Dict[str, Any]:
        """
        Send a message to another component.
        
        Args:
            target_component: Name of the target component
            message_type: Type of message
            data: Message data
            priority: Message priority (low, normal, high, critical)
            
        Returns:
            Dict[str, Any]: Response from the target component
        """
        # Prepare the message
        message = {
            'type': message_type,
            'data': data,
            'metadata': {
                'source': self.component_name,
                'timestamp': datetime.now().isoformat(),
                'priority': priority,
                'id': f"{self.component_name}-{target_component}-{message_type}-{int(datetime.now().timestamp() * 1000)}"
            }
        }
        
        # Send the message via the component's message endpoint
        return await self.call_component_api(
            target_component=target_component,
            endpoint='/messages',
            method='POST',
            data=message
        )
    
    async def broadcast_message(
        self, 
        message_type: str, 
        data: Dict[str, Any],
        target_components: Optional[List[str]] = None,
        priority: str = 'normal'
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Broadcast a message to multiple components.
        
        Args:
            message_type: Type of message
            data: Message data
            target_components: List of target components (None for all)
            priority: Message priority (low, normal, high, critical)
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Responses from the target components
        """
        if target_components is None:
            # Broadcast to all components except self
            target_components = [
                component for component in self.config.get('components', {})
                if component != self.component_name
            ]
        
        # Send the message to each target component
        results = {}
        for target_component in target_components:
            try:
                response = await self.send_message(
                    target_component=target_component,
                    message_type=message_type,
                    data=data,
                    priority=priority
                )
                results[target_component] = {
                    'success': True,
                    'response': response
                }
            except Exception as e:
                logger.error(f"Failed to send message to {target_component}: {e}")
                results[target_component] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    async def get_component_health(self, target_component: str) -> Dict[str, Any]:
        """
        Get the health status of another component.
        
        Args:
            target_component: Name of the target component
            
        Returns:
            Dict[str, Any]: Health status of the target component
        """
        try:
            return await self.call_component_api(
                target_component=target_component,
                endpoint='/health',
                method='GET'
            )
        except Exception as e:
            logger.error(f"Failed to get health status of {target_component}: {e}")
            return {
                'status': 'unknown',
                'message': f"Failed to get health status: {str(e)}"
            }
    
    async def get_all_components_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the health status of all components.
        
        Returns:
            Dict[str, Dict[str, Any]]: Health status of all components
        """
        results = {}
        for component in self.config.get('components', {}):
            if component != self.component_name:
                results[component] = await self.get_component_health(component)
        
        return results
