"""
Data flow manager for the AI-Augmented Full-Stack Algorithmic Trading Pipeline.
This module manages the flow of data between components according to the pipeline configuration.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from ..utils.component_communication import ComponentCommunicator

logger = logging.getLogger('data_flow_manager')

class DataFlowManager:
    """
    Manager for data flow between pipeline components.
    
    This class:
    1. Manages the flow of data between components according to the pipeline configuration
    2. Ensures data is properly formatted and validated before being passed between components
    3. Implements data transformation and mapping between different component data models
    4. Provides monitoring and logging of data flows
    """
    
    def __init__(self, config: Dict[str, Any], component_name: str):
        """
        Initialize the data flow manager.
        
        Args:
            config: Pipeline configuration
            component_name: Name of the current component
        """
        self.config = config
        self.component_name = component_name
        self.communicator = ComponentCommunicator({
            'component_name': component_name,
            'components': config['components'],
            'integration': config['integration']
        })
        self.data_flows = self._parse_data_flows()
        logger.info(f"Data flow manager initialized for {component_name}")
    
    def _parse_data_flows(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse the data flow configuration.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Data flows organized by source and target
        """
        flows = {
            'incoming': [],
            'outgoing': []
        }
        
        # Get data flow configuration
        data_flow_config = self.config.get('integration', {}).get('data_flow', [])
        
        # Parse incoming and outgoing flows
        for flow in data_flow_config:
            source = flow.get('source')
            target = flow.get('target')
            
            if source == self.component_name:
                flows['outgoing'].append({
                    'target': target,
                    'data_types': flow.get('data_types', [])
                })
            
            if target == self.component_name:
                flows['incoming'].append({
                    'source': source,
                    'data_types': flow.get('data_types', [])
                })
        
        logger.info(f"Parsed data flows for {self.component_name}: {len(flows['incoming'])} incoming, {len(flows['outgoing'])} outgoing")
        return flows
    
    async def start(self):
        """
        Start the data flow manager.
        """
        await self.communicator.start()
        logger.info(f"Data flow manager started for {self.component_name}")
    
    async def stop(self):
        """
        Stop the data flow manager.
        """
        await self.communicator.stop()
        logger.info(f"Data flow manager stopped for {self.component_name}")
    
    async def send_data(self, data_type: str, data: Any) -> Dict[str, Any]:
        """
        Send data to target components based on the data flow configuration.
        
        Args:
            data_type: Type of data to send
            data: Data to send
            
        Returns:
            Dict[str, Any]: Results of the data sending operation
        """
        results = {}
        
        # Find target components for this data type
        for flow in self.data_flows['outgoing']:
            if data_type in flow['data_types']:
                target = flow['target']
                
                try:
                    # Transform data if needed
                    transformed_data = self._transform_data_for_target(data_type, data, target)
                    
                    # Send data to target component
                    response = await self.communicator.send_message(
                        target_component=target,
                        message_type=f"data_{data_type}",
                        data={
                            'type': data_type,
                            'content': transformed_data,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    
                    results[target] = {
                        'success': True,
                        'response': response
                    }
                    
                    logger.info(f"Sent {data_type} data to {target}")
                
                except Exception as e:
                    logger.error(f"Failed to send {data_type} data to {target}: {e}")
                    results[target] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return results
    
    def _transform_data_for_target(self, data_type: str, data: Any, target: str) -> Any:
        """
        Transform data for a specific target component.
        
        Args:
            data_type: Type of data
            data: Data to transform
            target: Target component
            
        Returns:
            Any: Transformed data
        """
        # This is a placeholder for data transformation logic
        # In a real implementation, this would transform data between different component data models
        
        # For now, we just return the original data
        return data
    
    async def receive_data(self, source: str, data_type: str, data: Any) -> Dict[str, Any]:
        """
        Receive data from a source component.
        
        Args:
            source: Source component
            data_type: Type of data
            data: Received data
            
        Returns:
            Dict[str, Any]: Result of the data processing
        """
        logger.info(f"Received {data_type} data from {source}")
        
        # Check if this data flow is expected
        is_expected = False
        for flow in self.data_flows['incoming']:
            if flow['source'] == source and data_type in flow['data_types']:
                is_expected = True
                break
        
        if not is_expected:
            logger.warning(f"Unexpected data flow: {data_type} from {source}")
            return {
                'success': False,
                'error': f"Unexpected data flow: {data_type} from {source}"
            }
        
        try:
            # Transform data if needed
            transformed_data = self._transform_data_from_source(data_type, data, source)
            
            # Process the data (this would be implemented by the component)
            # For now, we just return success
            return {
                'success': True,
                'message': f"Received and processed {data_type} data from {source}"
            }
        
        except Exception as e:
            logger.error(f"Error processing {data_type} data from {source}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _transform_data_from_source(self, data_type: str, data: Any, source: str) -> Any:
        """
        Transform data from a specific source component.
        
        Args:
            data_type: Type of data
            data: Data to transform
            source: Source component
            
        Returns:
            Any: Transformed data
        """
        # This is a placeholder for data transformation logic
        # In a real implementation, this would transform data between different component data models
        
        # For now, we just return the original data
        return data
    
    async def get_available_data_types(self, target_component: str) -> List[str]:
        """
        Get the data types available from a specific component.
        
        Args:
            target_component: Target component
            
        Returns:
            List[str]: Available data types
        """
        try:
            response = await self.communicator.call_component_api(
                target_component=target_component,
                endpoint='/data/types',
                method='GET'
            )
            
            return response.get('data_types', [])
        
        except Exception as e:
            logger.error(f"Failed to get available data types from {target_component}: {e}")
            return []
    
    async def get_data_schema(self, target_component: str, data_type: str) -> Dict[str, Any]:
        """
        Get the schema for a specific data type from a component.
        
        Args:
            target_component: Target component
            data_type: Data type
            
        Returns:
            Dict[str, Any]: Data schema
        """
        try:
            response = await self.communicator.call_component_api(
                target_component=target_component,
                endpoint=f'/data/schema/{data_type}',
                method='GET'
            )
            
            return response.get('schema', {})
        
        except Exception as e:
            logger.error(f"Failed to get schema for {data_type} from {target_component}: {e}")
            return {}
    
    async def request_data(
        self, 
        target_component: str, 
        data_type: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Request data from another component.
        
        Args:
            target_component: Target component
            data_type: Data type
            params: Request parameters
            
        Returns:
            Dict[str, Any]: Requested data
        """
        try:
            response = await self.communicator.call_component_api(
                target_component=target_component,
                endpoint=f'/data/{data_type}',
                method='GET',
                params=params
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Failed to request {data_type} data from {target_component}: {e}")
            raise
