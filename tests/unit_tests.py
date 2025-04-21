"""
Unit tests for the AI-Augmented Full-Stack Algorithmic Trading Pipeline.
This module provides unit tests for individual components and utilities.
"""

import os
import sys
import unittest
import pytest
import asyncio
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import json
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/ubuntu/trading_pipeline/logs/unit_tests.log')
    ]
)

logger = logging.getLogger('pipeline_unit_tests')

# Add the project root to the Python path
sys.path.append('/home/ubuntu/trading_pipeline')

# Import pipeline modules
from utils.component_communication import ComponentCommunicator
from utils.data_flow import DataFlowManager
from utils.component_base import ComponentBase

class TestComponentCommunication(unittest.TestCase):
    """
    Unit tests for the ComponentCommunicator class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'component_name': 'test_component',
            'components': {
                'data_ingestion': {
                    'host': 'localhost',
                    'port': 8001,
                    'api_prefix': '/api/v1'
                },
                'semantic_signal': {
                    'host': 'localhost',
                    'port': 8002,
                    'api_prefix': '/api/v1'
                }
            },
            'integration': {
                'message_timeout': 5
            }
        }
        self.communicator = ComponentCommunicator(self.config)
    
    @patch('aiohttp.ClientSession.request')
    async def test_call_component_api_success(self, mock_request):
        """Test successful API call to another component."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = asyncio.coroutine(lambda: {'status': 'success'})
        mock_response.__aenter__.return_value = mock_response
        mock_request.return_value = mock_response
        
        # Start the communicator
        await self.communicator.start()
        
        # Call the API
        result = await self.communicator.call_component_api(
            target_component='data_ingestion',
            endpoint='/test',
            method='GET'
        )
        
        # Verify the result
        self.assertEqual(result, {'status': 'success'})
        
        # Verify the request was made with the correct parameters
        mock_request.assert_called_once_with(
            method='GET',
            url='http://localhost:8001/api/v1/test',
            json=None,
            params=None,
            headers={
                'Content-Type': 'application/json',
                'X-Source-Component': 'test_component'
            },
            timeout=5
        )
        
        # Stop the communicator
        await self.communicator.stop()
    
    @patch('aiohttp.ClientSession.request')
    async def test_call_component_api_error(self, mock_request):
        """Test API call with error response."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = asyncio.coroutine(lambda: 'Internal Server Error')
        mock_response.__aenter__.return_value = mock_response
        mock_request.return_value = mock_response
        
        # Start the communicator
        await self.communicator.start()
        
        # Call the API and expect an exception
        with self.assertRaises(Exception):
            await self.communicator.call_component_api(
                target_component='data_ingestion',
                endpoint='/test',
                method='GET',
                retries=1
            )
        
        # Stop the communicator
        await self.communicator.stop()
    
    @patch('aiohttp.ClientSession.request')
    async def test_send_message(self, mock_request):
        """Test sending a message to another component."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = asyncio.coroutine(lambda: {'status': 'received'})
        mock_response.__aenter__.return_value = mock_response
        mock_request.return_value = mock_response
        
        # Start the communicator
        await self.communicator.start()
        
        # Send a message
        result = await self.communicator.send_message(
            target_component='semantic_signal',
            message_type='test_message',
            data={'key': 'value'},
            priority='high'
        )
        
        # Verify the result
        self.assertEqual(result, {'status': 'received'})
        
        # Verify the request was made with the correct parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args[1]
        self.assertEqual(call_args['method'], 'POST')
        self.assertEqual(call_args['url'], 'http://localhost:8002/api/v1/messages')
        
        # Verify the message format
        message = call_args['json']
        self.assertEqual(message['type'], 'test_message')
        self.assertEqual(message['data'], {'key': 'value'})
        self.assertEqual(message['metadata']['source'], 'test_component')
        self.assertEqual(message['metadata']['priority'], 'high')
        
        # Stop the communicator
        await self.communicator.stop()

class TestDataFlowManager(unittest.TestCase):
    """
    Unit tests for the DataFlowManager class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'components': {
                'data_ingestion': {
                    'host': 'localhost',
                    'port': 8001,
                    'api_prefix': '/api/v1'
                },
                'semantic_signal': {
                    'host': 'localhost',
                    'port': 8002,
                    'api_prefix': '/api/v1'
                },
                'strategy_generator': {
                    'host': 'localhost',
                    'port': 8003,
                    'api_prefix': '/api/v1'
                }
            },
            'integration': {
                'data_flow': [
                    {
                        'source': 'data_ingestion',
                        'target': 'semantic_signal',
                        'data_types': ['market_data', 'news']
                    },
                    {
                        'source': 'semantic_signal',
                        'target': 'strategy_generator',
                        'data_types': ['sentiment_signals', 'entity_events']
                    }
                ]
            }
        }
        self.data_flow = DataFlowManager(self.config, 'semantic_signal')
    
    def test_parse_data_flows(self):
        """Test parsing of data flow configuration."""
        flows = self.data_flow._parse_data_flows()
        
        # Verify incoming flows
        self.assertEqual(len(flows['incoming']), 1)
        self.assertEqual(flows['incoming'][0]['source'], 'data_ingestion')
        self.assertEqual(flows['incoming'][0]['data_types'], ['market_data', 'news'])
        
        # Verify outgoing flows
        self.assertEqual(len(flows['outgoing']), 1)
        self.assertEqual(flows['outgoing'][0]['target'], 'strategy_generator')
        self.assertEqual(flows['outgoing'][0]['data_types'], ['sentiment_signals', 'entity_events'])
    
    @patch.object(ComponentCommunicator, 'send_message')
    async def test_send_data(self, mock_send_message):
        """Test sending data to target components."""
        # Mock the send_message method
        mock_send_message.return_value = {'status': 'received'}
        
        # Start the data flow manager
        await self.data_flow.start()
        
        # Send data
        result = await self.data_flow.send_data(
            data_type='sentiment_signals',
            data={'symbol': 'AAPL', 'sentiment': 0.8}
        )
        
        # Verify the result
        self.assertEqual(result['strategy_generator']['success'], True)
        self.assertEqual(result['strategy_generator']['response'], {'status': 'received'})
        
        # Verify the send_message call
        mock_send_message.assert_called_once()
        call_args = mock_send_message.call_args[1]
        self.assertEqual(call_args['target_component'], 'strategy_generator')
        self.assertEqual(call_args['message_type'], 'data_sentiment_signals')
        self.assertEqual(call_args['data']['type'], 'sentiment_signals')
        self.assertEqual(call_args['data']['content'], {'symbol': 'AAPL', 'sentiment': 0.8})
        
        # Stop the data flow manager
        await self.data_flow.stop()
    
    async def test_receive_data_expected_flow(self):
        """Test receiving data from an expected source."""
        # Receive data
        result = await self.data_flow.receive_data(
            source='data_ingestion',
            data_type='market_data',
            data={'symbol': 'AAPL', 'price': 150.0}
        )
        
        # Verify the result
        self.assertEqual(result['success'], True)
    
    async def test_receive_data_unexpected_flow(self):
        """Test receiving data from an unexpected source."""
        # Receive data
        result = await self.data_flow.receive_data(
            source='strategy_generator',  # This is not an expected source for semantic_signal
            data_type='strategy',
            data={'id': 'strategy1', 'type': 'momentum'}
        )
        
        # Verify the result
        self.assertEqual(result['success'], False)
        self.assertIn('Unexpected data flow', result['error'])

class TestComponentBase(unittest.TestCase):
    """
    Unit tests for the ComponentBase class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.config_dir = '/home/ubuntu/trading_pipeline/tests/temp'
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_path = f'{self.config_dir}/test_config.yaml'
        
        self.config = {
            'general': {
                'environment': 'development',
                'log_level': 'INFO',
                'data_dir': '/home/ubuntu/trading_pipeline/data',
            },
            'components': {
                'test_component': {
                    'enabled': True,
                    'host': 'localhost',
                    'port': 9999,
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
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create a test component class
        class TestComponent(ComponentBase):
            async def initialize(self):
                self.initialized = True
            
            async def cleanup(self):
                self.cleaned_up = True
            
            def get_health_details(self):
                return {'test_metric': 100}
            
            async def handle_message(self, message_type, data, metadata):
                if message_type == 'test_message':
                    return {'success': True, 'message': 'Test message handled'}
                return await super().handle_message(message_type, data, metadata)
            
            def get_available_data_types(self):
                return ['test_data']
            
            def get_data_schema(self, data_type):
                if data_type == 'test_data':
                    return {'type': 'object', 'properties': {'key': {'type': 'string'}}}
                return {}
        
        self.component_class = TestComponent
        self.component = TestComponent('test_component', config_path=self.config_path)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary config file
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
    
    def test_load_config(self):
        """Test loading of configuration."""
        config = self.component._load_config()
        
        # Verify the config
        self.assertEqual(config['general']['environment'], 'development')
        self.assertEqual(config['components']['test_component']['port'], 9999)
    
    def test_create_app(self):
        """Test creation of FastAPI application."""
        app = self.component._create_app()
        
        # Verify the app
        self.assertEqual(app.title, 'Test Component API')
        
        # Verify the routes
        route_paths = [route.path for route in app.routes]
        self.assertIn('/health', route_paths)
        self.assertIn('/messages', route_paths)
        self.assertIn('/data/types', route_paths)
        self.assertIn('/data/schema/{data_type}', route_paths)
    
    async def test_start_stop(self):
        """Test starting and stopping the component."""
        # Start the component
        await self.component.start()
        
        # Verify the component is running
        self.assertTrue(self.component.running)
        self.assertTrue(hasattr(self.component, 'initialized'))
        self.assertTrue(self.component.initialized)
        
        # Stop the component
        await self.component.stop()
        
        # Verify the component is stopped
        self.assertFalse(self.component.running)
        self.assertTrue(hasattr(self.component, 'cleaned_up'))
        self.assertTrue(self.component.cleaned_up)
    
    async def test_handle_message(self):
        """Test message handling."""
        # Test a supported message type
        result = await self.component.handle_message(
            message_type='test_message',
            data={'key': 'value'},
            metadata={'source': 'test_source'}
        )
        
        # Verify the result
        self.assertEqual(result['success'], True)
        self.assertEqual(result['message'], 'Test message handled')
        
        # Test an unsupported message type
        result = await self.component.handle_message(
            message_type='unsupported_message',
            data={'key': 'value'},
            metadata={'source': 'test_source'}
        )
        
        # Verify the result
        self.assertEqual(result['success'], False)
        self.assertIn('not supported', result['error'])
    
    def test_get_available_data_types(self):
        """Test getting available data types."""
        data_types = self.component.get_available_data_types()
        
        # Verify the data types
        self.assertEqual(data_types, ['test_data'])
    
    def test_get_data_schema(self):
        """Test getting data schema."""
        # Test a supported data type
        schema = self.component.get_data_schema('test_data')
        
        # Verify the schema
        self.assertEqual(schema['type'], 'object')
        self.assertIn('properties', schema)
        
        # Test an unsupported data type
        schema = self.component.get_data_schema('unsupported_data')
        
        # Verify the schema
        self.assertEqual(schema, {})

# Run the tests
if __name__ == '__main__':
    unittest.main()
