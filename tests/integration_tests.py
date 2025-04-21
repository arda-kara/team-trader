"""
Test suite for the AI-Augmented Full-Stack Algorithmic Trading Pipeline.
This module provides tests for the integrated pipeline components.
"""

import os
import sys
import logging
import asyncio
import unittest
import pytest
import yaml
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import requests
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/ubuntu/trading_pipeline/logs/tests.log')
    ]
)

logger = logging.getLogger('pipeline_tests')

class PipelineIntegrationTests:
    """
    Tests for the integrated pipeline components.
    
    This class:
    1. Tests the integration between pipeline components
    2. Verifies data flow between components
    3. Tests error handling and recovery
    4. Validates end-to-end functionality
    """
    
    def __init__(self, config_path: str = '/home/ubuntu/trading_pipeline/config/pipeline.yaml'):
        """
        Initialize the pipeline tests.
        
        Args:
            config_path: Path to the pipeline configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.component_urls = self._get_component_urls()
        logger.info("Pipeline tests initialized")
    
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
            raise
    
    def _get_component_urls(self) -> Dict[str, str]:
        """
        Get the base URLs for all components.
        
        Returns:
            Dict[str, str]: Component URLs
        """
        urls = {}
        for component_name, component_config in self.config['components'].items():
            if component_config['enabled']:
                host = component_config['host']
                port = component_config['port']
                api_prefix = component_config['api_prefix']
                urls[component_name] = f"http://{host}:{port}{api_prefix}"
        
        return urls
    
    async def test_component_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Test the health of all components.
        
        Returns:
            Dict[str, Dict[str, Any]]: Health status of all components
        """
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for component_name, base_url in self.component_urls.items():
                try:
                    url = f"{base_url}/health"
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            results[component_name] = {
                                'status': 'passed',
                                'health': health_data
                            }
                            logger.info(f"Component {component_name} health check passed")
                        else:
                            results[component_name] = {
                                'status': 'failed',
                                'error': f"Health check failed with status {response.status}"
                            }
                            logger.error(f"Component {component_name} health check failed with status {response.status}")
                
                except Exception as e:
                    results[component_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    logger.error(f"Error checking health of component {component_name}: {e}")
        
        return results
    
    async def test_data_ingestion(self) -> Dict[str, Any]:
        """
        Test the data ingestion component.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'status': 'unknown',
            'tests': {}
        }
        
        try:
            base_url = self.component_urls.get('data_ingestion')
            if not base_url:
                results['status'] = 'skipped'
                results['error'] = "Data ingestion component not enabled"
                return results
            
            async with aiohttp.ClientSession() as session:
                # Test market data endpoint
                market_data_url = f"{base_url}/market-data"
                params = {'symbols': 'AAPL,MSFT,GOOGL'}
                async with session.get(market_data_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        market_data = await response.json()
                        results['tests']['market_data'] = {
                            'status': 'passed',
                            'data': market_data
                        }
                        logger.info("Market data test passed")
                    else:
                        results['tests']['market_data'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Market data test failed with status {response.status}")
                
                # Test news data endpoint
                news_url = f"{base_url}/news"
                params = {'keywords': 'finance,stocks,market'}
                async with session.get(news_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        news_data = await response.json()
                        results['tests']['news_data'] = {
                            'status': 'passed',
                            'data': news_data
                        }
                        logger.info("News data test passed")
                    else:
                        results['tests']['news_data'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"News data test failed with status {response.status}")
                
                # Test economic data endpoint
                economic_url = f"{base_url}/economic-data"
                params = {'indicators': 'GDP,CPI,UNEMPLOYMENT'}
                async with session.get(economic_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        economic_data = await response.json()
                        results['tests']['economic_data'] = {
                            'status': 'passed',
                            'data': economic_data
                        }
                        logger.info("Economic data test passed")
                    else:
                        results['tests']['economic_data'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Economic data test failed with status {response.status}")
            
            # Determine overall status
            if all(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'passed'
            elif any(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
        
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Error testing data ingestion component: {e}")
        
        return results
    
    async def test_semantic_signal(self) -> Dict[str, Any]:
        """
        Test the semantic signal component.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'status': 'unknown',
            'tests': {}
        }
        
        try:
            base_url = self.component_urls.get('semantic_signal')
            if not base_url:
                results['status'] = 'skipped'
                results['error'] = "Semantic signal component not enabled"
                return results
            
            async with aiohttp.ClientSession() as session:
                # Test sentiment analysis endpoint
                sentiment_url = f"{base_url}/sentiment"
                data = {
                    'text': 'The company reported strong earnings, exceeding analyst expectations. Revenue grew by 20% year-over-year.'
                }
                async with session.post(sentiment_url, json=data, timeout=10) as response:
                    if response.status == 200:
                        sentiment_data = await response.json()
                        results['tests']['sentiment_analysis'] = {
                            'status': 'passed',
                            'data': sentiment_data
                        }
                        logger.info("Sentiment analysis test passed")
                    else:
                        results['tests']['sentiment_analysis'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Sentiment analysis test failed with status {response.status}")
                
                # Test entity extraction endpoint
                entity_url = f"{base_url}/entities"
                data = {
                    'text': 'Apple Inc. announced a new partnership with Microsoft Corporation to enhance cloud services.'
                }
                async with session.post(entity_url, json=data, timeout=10) as response:
                    if response.status == 200:
                        entity_data = await response.json()
                        results['tests']['entity_extraction'] = {
                            'status': 'passed',
                            'data': entity_data
                        }
                        logger.info("Entity extraction test passed")
                    else:
                        results['tests']['entity_extraction'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Entity extraction test failed with status {response.status}")
                
                # Test signal generation endpoint
                signal_url = f"{base_url}/signals"
                data = {
                    'symbol': 'AAPL',
                    'sources': ['news', 'social_media', 'earnings_calls']
                }
                async with session.post(signal_url, json=data, timeout=10) as response:
                    if response.status == 200:
                        signal_data = await response.json()
                        results['tests']['signal_generation'] = {
                            'status': 'passed',
                            'data': signal_data
                        }
                        logger.info("Signal generation test passed")
                    else:
                        results['tests']['signal_generation'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Signal generation test failed with status {response.status}")
            
            # Determine overall status
            if all(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'passed'
            elif any(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
        
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Error testing semantic signal component: {e}")
        
        return results
    
    async def test_strategy_generator(self) -> Dict[str, Any]:
        """
        Test the strategy generator component.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'status': 'unknown',
            'tests': {}
        }
        
        try:
            base_url = self.component_urls.get('strategy_generator')
            if not base_url:
                results['status'] = 'skipped'
                results['error'] = "Strategy generator component not enabled"
                return results
            
            async with aiohttp.ClientSession() as session:
                # Test strategy generation endpoint
                strategy_url = f"{base_url}/strategies/generate"
                data = {
                    'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                    'strategy_type': 'momentum',
                    'timeframe': '1d',
                    'lookback_period': 30
                }
                async with session.post(strategy_url, json=data, timeout=15) as response:
                    if response.status == 200:
                        strategy_data = await response.json()
                        results['tests']['strategy_generation'] = {
                            'status': 'passed',
                            'data': strategy_data
                        }
                        logger.info("Strategy generation test passed")
                    else:
                        results['tests']['strategy_generation'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Strategy generation test failed with status {response.status}")
                
                # Test backtest endpoint
                backtest_url = f"{base_url}/backtest"
                data = {
                    'strategy_id': strategy_data.get('strategy_id', 'test_strategy'),
                    'start_date': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'initial_capital': 100000
                }
                async with session.post(backtest_url, json=data, timeout=20) as response:
                    if response.status == 200:
                        backtest_data = await response.json()
                        results['tests']['backtest'] = {
                            'status': 'passed',
                            'data': backtest_data
                        }
                        logger.info("Backtest test passed")
                    else:
                        results['tests']['backtest'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Backtest test failed with status {response.status}")
                
                # Test optimization endpoint
                optimization_url = f"{base_url}/optimize"
                data = {
                    'strategy_id': strategy_data.get('strategy_id', 'test_strategy'),
                    'optimization_method': 'grid_search',
                    'parameters': {
                        'fast_period': [5, 10, 15],
                        'slow_period': [20, 30, 40],
                        'signal_threshold': [0.5, 0.7, 0.9]
                    }
                }
                async with session.post(optimization_url, json=data, timeout=30) as response:
                    if response.status == 200:
                        optimization_data = await response.json()
                        results['tests']['optimization'] = {
                            'status': 'passed',
                            'data': optimization_data
                        }
                        logger.info("Optimization test passed")
                    else:
                        results['tests']['optimization'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Optimization test failed with status {response.status}")
            
            # Determine overall status
            if all(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'passed'
            elif any(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
        
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Error testing strategy generator component: {e}")
        
        return results
    
    async def test_execution_engine(self) -> Dict[str, Any]:
        """
        Test the execution engine component.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'status': 'unknown',
            'tests': {}
        }
        
        try:
            base_url = self.component_urls.get('execution_engine')
            if not base_url:
                results['status'] = 'skipped'
                results['error'] = "Execution engine component not enabled"
                return results
            
            async with aiohttp.ClientSession() as session:
                # Test order creation endpoint
                order_url = f"{base_url}/orders"
                data = {
                    'symbol': 'AAPL',
                    'side': 'buy',
                    'quantity': 10,
                    'order_type': 'market',
                    'time_in_force': 'day'
                }
                async with session.post(order_url, json=data, timeout=10) as response:
                    if response.status == 200:
                        order_data = await response.json()
                        results['tests']['order_creation'] = {
                            'status': 'passed',
                            'data': order_data
                        }
                        logger.info("Order creation test passed")
                    else:
                        results['tests']['order_creation'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Order creation test failed with status {response.status}")
                
                # Test order status endpoint
                if results['tests'].get('order_creation', {}).get('status') == 'passed':
                    order_id = order_data.get('order_id', 'test_order')
                    status_url = f"{base_url}/orders/{order_id}"
                    async with session.get(status_url, timeout=10) as response:
                        if response.status == 200:
                            status_data = await response.json()
                            results['tests']['order_status'] = {
                                'status': 'passed',
                                'data': status_data
                            }
                            logger.info("Order status test passed")
                        else:
                            results['tests']['order_status'] = {
                                'status': 'failed',
                                'error': f"Request failed with status {response.status}"
                            }
                            logger.error(f"Order status test failed with status {response.status}")
                
                # Test position endpoint
                position_url = f"{base_url}/positions"
                async with session.get(position_url, timeout=10) as response:
                    if response.status == 200:
                        position_data = await response.json()
                        results['tests']['positions'] = {
                            'status': 'passed',
                            'data': position_data
                        }
                        logger.info("Positions test passed")
                    else:
                        results['tests']['positions'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Positions test failed with status {response.status}")
            
            # Determine overall status
            if all(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'passed'
            elif any(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
        
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Error testing execution engine component: {e}")
        
        return results
    
    async def test_risk_management(self) -> Dict[str, Any]:
        """
        Test the risk management component.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'status': 'unknown',
            'tests': {}
        }
        
        try:
            base_url = self.component_urls.get('risk_management')
            if not base_url:
                results['status'] = 'skipped'
                results['error'] = "Risk management component not enabled"
                return results
            
            async with aiohttp.ClientSession() as session:
                # Test risk metrics endpoint
                metrics_url = f"{base_url}/risk-metrics"
                async with session.get(metrics_url, timeout=10) as response:
                    if response.status == 200:
                        metrics_data = await response.json()
                        results['tests']['risk_metrics'] = {
                            'status': 'passed',
                            'data': metrics_data
                        }
                        logger.info("Risk metrics test passed")
                    else:
                        results['tests']['risk_metrics'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Risk metrics test failed with status {response.status}")
                
                # Test portfolio optimization endpoint
                optimization_url = f"{base_url}/portfolio/optimize"
                data = {
                    'optimization_method': 'mean_variance',
                    'risk_aversion': 0.5,
                    'constraints': {
                        'max_position_size': 0.2,
                        'max_sector_exposure': 0.3
                    }
                }
                async with session.post(optimization_url, json=data, timeout=15) as response:
                    if response.status == 200:
                        optimization_data = await response.json()
                        results['tests']['portfolio_optimization'] = {
                            'status': 'passed',
                            'data': optimization_data
                        }
                        logger.info("Portfolio optimization test passed")
                    else:
                        results['tests']['portfolio_optimization'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Portfolio optimization test failed with status {response.status}")
                
                # Test risk check endpoint
                check_url = f"{base_url}/risk-check"
                data = {
                    'order': {
                        'symbol': 'AAPL',
                        'side': 'buy',
                        'quantity': 100,
                        'order_type': 'market'
                    }
                }
                async with session.post(check_url, json=data, timeout=10) as response:
                    if response.status == 200:
                        check_data = await response.json()
                        results['tests']['risk_check'] = {
                            'status': 'passed',
                            'data': check_data
                        }
                        logger.info("Risk check test passed")
                    else:
                        results['tests']['risk_check'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Risk check test failed with status {response.status}")
            
            # Determine overall status
            if all(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'passed'
            elif any(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
        
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Error testing risk management component: {e}")
        
        return results
    
    async def test_agentic_oversight(self) -> Dict[str, Any]:
        """
        Test the agentic oversight component.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'status': 'unknown',
            'tests': {}
        }
        
        try:
            base_url = self.component_urls.get('agentic_oversight')
            if not base_url:
                results['status'] = 'skipped'
                results['error'] = "Agentic oversight component not enabled"
                return results
            
            async with aiohttp.ClientSession() as session:
                # Test agent status endpoint
                status_url = f"{base_url}/agents/status"
                async with session.get(status_url, timeout=10) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        results['tests']['agent_status'] = {
                            'status': 'passed',
                            'data': status_data
                        }
                        logger.info("Agent status test passed")
                    else:
                        results['tests']['agent_status'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Agent status test failed with status {response.status}")
                
                # Test decision request endpoint
                decision_url = f"{base_url}/decisions"
                data = {
                    'context': {
                        'market_conditions': 'volatile',
                        'portfolio_status': 'healthy',
                        'risk_metrics': {
                            'var': 0.02,
                            'sharpe': 1.5
                        }
                    },
                    'options': [
                        {'id': 'option1', 'description': 'Reduce exposure to volatile sectors'},
                        {'id': 'option2', 'description': 'Maintain current positions'},
                        {'id': 'option3', 'description': 'Increase hedging'}
                    ]
                }
                async with session.post(decision_url, json=data, timeout=15) as response:
                    if response.status == 200:
                        decision_data = await response.json()
                        results['tests']['decision_making'] = {
                            'status': 'passed',
                            'data': decision_data
                        }
                        logger.info("Decision making test passed")
                    else:
                        results['tests']['decision_making'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Decision making test failed with status {response.status}")
                
                # Test explanation endpoint
                explanation_url = f"{base_url}/explanations"
                data = {
                    'action': 'sell',
                    'symbol': 'AAPL',
                    'quantity': 50,
                    'reasoning': 'risk_management',
                    'context': {
                        'portfolio_concentration': 'high',
                        'market_volatility': 'increasing'
                    }
                }
                async with session.post(explanation_url, json=data, timeout=15) as response:
                    if response.status == 200:
                        explanation_data = await response.json()
                        results['tests']['explanation'] = {
                            'status': 'passed',
                            'data': explanation_data
                        }
                        logger.info("Explanation test passed")
                    else:
                        results['tests']['explanation'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Explanation test failed with status {response.status}")
            
            # Determine overall status
            if all(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'passed'
            elif any(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
        
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Error testing agentic oversight component: {e}")
        
        return results
    
    async def test_dashboard(self) -> Dict[str, Any]:
        """
        Test the dashboard component.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'status': 'unknown',
            'tests': {}
        }
        
        try:
            base_url = self.component_urls.get('dashboard')
            if not base_url:
                results['status'] = 'skipped'
                results['error'] = "Dashboard component not enabled"
                return results
            
            async with aiohttp.ClientSession() as session:
                # Test API health endpoint
                health_url = f"{base_url}/health"
                async with session.get(health_url, timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        results['tests']['api_health'] = {
                            'status': 'passed',
                            'data': health_data
                        }
                        logger.info("Dashboard API health test passed")
                    else:
                        results['tests']['api_health'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Dashboard API health test failed with status {response.status}")
                
                # Test system status endpoint
                status_url = f"{base_url}/system/status"
                headers = {'Authorization': 'Bearer test_token'}  # This would be a valid token in a real test
                async with session.get(status_url, headers=headers, timeout=10) as response:
                    if response.status == 200 or response.status == 401:  # 401 is acceptable since we're using a fake token
                        results['tests']['system_status'] = {
                            'status': 'passed',
                            'response_status': response.status
                        }
                        logger.info("System status test passed")
                    else:
                        results['tests']['system_status'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"System status test failed with status {response.status}")
                
                # Test frontend access
                frontend_url = f"http://{self.config['components']['dashboard']['host']}:{self.config['components']['dashboard']['port']}"
                async with session.get(frontend_url, timeout=10) as response:
                    if response.status == 200:
                        results['tests']['frontend_access'] = {
                            'status': 'passed',
                            'response_status': response.status
                        }
                        logger.info("Frontend access test passed")
                    else:
                        results['tests']['frontend_access'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Frontend access test failed with status {response.status}")
            
            # Determine overall status
            if all(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'passed'
            elif any(test['status'] == 'passed' for test in results['tests'].values()):
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
        
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Error testing dashboard component: {e}")
        
        return results
    
    async def test_end_to_end_flow(self) -> Dict[str, Any]:
        """
        Test the end-to-end data flow through the pipeline.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'status': 'unknown',
            'tests': {}
        }
        
        try:
            # Step 1: Get market data from data ingestion
            data_ingestion_url = self.component_urls.get('data_ingestion')
            if not data_ingestion_url:
                results['status'] = 'skipped'
                results['error'] = "Data ingestion component not enabled"
                return results
            
            async with aiohttp.ClientSession() as session:
                # Get market data
                market_data_url = f"{data_ingestion_url}/market-data"
                params = {'symbols': 'AAPL,MSFT,GOOGL'}
                async with session.get(market_data_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        market_data = await response.json()
                        results['tests']['step1_market_data'] = {
                            'status': 'passed',
                            'data': market_data
                        }
                        logger.info("Step 1: Market data retrieval passed")
                    else:
                        results['tests']['step1_market_data'] = {
                            'status': 'failed',
                            'error': f"Request failed with status {response.status}"
                        }
                        logger.error(f"Step 1: Market data retrieval failed with status {response.status}")
                        return results  # Stop the test if this step fails
                
                # Step 2: Generate semantic signals
                semantic_signal_url = self.component_urls.get('semantic_signal')
                if not semantic_signal_url:
                    results['tests']['step2_semantic_signal'] = {
                        'status': 'skipped',
                        'error': "Semantic signal component not enabled"
                    }
                else:
                    signal_url = f"{semantic_signal_url}/signals"
                    data = {
                        'symbol': 'AAPL',
                        'sources': ['news', 'social_media', 'earnings_calls']
                    }
                    async with session.post(signal_url, json=data, timeout=15) as response:
                        if response.status == 200:
                            signal_data = await response.json()
                            results['tests']['step2_semantic_signal'] = {
                                'status': 'passed',
                                'data': signal_data
                            }
                            logger.info("Step 2: Semantic signal generation passed")
                        else:
                            results['tests']['step2_semantic_signal'] = {
                                'status': 'failed',
                                'error': f"Request failed with status {response.status}"
                            }
                            logger.error(f"Step 2: Semantic signal generation failed with status {response.status}")
                
                # Step 3: Generate trading strategy
                strategy_generator_url = self.component_urls.get('strategy_generator')
                if not strategy_generator_url:
                    results['tests']['step3_strategy_generation'] = {
                        'status': 'skipped',
                        'error': "Strategy generator component not enabled"
                    }
                else:
                    strategy_url = f"{strategy_generator_url}/strategies/generate"
                    data = {
                        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                        'strategy_type': 'momentum',
                        'timeframe': '1d',
                        'lookback_period': 30,
                        'signals': results['tests'].get('step2_semantic_signal', {}).get('data', {})
                    }
                    async with session.post(strategy_url, json=data, timeout=15) as response:
                        if response.status == 200:
                            strategy_data = await response.json()
                            results['tests']['step3_strategy_generation'] = {
                                'status': 'passed',
                                'data': strategy_data
                            }
                            logger.info("Step 3: Strategy generation passed")
                        else:
                            results['tests']['step3_strategy_generation'] = {
                                'status': 'failed',
                                'error': f"Request failed with status {response.status}"
                            }
                            logger.error(f"Step 3: Strategy generation failed with status {response.status}")
                
                # Step 4: Risk check
                risk_management_url = self.component_urls.get('risk_management')
                if not risk_management_url:
                    results['tests']['step4_risk_check'] = {
                        'status': 'skipped',
                        'error': "Risk management component not enabled"
                    }
                else:
                    risk_check_url = f"{risk_management_url}/risk-check"
                    strategy = results['tests'].get('step3_strategy_generation', {}).get('data', {})
                    data = {
                        'strategy': strategy,
                        'portfolio': {
                            'positions': [
                                {'symbol': 'AAPL', 'quantity': 100, 'avg_price': 150.0},
                                {'symbol': 'MSFT', 'quantity': 50, 'avg_price': 250.0}
                            ],
                            'cash': 50000
                        }
                    }
                    async with session.post(risk_check_url, json=data, timeout=15) as response:
                        if response.status == 200:
                            risk_check_data = await response.json()
                            results['tests']['step4_risk_check'] = {
                                'status': 'passed',
                                'data': risk_check_data
                            }
                            logger.info("Step 4: Risk check passed")
                        else:
                            results['tests']['step4_risk_check'] = {
                                'status': 'failed',
                                'error': f"Request failed with status {response.status}"
                            }
                            logger.error(f"Step 4: Risk check failed with status {response.status}")
                
                # Step 5: Execute order
                execution_engine_url = self.component_urls.get('execution_engine')
                if not execution_engine_url:
                    results['tests']['step5_order_execution'] = {
                        'status': 'skipped',
                        'error': "Execution engine component not enabled"
                    }
                else:
                    order_url = f"{execution_engine_url}/orders"
                    risk_approved_strategy = results['tests'].get('step4_risk_check', {}).get('data', {}).get('approved_strategy', {})
                    data = {
                        'symbol': 'AAPL',
                        'side': 'buy',
                        'quantity': 10,
                        'order_type': 'market',
                        'time_in_force': 'day',
                        'strategy_id': risk_approved_strategy.get('strategy_id', 'test_strategy')
                    }
                    async with session.post(order_url, json=data, timeout=15) as response:
                        if response.status == 200:
                            order_data = await response.json()
                            results['tests']['step5_order_execution'] = {
                                'status': 'passed',
                                'data': order_data
                            }
                            logger.info("Step 5: Order execution passed")
                        else:
                            results['tests']['step5_order_execution'] = {
                                'status': 'failed',
                                'error': f"Request failed with status {response.status}"
                            }
                            logger.error(f"Step 5: Order execution failed with status {response.status}")
                
                # Step 6: Agent oversight
                agentic_oversight_url = self.component_urls.get('agentic_oversight')
                if not agentic_oversight_url:
                    results['tests']['step6_agent_oversight'] = {
                        'status': 'skipped',
                        'error': "Agentic oversight component not enabled"
                    }
                else:
                    explanation_url = f"{agentic_oversight_url}/explanations"
                    order = results['tests'].get('step5_order_execution', {}).get('data', {})
                    data = {
                        'action': 'buy',
                        'symbol': 'AAPL',
                        'quantity': order.get('quantity', 10),
                        'reasoning': 'strategy_signal',
                        'context': {
                            'strategy': results['tests'].get('step3_strategy_generation', {}).get('data', {}),
                            'risk_check': results['tests'].get('step4_risk_check', {}).get('data', {}),
                            'order': order
                        }
                    }
                    async with session.post(explanation_url, json=data, timeout=15) as response:
                        if response.status == 200:
                            explanation_data = await response.json()
                            results['tests']['step6_agent_oversight'] = {
                                'status': 'passed',
                                'data': explanation_data
                            }
                            logger.info("Step 6: Agent oversight passed")
                        else:
                            results['tests']['step6_agent_oversight'] = {
                                'status': 'failed',
                                'error': f"Request failed with status {response.status}"
                            }
                            logger.error(f"Step 6: Agent oversight failed with status {response.status}")
            
            # Determine overall status
            passed_tests = [test for test in results['tests'].values() if test['status'] == 'passed']
            skipped_tests = [test for test in results['tests'].values() if test['status'] == 'skipped']
            
            if len(passed_tests) + len(skipped_tests) == len(results['tests']):
                results['status'] = 'passed'
            elif len(passed_tests) > 0:
                results['status'] = 'partial'
            else:
                results['status'] = 'failed'
        
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Error testing end-to-end flow: {e}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests.
        
        Returns:
            Dict[str, Any]: Test results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test component health
        results['tests']['component_health'] = await self.test_component_health()
        
        # Test individual components
        results['tests']['data_ingestion'] = await self.test_data_ingestion()
        results['tests']['semantic_signal'] = await self.test_semantic_signal()
        results['tests']['strategy_generator'] = await self.test_strategy_generator()
        results['tests']['execution_engine'] = await self.test_execution_engine()
        results['tests']['risk_management'] = await self.test_risk_management()
        results['tests']['agentic_oversight'] = await self.test_agentic_oversight()
        results['tests']['dashboard'] = await self.test_dashboard()
        
        # Test end-to-end flow
        results['tests']['end_to_end_flow'] = await self.test_end_to_end_flow()
        
        # Determine overall status
        passed_tests = [test for test in results['tests'].values() if test.get('status') == 'passed']
        partial_tests = [test for test in results['tests'].values() if test.get('status') == 'partial']
        failed_tests = [test for test in results['tests'].values() if test.get('status') == 'failed']
        skipped_tests = [test for test in results['tests'].values() if test.get('status') == 'skipped']
        
        results['summary'] = {
            'total_tests': len(results['tests']),
            'passed': len(passed_tests),
            'partial': len(partial_tests),
            'failed': len(failed_tests),
            'skipped': len(skipped_tests)
        }
        
        if len(failed_tests) == 0 and len(partial_tests) == 0:
            results['status'] = 'passed'
        elif len(failed_tests) == 0:
            results['status'] = 'partial'
        else:
            results['status'] = 'failed'
        
        return results

# Main entry point
async def main():
    """
    Main entry point for the pipeline tests.
    """
    # Create the pipeline tests
    tests = PipelineIntegrationTests()
    
    # Run all tests
    results = await tests.run_all_tests()
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Save results to file
    os.makedirs('/home/ubuntu/trading_pipeline/logs', exist_ok=True)
    with open('/home/ubuntu/trading_pipeline/logs/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Return exit code based on test results
    if results['status'] == 'passed':
        return 0
    elif results['status'] == 'partial':
        return 1
    else:
        return 2

if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
