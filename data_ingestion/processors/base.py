"""
Data processor base class for normalizing and preprocessing data.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..storage.redis_client import RedisQueue

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """Base class for all data processors."""
    
    def __init__(self, processor_type: str, input_queue: Optional[RedisQueue] = None, 
                output_queue: Optional[RedisQueue] = None):
        """Initialize processor.
        
        Args:
            processor_type: Type of processor
            input_queue: Optional input Redis queue
            output_queue: Optional output Redis queue
        """
        self.processor_type = processor_type
        self.input_queue = input_queue
        self.output_queue = output_queue
    
    def _publish_data(self, data: Union[Dict, List[Dict]]) -> bool:
        """Publish data to output queue.
        
        Args:
            data: Data to publish
            
        Returns:
            bool: Success status
        """
        if not self.output_queue:
            return False
            
        if isinstance(data, list):
            for item in data:
                self.output_queue.push(item)
            return True
        else:
            return self.output_queue.push(data)
    
    @abstractmethod
    async def process(self, data: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Process data.
        
        Args:
            data: Input data
            
        Returns:
            Dict[str, Any]: Processing results
        """
        pass
    
    async def process_queue(self, batch_size: int = 100, timeout: int = 1) -> Dict[str, Any]:
        """Process data from input queue.
        
        Args:
            batch_size: Maximum number of items to process
            timeout: Timeout in seconds for blocking pop
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if not self.input_queue:
            return {
                "success": False,
                "error": "No input queue configured",
                "records_processed": 0
            }
            
        items = []
        for _ in range(batch_size):
            item = self.input_queue.pop_blocking(timeout)
            if not item:
                break
                
            try:
                import json
                items.append(json.loads(item))
            except Exception as e:
                logger.error(f"Error parsing queue item: {e}")
        
        if not items:
            return {
                "success": True,
                "records_processed": 0,
                "message": "No items in queue"
            }
            
        try:
            result = await self.process(items)
            return result
        except Exception as e:
            logger.exception(f"Error in {self.processor_type} processor")
            return {
                "success": False,
                "error": str(e),
                "records_processed": 0
            }

class ProcessorRegistry:
    """Registry for data processors."""
    
    _processors = {}
    
    @classmethod
    def register(cls, processor_type: str, processor_class):
        """Register a processor.
        
        Args:
            processor_type: Type of processor
            processor_class: Processor class
        """
        cls._processors[processor_type] = processor_class
    
    @classmethod
    def get_processor(cls, processor_type: str, *args, **kwargs):
        """Get processor instance.
        
        Args:
            processor_type: Type of processor
            
        Returns:
            BaseProcessor: Processor instance
        """
        processor_class = cls._processors.get(processor_type)
        
        if not processor_class:
            raise ValueError(f"No processor registered for {processor_type}")
            
        return processor_class(*args, **kwargs)
    
    @classmethod
    def list_processors(cls) -> List[str]:
        """List all registered processors.
        
        Returns:
            List[str]: List of processor types
        """
        return list(cls._processors.keys())
