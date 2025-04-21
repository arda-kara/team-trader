"""
Base collector class and utilities for data collection.
"""

import asyncio
import datetime
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..storage.database import get_db_session
from ..storage.models import DataCollectionLog
from ..storage.redis_client import RedisQueue

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self, collector_type: str, source: str, queue: Optional[RedisQueue] = None):
        """Initialize collector.
        
        Args:
            collector_type: Type of collector (market_data, news_data, economic_data)
            source: Data source name
            queue: Optional Redis queue for publishing collected data
        """
        self.collector_type = collector_type
        self.source = source
        self.queue = queue
        self.log_entry_id = None
    
    def _start_collection_log(self) -> int:
        """Start a collection log entry.
        
        Returns:
            int: Log entry ID
        """
        with get_db_session() as session:
            log_entry = DataCollectionLog(
                collector_type=self.collector_type,
                source=self.source,
                start_time=datetime.datetime.utcnow(),
                status="running"
            )
            session.add(log_entry)
            session.flush()
            log_id = log_entry.id
            
        self.log_entry_id = log_id
        return log_id
    
    def _update_collection_log(self, status: str, records_processed: int = 0, error_message: Optional[str] = None):
        """Update collection log entry.
        
        Args:
            status: Collection status
            records_processed: Number of records processed
            error_message: Optional error message
        """
        if not self.log_entry_id:
            return
            
        with get_db_session() as session:
            log_entry = session.query(DataCollectionLog).get(self.log_entry_id)
            if log_entry:
                log_entry.status = status
                log_entry.end_time = datetime.datetime.utcnow()
                log_entry.records_processed = records_processed
                if error_message:
                    log_entry.error_message = error_message
    
    def _publish_data(self, data: Union[Dict, List[Dict]]) -> bool:
        """Publish data to queue.
        
        Args:
            data: Data to publish
            
        Returns:
            bool: Success status
        """
        if not self.queue:
            return False
            
        if isinstance(data, list):
            for item in data:
                self.queue.push(item)
            return True
        else:
            return self.queue.push(data)
    
    @abstractmethod
    async def collect(self, *args, **kwargs) -> Dict[str, Any]:
        """Collect data from source.
        
        Returns:
            Dict[str, Any]: Collection results
        """
        pass
    
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run collection process with logging.
        
        Returns:
            Dict[str, Any]: Collection results
        """
        self._start_collection_log()
        records_processed = 0
        
        try:
            result = await self.collect(*args, **kwargs)
            records_processed = result.get("records_processed", 0)
            self._update_collection_log("completed", records_processed)
            return result
        except Exception as e:
            logger.exception(f"Error in {self.collector_type} collector for {self.source}")
            self._update_collection_log("failed", records_processed, str(e))
            return {
                "success": False,
                "error": str(e),
                "records_processed": records_processed
            }

class CollectorRegistry:
    """Registry for data collectors."""
    
    _collectors = {}
    
    @classmethod
    def register(cls, collector_type: str, source: str, collector_class):
        """Register a collector.
        
        Args:
            collector_type: Type of collector
            source: Data source name
            collector_class: Collector class
        """
        key = f"{collector_type}:{source}"
        cls._collectors[key] = collector_class
    
    @classmethod
    def get_collector(cls, collector_type: str, source: str, *args, **kwargs):
        """Get collector instance.
        
        Args:
            collector_type: Type of collector
            source: Data source name
            
        Returns:
            BaseCollector: Collector instance
        """
        key = f"{collector_type}:{source}"
        collector_class = cls._collectors.get(key)
        
        if not collector_class:
            raise ValueError(f"No collector registered for {key}")
            
        return collector_class(*args, **kwargs)
    
    @classmethod
    def list_collectors(cls) -> List[str]:
        """List all registered collectors.
        
        Returns:
            List[str]: List of collector keys
        """
        return list(cls._collectors.keys())
