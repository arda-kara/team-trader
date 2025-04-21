"""
Main entry point for the data ingestion layer.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.config.settings import settings
from data_ingestion.collectors.base import CollectorRegistry
from data_ingestion.processors.base import ProcessorRegistry
from data_ingestion.storage.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

async def run_collectors():
    """Run all registered collectors."""
    logger.info("Starting data collection...")
    
    # Market data collectors
    market_collectors = [
        CollectorRegistry.get_collector("market_data", "alpaca"),
        CollectorRegistry.get_collector("market_data", "yahoo")
    ]
    
    # News data collectors
    news_collectors = [
        CollectorRegistry.get_collector("news_data", "newsapi")
    ]
    
    # Economic data collectors
    economic_collectors = [
        CollectorRegistry.get_collector("economic_data", "fred")
    ]
    
    # Run market data collectors
    market_tasks = [collector.run() for collector in market_collectors]
    market_results = await asyncio.gather(*market_tasks, return_exceptions=True)
    
    for i, result in enumerate(market_results):
        if isinstance(result, Exception):
            logger.error(f"Market collector {i} failed: {result}")
        else:
            logger.info(f"Market collector {i} processed {result.get('records_processed', 0)} records")
    
    # Run news data collectors
    news_tasks = [collector.run() for collector in news_collectors]
    news_results = await asyncio.gather(*news_tasks, return_exceptions=True)
    
    for i, result in enumerate(news_results):
        if isinstance(result, Exception):
            logger.error(f"News collector {i} failed: {result}")
        else:
            logger.info(f"News collector {i} processed {result.get('records_processed', 0)} records")
    
    # Run economic data collectors
    economic_tasks = [collector.run() for collector in economic_collectors]
    economic_results = await asyncio.gather(*economic_tasks, return_exceptions=True)
    
    for i, result in enumerate(economic_results):
        if isinstance(result, Exception):
            logger.error(f"Economic collector {i} failed: {result}")
        else:
            logger.info(f"Economic collector {i} processed {result.get('records_processed', 0)} records")
    
    logger.info("Data collection completed")

async def run_processors():
    """Run all registered processors."""
    logger.info("Starting data processing...")
    
    # Get processors
    market_processor = ProcessorRegistry.get_processor("market_data")
    news_processor = ProcessorRegistry.get_processor("news_data")
    economic_processor = ProcessorRegistry.get_processor("economic_data")
    
    # Process market data
    market_result = await market_processor.process_queue()
    logger.info(f"Market processor processed {market_result.get('records_processed', 0)} records")
    
    # Process news data
    news_result = await news_processor.process_queue()
    logger.info(f"News processor processed {news_result.get('records_processed', 0)} records")
    
    # Process economic data
    economic_result = await economic_processor.process_queue()
    logger.info(f"Economic processor processed {economic_result.get('records_processed', 0)} records")
    
    logger.info("Data processing completed")

async def main():
    """Main entry point."""
    logger.info("Initializing data ingestion layer...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Run collectors and processors
    await run_collectors()
    await run_processors()
    
    logger.info("Data ingestion completed")

if __name__ == "__main__":
    asyncio.run(main())
