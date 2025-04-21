# Data Ingestion Layer

This module is responsible for collecting, normalizing, and distributing data from various financial sources including market data, news, and economic indicators.

## Structure
- `collectors/`: Source-specific data collection modules
- `processors/`: Data normalization and preprocessing
- `storage/`: Database adapters and data persistence
- `api/`: Endpoints for data access
- `models/`: Data models and schemas
- `config/`: Configuration settings

## Dependencies
- FastAPI: Web framework for APIs
- SQLAlchemy: ORM for database operations
- Redis: Message broker and caching
- Pydantic: Data validation
- Requests/aiohttp: HTTP clients
- websockets: WebSocket client
- pandas: Data manipulation
