# Semantic Signal Generator

This module is responsible for processing textual data to extract actionable trading signals using natural language processing and large language models.

## Structure
- `processors/`: NLP pipeline components for text processing
- `models/`: Data models and schemas
- `extractors/`: Entity and event extraction components
- `sentiment/`: Sentiment analysis components
- `inference/`: Causal inference engine
- `llm/`: LLM integration components
- `api/`: API endpoints for signal access
- `config/`: Configuration settings

## Dependencies
- Transformers: Hugging Face models for NLP
- LangChain: Framework for LLM applications
- spaCy: NLP library for entity recognition
- FinBERT: Financial sentiment analysis
- Vector database: For embedding storage
- FastAPI: Web framework for APIs
- Redis: Message broker and caching
