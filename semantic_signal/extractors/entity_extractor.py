"""
Entity extraction processor for identifying financial entities in text.
"""

import logging
import re
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy
from spacy.tokens import Doc, Span

from ..config.settings import settings
from ..models.base import Entity, EntityType
from ..processors.redis_client import entity_cache

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Entity extractor for financial texts."""
    
    def __init__(self):
        """Initialize entity extractor."""
        self.spacy_model = settings.nlp.spacy_model
        self.entity_types = settings.entity.entity_types
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(self.spacy_model)
            logger.info(f"Loaded spaCy model: {self.spacy_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            # Fallback to smaller model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded fallback spaCy model: en_core_web_sm")
        
        # Load company aliases and ticker mappings
        self.company_aliases = self._load_company_aliases()
        self.ticker_mapping = self._load_ticker_mapping()
        
        # Add custom pipeline components
        self._add_custom_components()
    
    def _load_company_aliases(self) -> Dict[str, str]:
        """Load company aliases from file.
        
        Returns:
            Dict[str, str]: Mapping of aliases to canonical names
        """
        aliases_file = settings.entity.company_aliases_file
        aliases_path = os.path.join(os.path.dirname(__file__), "..", "data", aliases_file)
        
        try:
            if os.path.exists(aliases_path):
                with open(aliases_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Company aliases file not found: {aliases_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading company aliases: {e}")
            return {}
    
    def _load_ticker_mapping(self) -> Dict[str, Dict[str, str]]:
        """Load ticker mappings from file.
        
        Returns:
            Dict[str, Dict[str, str]]: Mapping of company names to ticker info
        """
        mapping_file = settings.entity.ticker_mapping_file
        mapping_path = os.path.join(os.path.dirname(__file__), "..", "data", mapping_file)
        
        try:
            if os.path.exists(mapping_path):
                with open(mapping_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Ticker mapping file not found: {mapping_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading ticker mapping: {e}")
            return {}
    
    def _add_custom_components(self):
        """Add custom pipeline components to spaCy."""
        # Add financial entity component
        if "financial_entity" not in self.nlp.pipe_names:
            self.nlp.add_pipe("financial_entity", last=True)
    
    def _normalize_entity(self, entity_text: str, entity_type: str) -> Tuple[str, Dict[str, Any]]:
        """Normalize entity text and extract metadata.
        
        Args:
            entity_text: Raw entity text
            entity_type: Entity type
            
        Returns:
            Tuple[str, Dict[str, Any]]: Normalized text and metadata
        """
        normalized_text = entity_text
        metadata = {}
        
        if entity_type == "COMPANY":
            # Check for aliases
            if entity_text.lower() in self.company_aliases:
                normalized_text = self.company_aliases[entity_text.lower()]
            
            # Check for ticker mapping
            if normalized_text in self.ticker_mapping:
                ticker_info = self.ticker_mapping[normalized_text]
                metadata.update(ticker_info)
            
        elif entity_type == "MONEY":
            # Extract numeric value and currency
            money_pattern = r'(?:[$€£¥])?\s*(\d+(?:[.,]\d+)?)\s*(?:million|billion|trillion|M|B|T)?'
            match = re.search(money_pattern, entity_text)
            
            if match:
                value = match.group(1)
                
                # Determine multiplier
                multiplier = 1
                if "million" in entity_text or "M" in entity_text:
                    multiplier = 1_000_000
                elif "billion" in entity_text or "B" in entity_text:
                    multiplier = 1_000_000_000
                elif "trillion" in entity_text or "T" in entity_text:
                    multiplier = 1_000_000_000_000
                
                # Determine currency
                currency = "USD"  # Default
                if "€" in entity_text:
                    currency = "EUR"
                elif "£" in entity_text:
                    currency = "GBP"
                elif "¥" in entity_text:
                    currency = "JPY"
                
                # Clean value and convert to number
                clean_value = value.replace(",", "")
                try:
                    numeric_value = float(clean_value) * multiplier
                    normalized_text = str(numeric_value)
                    metadata = {"currency": currency, "value": numeric_value}
                except ValueError:
                    pass
        
        elif entity_type == "PERCENT":
            # Extract numeric value
            percent_pattern = r'(\d+(?:[.,]\d+)?)\s*%'
            match = re.search(percent_pattern, entity_text)
            
            if match:
                value = match.group(1)
                
                # Clean value and convert to number
                clean_value = value.replace(",", "")
                try:
                    numeric_value = float(clean_value)
                    normalized_text = str(numeric_value)
                    metadata = {"value": numeric_value}
                except ValueError:
                    pass
        
        return normalized_text, metadata
    
    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[Entity]:
        """Extract entities from text.
        
        Args:
            text: Input text
            entity_types: Optional list of entity types to extract
            
        Returns:
            List[Entity]: Extracted entities
        """
        # Check cache
        cache_key = f"entity:{hash(text)}"
        cached_result = entity_cache.get(cache_key)
        
        if cached_result:
            # Convert cached result back to Entity objects
            return [Entity(**entity_dict) for entity_dict in cached_result]
        
        # Filter entity types
        if not entity_types:
            entity_types = self.entity_types
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        seen_spans = set()  # Track spans to avoid duplicates
        
        # Process named entities
        for ent in doc.ents:
            # Map spaCy entity types to our types
            entity_type = self._map_spacy_entity_type(ent.label_)
            
            if entity_type in entity_types:
                span_key = (ent.start_char, ent.end_char)
                
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    
                    # Normalize entity and get metadata
                    normalized_text, metadata = self._normalize_entity(ent.text, entity_type)
                    
                    entity = Entity(
                        text=ent.text,
                        type=entity_type,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        normalized_text=normalized_text,
                        metadata=metadata
                    )
                    
                    entities.append(entity)
        
        # Process financial entities (from custom component)
        if hasattr(doc._, "financial_entities"):
            for financial_ent in doc._.financial_entities:
                entity_type = financial_ent["type"]
                
                if entity_type in entity_types:
                    start_char = financial_ent["start_char"]
                    end_char = financial_ent["end_char"]
                    span_key = (start_char, end_char)
                    
                    if span_key not in seen_spans:
                        seen_spans.add(span_key)
                        
                        # Normalize entity and get metadata
                        normalized_text, metadata = self._normalize_entity(
                            financial_ent["text"], entity_type
                        )
                        
                        entity = Entity(
                            text=financial_ent["text"],
                            type=entity_type,
                            start_char=start_char,
                            end_char=end_char,
                            normalized_text=normalized_text,
                            metadata=metadata
                        )
                        
                        entities.append(entity)
        
        # Cache result
        entity_cache.set(
            cache_key, 
            [entity.dict() for entity in entities],
            ttl=3600  # 1 hour
        )
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_type: str) -> str:
        """Map spaCy entity type to our entity type.
        
        Args:
            spacy_type: spaCy entity type
            
        Returns:
            str: Mapped entity type
        """
        mapping = {
            "ORG": "COMPANY",
            "PERSON": "PERSON",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "PRODUCT": "PRODUCT",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT"
        }
        
        return mapping.get(spacy_type, "UNKNOWN")

# Register custom spaCy component
@spacy.Language.component("financial_entity")
def financial_entity_component(doc: Doc) -> Doc:
    """Custom spaCy component for financial entity extraction."""
    financial_entities = []
    
    # Extract ticker symbols ($AAPL, $MSFT, etc.)
    ticker_pattern = r'\$([A-Z]{1,5})'
    for match in re.finditer(ticker_pattern, doc.text):
        ticker = match.group(1)
        start_char = match.start()
        end_char = match.end()
        
        financial_entities.append({
            "text": f"${ticker}",
            "type": "COMPANY",
            "start_char": start_char,
            "end_char": end_char,
            "metadata": {"ticker": ticker}
        })
    
    # Extract financial metrics
    metric_patterns = [
        (r'revenue', "FINANCIAL_METRIC"),
        (r'earnings', "FINANCIAL_METRIC"),
        (r'profit', "FINANCIAL_METRIC"),
        (r'loss', "FINANCIAL_METRIC"),
        (r'EBITDA', "FINANCIAL_METRIC"),
        (r'EPS', "FINANCIAL_METRIC"),
        (r'P/E', "FINANCIAL_METRIC"),
        (r'market cap', "FINANCIAL_METRIC")
    ]
    
    for pattern, entity_type in metric_patterns:
        for match in re.finditer(pattern, doc.text, re.IGNORECASE):
            start_char = match.start()
            end_char = match.end()
            
            financial_entities.append({
                "text": match.group(),
                "type": entity_type,
                "start_char": start_char,
                "end_char": end_char,
                "metadata": {}
            })
    
    # Store financial entities on Doc
    doc._.financial_entities = financial_entities
    
    return doc

# Set extension on Doc
if not Doc.has_extension("financial_entities"):
    Doc.set_extension("financial_entities", default=[])

# Create extractor instance
entity_extractor = EntityExtractor()
