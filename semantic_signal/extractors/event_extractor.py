"""
Event extraction processor for identifying financial events in text.
"""

import logging
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..config.settings import settings
from ..models.base import Entity, Event, EventType
from ..extractors.entity_extractor import entity_extractor
from ..llm.client import llm_client
from ..processors.redis_client import event_cache

logger = logging.getLogger(__name__)

class EventExtractor:
    """Event extractor for financial texts."""
    
    def __init__(self):
        """Initialize event extractor."""
        self.event_types = settings.event.event_types
        self.event_templates = self._load_event_templates()
    
    def _load_event_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load event templates from file.
        
        Returns:
            Dict[str, Dict[str, Any]]: Event templates by type
        """
        templates_file = settings.event.event_templates_file
        templates_path = os.path.join(os.path.dirname(__file__), "..", "data", templates_file)
        
        try:
            if os.path.exists(templates_path):
                with open(templates_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Event templates file not found: {templates_path}")
                # Return default templates
                return self._default_templates()
        except Exception as e:
            logger.error(f"Error loading event templates: {e}")
            return self._default_templates()
    
    def _default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create default event templates.
        
        Returns:
            Dict[str, Dict[str, Any]]: Default event templates
        """
        return {
            "MERGER_ACQUISITION": {
                "patterns": [
                    "acquire", "acquisition", "merge", "merger", "takeover", "buy out",
                    "purchased", "acquiring", "merging with"
                ],
                "required_entities": ["COMPANY"],
                "prompt": """
                Identify if this text describes a merger or acquisition event.
                
                Text: {text}
                
                If this is a merger or acquisition, extract the following information:
                - Acquirer: The company making the acquisition
                - Target: The company being acquired
                - Amount: The acquisition amount (if mentioned)
                - Status: Announced, Completed, Rumored, etc.
                
                Format your response as JSON:
                {{
                    "is_merger_acquisition": true/false,
                    "acquirer": "Company name",
                    "target": "Company name",
                    "amount": "Amount with currency",
                    "status": "Status of the deal"
                }}
                
                If this is not a merger or acquisition, return {{"is_merger_acquisition": false}}
                """
            },
            "EARNINGS_REPORT": {
                "patterns": [
                    "earnings", "quarterly results", "financial results", "reported earnings",
                    "profit", "revenue", "EPS", "beat expectations", "missed expectations"
                ],
                "required_entities": ["COMPANY", "FINANCIAL_METRIC"],
                "prompt": """
                Identify if this text describes an earnings report event.
                
                Text: {text}
                
                If this is an earnings report, extract the following information:
                - Company: The company reporting earnings
                - Period: The time period of the report (Q1, Q2, etc.)
                - Revenue: Revenue figure (if mentioned)
                - EPS: Earnings per share (if mentioned)
                - Performance: Whether they beat, met, or missed expectations
                
                Format your response as JSON:
                {{
                    "is_earnings_report": true/false,
                    "company": "Company name",
                    "period": "Reporting period",
                    "revenue": "Revenue figure",
                    "eps": "EPS figure",
                    "performance": "beat/met/missed"
                }}
                
                If this is not an earnings report, return {{"is_earnings_report": false}}
                """
            },
            "PRODUCT_LAUNCH": {
                "patterns": [
                    "launch", "announce", "unveil", "introduce", "debut", "release",
                    "new product", "new service", "new feature"
                ],
                "required_entities": ["COMPANY", "PRODUCT"],
                "prompt": """
                Identify if this text describes a product launch event.
                
                Text: {text}
                
                If this is a product launch, extract the following information:
                - Company: The company launching the product
                - Product: The product being launched
                - Date: The launch date (if mentioned)
                - Features: Key features of the product (if mentioned)
                
                Format your response as JSON:
                {{
                    "is_product_launch": true/false,
                    "company": "Company name",
                    "product": "Product name",
                    "date": "Launch date",
                    "features": ["feature1", "feature2", ...]
                }}
                
                If this is not a product launch, return {{"is_product_launch": false}}
                """
            },
            "LEADERSHIP_CHANGE": {
                "patterns": [
                    "CEO", "executive", "resign", "appoint", "name", "promote",
                    "step down", "successor", "leadership", "board"
                ],
                "required_entities": ["COMPANY", "PERSON"],
                "prompt": """
                Identify if this text describes a leadership change event.
                
                Text: {text}
                
                If this is a leadership change, extract the following information:
                - Company: The company experiencing the leadership change
                - Person: The person involved
                - Role: The role or position
                - Change: The type of change (appointed, resigned, etc.)
                
                Format your response as JSON:
                {{
                    "is_leadership_change": true/false,
                    "company": "Company name",
                    "person": "Person name",
                    "role": "Role or position",
                    "change": "appointed/resigned/promoted/etc."
                }}
                
                If this is not a leadership change, return {{"is_leadership_change": false}}
                """
            },
            "REGULATORY_CHANGE": {
                "patterns": [
                    "regulation", "regulatory", "approve", "approval", "reject", "investigation",
                    "SEC", "FTC", "FDA", "compliance", "lawsuit", "legal"
                ],
                "required_entities": ["COMPANY", "LOCATION"],
                "prompt": """
                Identify if this text describes a regulatory change or action.
                
                Text: {text}
                
                If this is a regulatory change, extract the following information:
                - Entity: The regulatory body or government entity
                - Target: The company or industry affected
                - Action: The regulatory action taken
                - Impact: The potential impact (if mentioned)
                
                Format your response as JSON:
                {{
                    "is_regulatory_change": true/false,
                    "entity": "Regulatory body",
                    "target": "Company or industry",
                    "action": "Action taken",
                    "impact": "Potential impact"
                }}
                
                If this is not a regulatory change, return {{"is_regulatory_change": false}}
                """
            }
        }
    
    def _check_pattern_match(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns.
        
        Args:
            text: Input text
            patterns: List of pattern strings
            
        Returns:
            bool: True if any pattern matches
        """
        text_lower = text.lower()
        for pattern in patterns:
            if pattern.lower() in text_lower:
                return True
        return False
    
    def _check_required_entities(self, entities: List[Entity], required_types: List[str]) -> bool:
        """Check if required entity types are present.
        
        Args:
            entities: List of extracted entities
            required_types: List of required entity types
            
        Returns:
            bool: True if all required types are present
        """
        entity_types = {entity.type for entity in entities}
        for required_type in required_types:
            if required_type not in entity_types:
                return False
        return True
    
    async def _extract_event_details(self, event_type: str, text: str, entities: List[Entity]) -> Dict[str, Any]:
        """Extract event details using LLM.
        
        Args:
            event_type: Type of event
            text: Input text
            entities: Extracted entities
            
        Returns:
            Dict[str, Any]: Event details
        """
        template = self.event_templates.get(event_type)
        if not template or "prompt" not in template:
            return {}
        
        # Format prompt with text
        prompt = template["prompt"].format(text=text)
        
        try:
            # Call LLM
            response = await llm_client.generate_text(prompt)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response (in case LLM adds extra text)
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error extracting event details: {e}")
            return {}
    
    async def extract_events(self, text: str, entities: Optional[List[Entity]] = None, 
                           event_types: Optional[List[str]] = None) -> List[Event]:
        """Extract events from text.
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities
            event_types: Optional list of event types to extract
            
        Returns:
            List[Event]: Extracted events
        """
        # Check cache
        cache_key = f"event:{hash(text)}"
        cached_result = event_cache.get(cache_key)
        
        if cached_result:
            # Convert cached result back to Event objects
            return [Event(**event_dict) for event_dict in cached_result]
        
        # Filter event types
        if not event_types:
            event_types = self.event_types
        
        # Extract entities if not provided
        if not entities:
            entities = entity_extractor.extract_entities(text)
        
        # Extract events
        events = []
        
        for event_type in event_types:
            template = self.event_templates.get(event_type)
            if not template:
                continue
            
            # Check pattern match
            if "patterns" in template and not self._check_pattern_match(text, template["patterns"]):
                continue
            
            # Check required entities
            if "required_entities" in template and not self._check_required_entities(entities, template["required_entities"]):
                continue
            
            # Extract event details
            details = await self._extract_event_details(event_type, text, entities)
            
            # Check if event was detected
            is_event_key = f"is_{event_type.lower()}"
            if is_event_key in details and not details[is_event_key]:
                continue
            
            # Create event
            event = Event(
                type=event_type,
                text=text,
                entities=entities,
                metadata=details
            )
            
            events.append(event)
        
        # Cache result
        event_cache.set(
            cache_key, 
            [event.dict() for event in events],
            ttl=3600  # 1 hour
        )
        
        return events

# Create extractor instance
event_extractor = EventExtractor()
