"""
Base models for the agentic oversight system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import uuid

# Enums
class AgentType(str, Enum):
    """Agent type enumeration."""
    MONITORING = "monitoring"
    DECISION = "decision"
    EXPLANATION = "explanation"
    LEARNING = "learning"
    HUMAN_INTERFACE = "human_interface"

class AgentStatus(str, Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    PAUSED = "paused"
    TERMINATED = "terminated"

class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(str, Enum):
    """Task priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertLevel(str, Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class DecisionConfidence(str, Enum):
    """Decision confidence enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ApprovalStatus(str, Enum):
    """Approval status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ReasoningFramework(str, Enum):
    """Reasoning framework enumeration."""
    REACT = "react"
    COT = "cot"
    TOT = "tot"
    CUSTOM = "custom"

class CoordinationStrategy(str, Enum):
    """Coordination strategy enumeration."""
    HIERARCHICAL = "hierarchical"
    DEMOCRATIC = "democratic"
    MARKET = "market"

class ConflictResolutionStrategy(str, Enum):
    """Conflict resolution strategy enumeration."""
    CONSENSUS = "consensus"
    AUTHORITY = "authority"
    VOTING = "voting"

class MemoryType(str, Enum):
    """Memory type enumeration."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

# Base Models
class Agent(BaseModel):
    """Agent model."""
    id: str = Field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    name: str
    type: AgentType
    status: AgentStatus = AgentStatus.IDLE
    capabilities: List[str] = []
    config: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "agent_12345678",
                "name": "MarketMonitor",
                "type": "monitoring",
                "status": "idle",
                "capabilities": ["market_data_analysis", "anomaly_detection"],
                "config": {
                    "temperature": 0.1,
                    "polling_interval_seconds": 30
                },
                "created_at": "2023-01-01T10:00:00Z",
                "last_active": "2023-01-01T10:00:00Z"
            }
        }

class Task(BaseModel):
    """Task model."""
    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    type: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None
    created_by: Optional[str] = None
    due_by: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    dependencies: List[str] = []
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "task_12345678",
                "title": "Analyze Market Volatility",
                "description": "Analyze current market volatility and detect anomalies",
                "type": "market_analysis",
                "status": "pending",
                "priority": "high",
                "created_at": "2023-01-01T10:00:00Z",
                "updated_at": "2023-01-01T10:00:00Z",
                "assigned_to": "agent_12345678",
                "created_by": "coordinator",
                "due_by": "2023-01-01T10:05:00Z",
                "input_data": {
                    "market_data": "https://api.example.com/market_data",
                    "timeframe": "1h"
                }
            }
        }

class Memory(BaseModel):
    """Memory model."""
    id: str = Field(default_factory=lambda: f"memory_{uuid.uuid4().hex[:8]}")
    agent_id: str
    type: MemoryType
    content: str
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    context: Dict[str, Any] = {}
    importance: float = 0.5
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "memory_12345678",
                "agent_id": "agent_12345678",
                "type": "short_term",
                "content": "Market volatility increased by 15% in the last hour",
                "created_at": "2023-01-01T10:00:00Z",
                "expires_at": "2023-01-01T13:00:00Z",
                "context": {
                    "market": "US Equities",
                    "timeframe": "1h"
                },
                "importance": 0.8
            }
        }

class Reasoning(BaseModel):
    """Reasoning model."""
    id: str = Field(default_factory=lambda: f"reasoning_{uuid.uuid4().hex[:8]}")
    agent_id: str
    task_id: str
    framework: ReasoningFramework
    steps: List[Dict[str, Any]] = []
    conclusion: str
    confidence: float
    alternatives: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: int
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "reasoning_12345678",
                "agent_id": "agent_12345678",
                "task_id": "task_12345678",
                "framework": "react",
                "steps": [
                    {
                        "step": 1,
                        "thought": "I need to analyze the market volatility data",
                        "action": "retrieve_market_data",
                        "observation": "VIX is at 25, up 15% from yesterday"
                    },
                    {
                        "step": 2,
                        "thought": "This is a significant increase in volatility",
                        "action": "check_historical_context",
                        "observation": "This is 2 standard deviations above the 30-day average"
                    }
                ],
                "conclusion": "Market volatility has increased significantly and is now at an elevated level",
                "confidence": 0.85,
                "alternatives": [
                    {
                        "conclusion": "Market volatility increase is temporary and likely to revert",
                        "confidence": 0.35
                    }
                ],
                "created_at": "2023-01-01T10:01:00Z",
                "duration_ms": 1200
            }
        }

class Decision(BaseModel):
    """Decision model."""
    id: str = Field(default_factory=lambda: f"decision_{uuid.uuid4().hex[:8]}")
    agent_id: str
    task_id: str
    reasoning_id: str
    decision: str
    confidence: DecisionConfidence
    justification: str
    alternatives: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    requires_approval: bool = False
    approval_status: Optional[ApprovalStatus] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "decision_12345678",
                "agent_id": "agent_12345678",
                "task_id": "task_12345678",
                "reasoning_id": "reasoning_12345678",
                "decision": "Reduce position sizes by 20% due to increased market volatility",
                "confidence": "high",
                "justification": "Market volatility has increased significantly and is now 2 standard deviations above the 30-day average",
                "alternatives": [
                    {
                        "decision": "Maintain current positions",
                        "confidence": "medium",
                        "justification": "Volatility increase may be temporary"
                    },
                    {
                        "decision": "Hedge with VIX futures",
                        "confidence": "medium",
                        "justification": "Direct hedge against volatility"
                    }
                ],
                "created_at": "2023-01-01T10:02:00Z",
                "requires_approval": True,
                "approval_status": "pending"
            }
        }

class Alert(BaseModel):
    """Alert model."""
    id: str = Field(default_factory=lambda: f"alert_{uuid.uuid4().hex[:8]}")
    agent_id: str
    level: AlertLevel
    title: str
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[str] = None
    actions: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "alert_12345678",
                "agent_id": "agent_12345678",
                "level": "warning",
                "title": "Increased Market Volatility",
                "message": "Market volatility has increased by 15% in the last hour, which is 2 standard deviations above the 30-day average",
                "created_at": "2023-01-01T10:01:30Z",
                "acknowledged": False,
                "related_entity_type": "market",
                "related_entity_id": "us_equities",
                "actions": [
                    {
                        "type": "reduce_exposure",
                        "description": "Reduce position sizes by 20%",
                        "url": "/api/actions/reduce_exposure"
                    },
                    {
                        "type": "acknowledge",
                        "description": "Acknowledge alert",
                        "url": "/api/alerts/12345678/acknowledge"
                    }
                ]
            }
        }

class Explanation(BaseModel):
    """Explanation model."""
    id: str = Field(default_factory=lambda: f"explanation_{uuid.uuid4().hex[:8]}")
    agent_id: str
    decision_id: str
    explanation: str
    detail_level: str
    audience: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    factors: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "explanation_12345678",
                "agent_id": "agent_12345678",
                "decision_id": "decision_12345678",
                "explanation": "We are reducing position sizes by 20% because market volatility has increased significantly. The VIX index, which measures market volatility, is up 15% today and is now at levels that historically indicate increased risk. By reducing position sizes, we aim to decrease overall portfolio risk while maintaining exposure to our core positions.",
                "detail_level": "high",
                "audience": "portfolio_manager",
                "created_at": "2023-01-01T10:03:00Z",
                "factors": [
                    {
                        "name": "VIX Index",
                        "value": "25",
                        "change": "+15%",
                        "importance": "high"
                    },
                    {
                        "name": "Historical Context",
                        "value": "2 standard deviations above 30-day average",
                        "importance": "medium"
                    }
                ],
                "context": {
                    "market_conditions": "Elevated volatility",
                    "portfolio_exposure": "80% invested",
                    "risk_profile": "moderate"
                }
            }
        }

class Coordination(BaseModel):
    """Coordination model."""
    id: str = Field(default_factory=lambda: f"coordination_{uuid.uuid4().hex[:8]}")
    strategy: CoordinationStrategy
    agents: List[str]
    tasks: List[str]
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Dict[str, Any] = {}
    conflicts: List[Dict[str, Any]] = []
    resolution_strategy: ConflictResolutionStrategy
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "coordination_12345678",
                "strategy": "hierarchical",
                "agents": ["agent_12345678", "agent_23456789"],
                "tasks": ["task_12345678", "task_23456789"],
                "status": "completed",
                "created_at": "2023-01-01T10:00:00Z",
                "updated_at": "2023-01-01T10:05:00Z",
                "completed_at": "2023-01-01T10:05:00Z",
                "result": {
                    "decision": "Reduce position sizes by 20%",
                    "confidence": "high"
                },
                "conflicts": [
                    {
                        "agents": ["agent_12345678", "agent_23456789"],
                        "topic": "reduction_amount",
                        "values": ["20%", "15%"],
                        "resolution": "20%",
                        "resolution_method": "authority"
                    }
                ],
                "resolution_strategy": "authority"
            }
        }

class HumanInteraction(BaseModel):
    """Human interaction model."""
    id: str = Field(default_factory=lambda: f"interaction_{uuid.uuid4().hex[:8]}")
    type: str
    user_id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    responded_at: Optional[datetime] = None
    response: Optional[str] = None
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "interaction_12345678",
                "type": "approval_request",
                "user_id": "user_12345678",
                "content": "Please approve the decision to reduce position sizes by 20% due to increased market volatility",
                "created_at": "2023-01-01T10:03:30Z",
                "responded_at": "2023-01-01T10:04:15Z",
                "response": "Approved",
                "related_entity_type": "decision",
                "related_entity_id": "decision_12345678"
            }
        }

class SystemStatus(BaseModel):
    """System status model."""
    id: str = Field(default_factory=lambda: f"status_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, Dict[str, Any]] = {}
    agents: Dict[str, Dict[str, Any]] = {}
    tasks: Dict[str, int] = {}
    alerts: Dict[str, int] = {}
    performance_metrics: Dict[str, Any] = {}
    overall_status: str
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "id": "status_12345678",
                "timestamp": "2023-01-01T10:05:00Z",
                "components": {
                    "data_ingestion": {
                        "status": "healthy",
                        "latency_ms": 250,
                        "error_rate": 0.001
                    },
                    "semantic_signal": {
                        "status": "healthy",
                        "latency_ms": 450,
                        "error_rate": 0.002
                    },
                    "strategy_generator": {
                        "status": "healthy",
                        "latency_ms": 550,
                        "error_rate": 0.0
                    },
                    "execution_engine": {
                        "status": "healthy",
                        "latency_ms": 120,
                        "error_rate": 0.0
                    },
                    "risk_management": {
                        "status": "healthy",
                        "latency_ms": 180,
                        "error_rate": 0.0
                    }
                },
                "agents": {
                    "monitoring": {
                        "count": 3,
                        "active": 2,
                        "idle": 1,
                        "error": 0
                    },
                    "decision": {
                        "count": 2,
                        "active": 1,
                        "idle": 1,
                        "error": 0
                    }
                },
                "tasks": {
                    "pending": 2,
                    "in_progress": 3,
                    "completed": 15,
                    "failed": 0
                },
                "alerts": {
                    "info": 5,
                    "warning": 1,
                    "critical": 0
                },
                "performance_metrics": {
                    "cpu_usage": 0.35,
                    "memory_usage": 0.42,
                    "llm_requests_per_minute": 12.5,
                    "average_llm_latency_ms": 850
                },
                "overall_status": "healthy"
            }
        }

# Request/Response Models
class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""
    name: str
    type: AgentType
    capabilities: List[str] = []
    config: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "name": "MarketMonitor",
                "type": "monitoring",
                "capabilities": ["market_data_analysis", "anomaly_detection"],
                "config": {
                    "temperature": 0.1,
                    "polling_interval_seconds": 30
                }
            }
        }

class CreateAgentResponse(BaseModel):
    """Response model for creating an agent."""
    agent: Agent
    
    class Config:
        schema_extra = {
            "example": {
                "agent": {
                    "id": "agent_12345678",
                    "name": "MarketMonitor",
                    "type": "monitoring",
                    "status": "idle",
                    "capabilities": ["market_data_analysis", "anomaly_detection"],
                    "config": {
                        "temperature": 0.1,
                        "polling_interval_seconds": 30
                    },
                    "created_at": "2023-01-01T10:00:00Z",
                    "last_active": "2023-01-01T10:00:00Z"
                }
            }
        }

class CreateTaskRequest(BaseModel):
    """Request model for creating a task."""
    title: str
    description: str
    type: str
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_to: Optional[str] = None
    created_by: Optional[str] = None
    due_by: Optional[datetime] = None
    input_data: Dict[str, Any] = {}
    dependencies: List[str] = []
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Analyze Market Volatility",
                "description": "Analyze current market volatility and detect anomalies",
                "type": "market_analysis",
                "priority": "high",
                "assigned_to": "agent_12345678",
                "created_by": "coordinator",
                "due_by": "2023-01-01T10:05:00Z",
                "input_data": {
                    "market_data": "https://api.example.com/market_data",
                    "timeframe": "1h"
                },
                "tags": ["market", "volatility", "analysis"]
            }
        }

class CreateTaskResponse(BaseModel):
    """Response model for creating a task."""
    task: Task
    
    class Config:
        schema_extra = {
            "example": {
                "task": {
                    "id": "task_12345678",
                    "title": "Analyze Market Volatility",
                    "description": "Analyze current market volatility and detect anomalies",
                    "type": "market_analysis",
                    "status": "pending",
                    "priority": "high",
                    "created_at": "2023-01-01T10:00:00Z",
                    "updated_at": "2023-01-01T10:00:00Z",
                    "assigned_to": "agent_12345678",
                    "created_by": "coordinator",
                    "due_by": "2023-01-01T10:05:00Z",
                    "input_data": {
                        "market_data": "https://api.example.com/market_data",
                        "timeframe": "1h"
                    },
                    "tags": ["market", "volatility", "analysis"]
                }
            }
        }

class CreateMemoryRequest(BaseModel):
    """Request model for creating a memory."""
    agent_id: str
    type: MemoryType
    content: str
    context: Dict[str, Any] = {}
    importance: float = 0.5
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "agent_12345678",
                "type": "short_term",
                "content": "Market volatility increased by 15% in the last hour",
                "context": {
                    "market": "US Equities",
                    "timeframe": "1h"
                },
                "importance": 0.8,
                "expires_at": "2023-01-01T13:00:00Z"
            }
        }

class CreateMemoryResponse(BaseModel):
    """Response model for creating a memory."""
    memory: Memory
    
    class Config:
        schema_extra = {
            "example": {
                "memory": {
                    "id": "memory_12345678",
                    "agent_id": "agent_12345678",
                    "type": "short_term",
                    "content": "Market volatility increased by 15% in the last hour",
                    "created_at": "2023-01-01T10:00:00Z",
                    "expires_at": "2023-01-01T13:00:00Z",
                    "context": {
                        "market": "US Equities",
                        "timeframe": "1h"
                    },
                    "importance": 0.8
                }
            }
        }

class CreateDecisionRequest(BaseModel):
    """Request model for creating a decision."""
    agent_id: str
    task_id: str
    reasoning_id: str
    decision: str
    confidence: DecisionConfidence
    justification: str
    alternatives: List[Dict[str, Any]] = []
    requires_approval: bool = False
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "agent_12345678",
                "task_id": "task_12345678",
                "reasoning_id": "reasoning_12345678",
                "decision": "Reduce position sizes by 20% due to increased market volatility",
                "confidence": "high",
                "justification": "Market volatility has increased significantly and is now 2 standard deviations above the 30-day average",
                "alternatives": [
                    {
                        "decision": "Maintain current positions",
                        "confidence": "medium",
                        "justification": "Volatility increase may be temporary"
                    },
                    {
                        "decision": "Hedge with VIX futures",
                        "confidence": "medium",
                        "justification": "Direct hedge against volatility"
                    }
                ],
                "requires_approval": True
            }
        }

class CreateDecisionResponse(BaseModel):
    """Response model for creating a decision."""
    decision: Decision
    
    class Config:
        schema_extra = {
            "example": {
                "decision": {
                    "id": "decision_12345678",
                    "agent_id": "agent_12345678",
                    "task_id": "task_12345678",
                    "reasoning_id": "reasoning_12345678",
                    "decision": "Reduce position sizes by 20% due to increased market volatility",
                    "confidence": "high",
                    "justification": "Market volatility has increased significantly and is now 2 standard deviations above the 30-day average",
                    "alternatives": [
                        {
                            "decision": "Maintain current positions",
                            "confidence": "medium",
                            "justification": "Volatility increase may be temporary"
                        },
                        {
                            "decision": "Hedge with VIX futures",
                            "confidence": "medium",
                            "justification": "Direct hedge against volatility"
                        }
                    ],
                    "created_at": "2023-01-01T10:02:00Z",
                    "requires_approval": True,
                    "approval_status": "pending"
                }
            }
        }

class CreateAlertRequest(BaseModel):
    """Request model for creating an alert."""
    agent_id: str
    level: AlertLevel
    title: str
    message: str
    expires_at: Optional[datetime] = None
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[str] = None
    actions: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "agent_12345678",
                "level": "warning",
                "title": "Increased Market Volatility",
                "message": "Market volatility has increased by 15% in the last hour, which is 2 standard deviations above the 30-day average",
                "related_entity_type": "market",
                "related_entity_id": "us_equities",
                "actions": [
                    {
                        "type": "reduce_exposure",
                        "description": "Reduce position sizes by 20%",
                        "url": "/api/actions/reduce_exposure"
                    },
                    {
                        "type": "acknowledge",
                        "description": "Acknowledge alert",
                        "url": "/api/alerts/12345678/acknowledge"
                    }
                ]
            }
        }

class CreateAlertResponse(BaseModel):
    """Response model for creating an alert."""
    alert: Alert
    
    class Config:
        schema_extra = {
            "example": {
                "alert": {
                    "id": "alert_12345678",
                    "agent_id": "agent_12345678",
                    "level": "warning",
                    "title": "Increased Market Volatility",
                    "message": "Market volatility has increased by 15% in the last hour, which is 2 standard deviations above the 30-day average",
                    "created_at": "2023-01-01T10:01:30Z",
                    "acknowledged": False,
                    "related_entity_type": "market",
                    "related_entity_id": "us_equities",
                    "actions": [
                        {
                            "type": "reduce_exposure",
                            "description": "Reduce position sizes by 20%",
                            "url": "/api/actions/reduce_exposure"
                        },
                        {
                            "type": "acknowledge",
                            "description": "Acknowledge alert",
                            "url": "/api/alerts/12345678/acknowledge"
                        }
                    ]
                }
            }
        }

class CreateExplanationRequest(BaseModel):
    """Request model for creating an explanation."""
    agent_id: str
    decision_id: str
    detail_level: str
    audience: str
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "agent_12345678",
                "decision_id": "decision_12345678",
                "detail_level": "high",
                "audience": "portfolio_manager",
                "context": {
                    "market_conditions": "Elevated volatility",
                    "portfolio_exposure": "80% invested",
                    "risk_profile": "moderate"
                }
            }
        }

class CreateExplanationResponse(BaseModel):
    """Response model for creating an explanation."""
    explanation: Explanation
    
    class Config:
        schema_extra = {
            "example": {
                "explanation": {
                    "id": "explanation_12345678",
                    "agent_id": "agent_12345678",
                    "decision_id": "decision_12345678",
                    "explanation": "We are reducing position sizes by 20% because market volatility has increased significantly. The VIX index, which measures market volatility, is up 15% today and is now at levels that historically indicate increased risk. By reducing position sizes, we aim to decrease overall portfolio risk while maintaining exposure to our core positions.",
                    "detail_level": "high",
                    "audience": "portfolio_manager",
                    "created_at": "2023-01-01T10:03:00Z",
                    "factors": [
                        {
                            "name": "VIX Index",
                            "value": "25",
                            "change": "+15%",
                            "importance": "high"
                        },
                        {
                            "name": "Historical Context",
                            "value": "2 standard deviations above 30-day average",
                            "importance": "medium"
                        }
                    ],
                    "context": {
                        "market_conditions": "Elevated volatility",
                        "portfolio_exposure": "80% invested",
                        "risk_profile": "moderate"
                    }
                }
            }
        }

class ApprovalRequest(BaseModel):
    """Request model for approval."""
    decision_id: str
    user_id: str
    approval_status: ApprovalStatus
    comment: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "decision_id": "decision_12345678",
                "user_id": "user_12345678",
                "approval_status": "approved",
                "comment": "Approved based on current market conditions"
            }
        }

class ApprovalResponse(BaseModel):
    """Response model for approval."""
    decision: Decision
    
    class Config:
        schema_extra = {
            "example": {
                "decision": {
                    "id": "decision_12345678",
                    "agent_id": "agent_12345678",
                    "task_id": "task_12345678",
                    "reasoning_id": "reasoning_12345678",
                    "decision": "Reduce position sizes by 20% due to increased market volatility",
                    "confidence": "high",
                    "justification": "Market volatility has increased significantly and is now 2 standard deviations above the 30-day average",
                    "created_at": "2023-01-01T10:02:00Z",
                    "requires_approval": True,
                    "approval_status": "approved",
                    "approved_by": "user_12345678",
                    "approved_at": "2023-01-01T10:04:15Z"
                }
            }
        }

class GetSystemStatusResponse(BaseModel):
    """Response model for getting system status."""
    status: SystemStatus
    
    class Config:
        schema_extra = {
            "example": {
                "status": {
                    "id": "status_12345678",
                    "timestamp": "2023-01-01T10:05:00Z",
                    "components": {
                        "data_ingestion": {
                            "status": "healthy",
                            "latency_ms": 250,
                            "error_rate": 0.001
                        },
                        "semantic_signal": {
                            "status": "healthy",
                            "latency_ms": 450,
                            "error_rate": 0.002
                        },
                        "strategy_generator": {
                            "status": "healthy",
                            "latency_ms": 550,
                            "error_rate": 0.0
                        },
                        "execution_engine": {
                            "status": "healthy",
                            "latency_ms": 120,
                            "error_rate": 0.0
                        },
                        "risk_management": {
                            "status": "healthy",
                            "latency_ms": 180,
                            "error_rate": 0.0
                        }
                    },
                    "agents": {
                        "monitoring": {
                            "count": 3,
                            "active": 2,
                            "idle": 1,
                            "error": 0
                        },
                        "decision": {
                            "count": 2,
                            "active": 1,
                            "idle": 1,
                            "error": 0
                        }
                    },
                    "tasks": {
                        "pending": 2,
                        "in_progress": 3,
                        "completed": 15,
                        "failed": 0
                    },
                    "alerts": {
                        "info": 5,
                        "warning": 1,
                        "critical": 0
                    },
                    "performance_metrics": {
                        "cpu_usage": 0.35,
                        "memory_usage": 0.42,
                        "llm_requests_per_minute": 12.5,
                        "average_llm_latency_ms": 850
                    },
                    "overall_status": "healthy"
                }
            }
        }
