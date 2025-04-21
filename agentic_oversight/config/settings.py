"""
Configuration settings for the agentic oversight system.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseSettings, Field

class AgentSettings(BaseSettings):
    """Settings for agents."""
    enabled: bool = True
    max_concurrent_agents: int = 10
    agent_timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    log_agent_actions: bool = True
    log_agent_reasoning: bool = True
    agent_types: List[str] = [
        "monitoring", 
        "decision", 
        "explanation", 
        "learning", 
        "human_interface"
    ]
    default_agent_config: Dict[str, Any] = {
        "temperature": 0.2,
        "max_tokens": 2000,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    agent_specific_config: Dict[str, Dict[str, Any]] = {
        "monitoring": {
            "temperature": 0.1,
            "polling_interval_seconds": 30,
            "alert_threshold": 0.7
        },
        "decision": {
            "temperature": 0.3,
            "decision_timeout_seconds": 45,
            "confidence_threshold": 0.8
        },
        "explanation": {
            "temperature": 0.7,
            "max_tokens": 3000,
            "detail_level": "high"
        },
        "learning": {
            "temperature": 0.2,
            "learning_rate": 0.01,
            "batch_size": 32
        },
        "human_interface": {
            "temperature": 0.7,
            "max_tokens": 4000,
            "style": "conversational"
        }
    }

class LLMSettings(BaseSettings):
    """Settings for LLM integration."""
    provider: str = "openai"  # openai, anthropic, local, etc.
    model: str = "gpt-4"
    api_key_env_var: str = "OPENAI_API_KEY"
    api_base_url: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 30
    streaming: bool = False
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    cache_responses: bool = True
    cache_ttl_seconds: int = 3600
    max_context_length: int = 16000
    token_limit_buffer: int = 1000
    cost_tracking_enabled: bool = True

class MemorySettings(BaseSettings):
    """Settings for agent memory."""
    memory_type: str = "hybrid"  # vector, relational, hybrid
    vector_db_connection: str = "redis://localhost:6379/1"
    relational_db_connection: str = "sqlite:///agentic_oversight.db"
    short_term_memory_ttl_seconds: int = 3600
    long_term_memory_enabled: bool = True
    memory_retrieval_strategy: str = "semantic"  # semantic, temporal, hybrid
    max_memory_items: int = 1000
    memory_embedding_model: str = "text-embedding-ada-002"
    memory_chunk_size: int = 1000
    memory_chunk_overlap: int = 200
    similarity_threshold: float = 0.7
    memory_refresh_interval_seconds: int = 300

class ReasoningSettings(BaseSettings):
    """Settings for agent reasoning."""
    reasoning_framework: str = "react"  # react, cot, tot, etc.
    max_reasoning_steps: int = 10
    reasoning_detail_level: str = "high"  # low, medium, high
    include_uncertainty: bool = True
    include_alternatives: bool = True
    max_alternatives: int = 3
    reasoning_timeout_seconds: int = 45
    structured_reasoning: bool = True
    reasoning_templates: Dict[str, str] = {
        "decision": "Given the following context:\n{context}\n\nI need to make a decision about {decision_point}.\n\nLet me think through this step by step:\n1. What are the key factors to consider?\n2. What are the potential options?\n3. What are the pros and cons of each option?\n4. What is the recommended decision?\n\nFactors to consider:\n{reasoning}\n\nRecommended decision: {decision}",
        "monitoring": "Based on the following data:\n{data}\n\nI need to determine if there are any anomalies or issues that require attention.\n\nAnalysis:\n{reasoning}\n\nFindings: {findings}\nAlert level: {alert_level}\nRecommended actions: {actions}",
        "explanation": "I need to explain the following decision:\n{decision}\n\nContext:\n{context}\n\nExplanation:\n{explanation}\n\nKey factors that influenced this decision:\n{factors}\n\nPotential alternatives considered:\n{alternatives}\n\nConfidence level: {confidence}"
    }

class CoordinatorSettings(BaseSettings):
    """Settings for agent coordinator."""
    coordination_strategy: str = "hierarchical"  # hierarchical, democratic, market
    max_coordination_rounds: int = 5
    coordination_timeout_seconds: int = 120
    conflict_resolution_strategy: str = "consensus"  # consensus, authority, voting
    task_prioritization_method: str = "importance_urgency"  # fifo, importance, urgency, importance_urgency
    task_assignment_method: str = "specialized"  # round_robin, specialized, load_balanced
    coordination_check_interval_seconds: int = 10
    coordination_log_level: str = "INFO"
    max_tasks_per_agent: int = 5
    task_timeout_multiplier: float = 2.0

class HumanInterfaceSettings(BaseSettings):
    """Settings for human interface."""
    interface_type: str = "api"  # api, cli, web, etc.
    notification_channels: List[str] = ["api", "email"]
    notification_levels: List[str] = ["critical", "warning", "info"]
    email_server: Optional[str] = None
    email_port: int = 587
    email_username: Optional[str] = None
    email_password_env_var: str = "EMAIL_PASSWORD"
    email_recipients: List[str] = []
    notification_cooldown_seconds: int = 300
    human_approval_required_for: List[str] = [
        "strategy_change", 
        "risk_limit_override", 
        "large_order"
    ]
    approval_timeout_seconds: int = 600
    default_approval_action: str = "reject"  # approve, reject
    interface_style: str = "detailed"  # minimal, standard, detailed
    explanation_detail_level: str = "high"  # low, medium, high

class MonitoringSettings(BaseSettings):
    """Settings for system monitoring."""
    monitoring_interval_seconds: int = 30
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 60
    performance_tracking_enabled: bool = True
    anomaly_detection_enabled: bool = True
    anomaly_detection_sensitivity: float = 0.7
    alert_thresholds: Dict[str, Dict[str, float]] = {
        "data_ingestion": {
            "latency_seconds": 5.0,
            "error_rate": 0.01,
            "data_freshness_seconds": 300
        },
        "semantic_signal": {
            "processing_time_seconds": 10.0,
            "error_rate": 0.02,
            "signal_quality": 0.7
        },
        "strategy_generator": {
            "generation_time_seconds": 15.0,
            "error_rate": 0.02,
            "strategy_quality": 0.7
        },
        "execution_engine": {
            "order_latency_seconds": 2.0,
            "error_rate": 0.005,
            "fill_rate": 0.95
        },
        "risk_management": {
            "processing_time_seconds": 5.0,
            "error_rate": 0.01,
            "limit_breach_rate": 0.05
        }
    }
    log_retention_days: int = 30
    metrics_retention_days: int = 90
    system_status_endpoint: str = "/api/system/status"

class APISettings(BaseSettings):
    """Settings for API."""
    host: str = "0.0.0.0"
    port: int = 8004
    debug: bool = False
    reload: bool = False
    workers: int = 1
    timeout_keep_alive: int = 5
    access_log: bool = True
    cors_origins: List[str] = ["*"]
    api_keys: List[str] = ["development_api_key"]
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_timeframe_seconds: int = 60
    docs_enabled: bool = True

class LoggingSettings(BaseSettings):
    """Settings for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "/var/log/trading_pipeline/agentic_oversight.log"
    console_output: bool = True
    log_agent_actions: bool = True
    log_agent_reasoning: bool = True
    log_coordinator_decisions: bool = True
    log_human_interactions: bool = True
    log_system_metrics: bool = True
    log_rotation_days: int = 7
    log_max_size_mb: int = 100
    log_compression: bool = True
    log_backup_count: int = 5

class Settings(BaseSettings):
    """Main settings for agentic oversight system."""
    environment: str = Field("development", env="TRADING_ENV")
    agents: AgentSettings = AgentSettings()
    llm: LLMSettings = LLMSettings()
    memory: MemorySettings = MemorySettings()
    reasoning: ReasoningSettings = ReasoningSettings()
    coordinator: CoordinatorSettings = CoordinatorSettings()
    human_interface: HumanInterfaceSettings = HumanInterfaceSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    data_ingestion_api_url: str = "http://localhost:8000/api"
    semantic_signal_api_url: str = "http://localhost:8001/api"
    strategy_generator_api_url: str = "http://localhost:8002/api"
    execution_engine_api_url: str = "http://localhost:8003/api"
    risk_management_api_url: str = "http://localhost:8004/api"
    dashboard_api_url: str = "http://localhost:8005/api"
    
    class Config:
        env_prefix = "AGENTIC_"
        env_nested_delimiter = "__"

# Create settings instance
settings = Settings()
