"""
Configuration settings for the dashboard interface.
"""

import os
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseSettings, Field

class ServerSettings(BaseSettings):
    """Server settings."""
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, description="Server port")
    debug: bool = Field(False, description="Debug mode")
    reload: bool = Field(False, description="Auto-reload on code changes")
    workers: int = Field(4, description="Number of worker processes")
    timeout: int = Field(60, description="Request timeout in seconds")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    static_dir: str = Field("static", description="Static files directory")
    static_url: str = Field("/static", description="Static files URL path")
    api_prefix: str = Field("/api", description="API URL prefix")
    docs_url: str = Field("/docs", description="API documentation URL")
    openapi_url: str = Field("/openapi.json", description="OpenAPI schema URL")

class AuthSettings(BaseSettings):
    """Authentication settings."""
    secret_key: str = Field(os.environ.get("DASHBOARD_SECRET_KEY", "supersecretkey"), description="JWT secret key")
    algorithm: str = Field("HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(30, description="Access token expiration time in minutes")
    refresh_token_expire_days: int = Field(7, description="Refresh token expiration time in days")
    password_min_length: int = Field(8, description="Minimum password length")
    password_require_special: bool = Field(True, description="Require special characters in password")
    password_require_number: bool = Field(True, description="Require numbers in password")
    password_require_uppercase: bool = Field(True, description="Require uppercase letters in password")
    password_require_lowercase: bool = Field(True, description="Require lowercase letters in password")
    max_login_attempts: int = Field(5, description="Maximum login attempts before lockout")
    lockout_minutes: int = Field(15, description="Account lockout time in minutes after max login attempts")

class DatabaseSettings(BaseSettings):
    """Database settings."""
    url: str = Field(os.environ.get("DASHBOARD_DATABASE_URL", "sqlite:///./dashboard.db"), description="Database URL")
    pool_size: int = Field(5, description="Database connection pool size")
    max_overflow: int = Field(10, description="Maximum overflow connections")
    pool_timeout: int = Field(30, description="Connection pool timeout in seconds")
    pool_recycle: int = Field(1800, description="Connection recycle time in seconds")
    echo: bool = Field(False, description="Echo SQL statements")
    connect_args: Dict[str, Any] = Field(default_factory=dict, description="Additional connection arguments")

class CacheSettings(BaseSettings):
    """Cache settings."""
    url: str = Field(os.environ.get("DASHBOARD_CACHE_URL", "redis://localhost:6379/0"), description="Cache URL")
    ttl: int = Field(300, description="Default cache TTL in seconds")
    prefix: str = Field("dashboard:", description="Cache key prefix")
    enable_page_cache: bool = Field(True, description="Enable page caching")
    enable_api_cache: bool = Field(True, description="Enable API response caching")
    page_cache_ttl: int = Field(60, description="Page cache TTL in seconds")
    api_cache_ttl: int = Field(30, description="API cache TTL in seconds")

class LoggingSettings(BaseSettings):
    """Logging settings."""
    level: str = Field("INFO", description="Logging level")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file: Optional[str] = Field(None, description="Log file path")
    max_size: int = Field(10 * 1024 * 1024, description="Maximum log file size in bytes")
    backup_count: int = Field(5, description="Number of backup log files")
    console: bool = Field(True, description="Log to console")
    json_format: bool = Field(False, description="Use JSON log format")

class UISettings(BaseSettings):
    """UI settings."""
    theme: str = Field("light", description="Default UI theme")
    allow_theme_change: bool = Field(True, description="Allow users to change theme")
    refresh_interval: int = Field(5, description="Data refresh interval in seconds")
    max_items_per_page: int = Field(100, description="Maximum items per page")
    default_items_per_page: int = Field(20, description="Default items per page")
    chart_animation: bool = Field(True, description="Enable chart animations")
    enable_notifications: bool = Field(True, description="Enable UI notifications")
    notification_duration: int = Field(5, description="Notification duration in seconds")
    date_format: str = Field("YYYY-MM-DD", description="Date format")
    time_format: str = Field("HH:mm:ss", description="Time format")
    timezone: str = Field("UTC", description="Default timezone")
    allow_timezone_change: bool = Field(True, description="Allow users to change timezone")
    default_currency: str = Field("USD", description="Default currency")
    number_format: str = Field("0,0.00", description="Number format")
    percentage_format: str = Field("0.00%", description="Percentage format")

class ComponentSettings(BaseSettings):
    """Component integration settings."""
    data_ingestion_url: str = Field("http://localhost:8001", description="Data ingestion service URL")
    semantic_signal_url: str = Field("http://localhost:8002", description="Semantic signal service URL")
    strategy_generator_url: str = Field("http://localhost:8003", description="Strategy generator service URL")
    execution_engine_url: str = Field("http://localhost:8004", description="Execution engine service URL")
    risk_management_url: str = Field("http://localhost:8005", description="Risk management service URL")
    agentic_oversight_url: str = Field("http://localhost:8006", description="Agentic oversight service URL")
    connection_timeout: int = Field(5, description="Component connection timeout in seconds")
    request_timeout: int = Field(30, description="Component request timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts for component requests")
    retry_delay: int = Field(1, description="Delay between retry attempts in seconds")
    health_check_interval: int = Field(60, description="Component health check interval in seconds")

class WebSocketSettings(BaseSettings):
    """WebSocket settings."""
    enable: bool = Field(True, description="Enable WebSocket connections")
    path: str = Field("/ws", description="WebSocket path")
    ping_interval: int = Field(30, description="Ping interval in seconds")
    ping_timeout: int = Field(10, description="Ping timeout in seconds")
    max_connections: int = Field(1000, description="Maximum WebSocket connections")
    max_message_size: int = Field(1024 * 1024, description="Maximum message size in bytes")
    compression: bool = Field(True, description="Enable WebSocket compression")

class Settings(BaseSettings):
    """Dashboard settings."""
    environment: str = Field(os.environ.get("ENVIRONMENT", "development"), description="Environment (development, staging, production)")
    debug: bool = Field(os.environ.get("DEBUG", "False").lower() == "true", description="Debug mode")
    testing: bool = Field(os.environ.get("TESTING", "False").lower() == "true", description="Testing mode")
    
    server: ServerSettings = Field(default_factory=ServerSettings, description="Server settings")
    auth: AuthSettings = Field(default_factory=AuthSettings, description="Authentication settings")
    database: DatabaseSettings = Field(default_factory=DatabaseSettings, description="Database settings")
    cache: CacheSettings = Field(default_factory=CacheSettings, description="Cache settings")
    logging: LoggingSettings = Field(default_factory=LoggingSettings, description="Logging settings")
    ui: UISettings = Field(default_factory=UISettings, description="UI settings")
    components: ComponentSettings = Field(default_factory=ComponentSettings, description="Component integration settings")
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings, description="WebSocket settings")
    
    class Config:
        """Pydantic config."""
        env_prefix = "DASHBOARD_"
        env_nested_delimiter = "__"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Environment-specific overrides
if settings.environment == "production":
    settings.debug = False
    settings.server.debug = False
    settings.server.reload = False
    settings.logging.level = "WARNING"
    settings.database.echo = False
elif settings.environment == "staging":
    settings.debug = False
    settings.server.debug = False
    settings.server.reload = False
    settings.logging.level = "INFO"
    settings.database.echo = False
elif settings.environment == "development":
    settings.debug = True
    settings.server.debug = True
    settings.server.reload = True
    settings.logging.level = "DEBUG"
    settings.database.echo = True
elif settings.environment == "testing":
    settings.debug = True
    settings.testing = True
    settings.server.debug = True
    settings.logging.level = "DEBUG"
    settings.database.echo = True
    settings.database.url = "sqlite:///:memory:"
