"""Elasticsearch-specific connection pooling implementation."""

from dataclasses import dataclass
from typing import Optional, Any, List

from .base import BasePoolConfig


@dataclass
class ElasticsearchPoolConfig(BasePoolConfig):
    """Configuration for Elasticsearch connection pools."""
    hosts: List[str] = None
    index: str = "records"
    api_key: Optional[str] = None
    basic_auth: Optional[tuple] = None
    verify_certs: bool = True
    ca_certs: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    ssl_show_warn: bool = True
    
    def __post_init__(self):
        """Set default hosts if not provided."""
        if self.hosts is None:
            self.hosts = ["http://localhost:9200"]
    
    def to_connection_string(self) -> str:
        """Convert to connection string (not used for ES, but required by base)."""
        return ";".join(self.hosts)
    
    def to_hash_key(self) -> tuple:
        """Create a hashable key for this configuration."""
        return (tuple(self.hosts), self.index)
    
    @classmethod
    def from_dict(cls, config: dict) -> "ElasticsearchPoolConfig":
        """Create from configuration dictionary."""
        # Handle both old-style (host, port) and new-style (hosts) configuration
        if "hosts" in config:
            hosts = config["hosts"]
        elif "host" in config:
            host = config["host"]
            port = config.get("port", 9200)
            # Check if it already has a scheme
            if host.startswith("http://") or host.startswith("https://"):
                hosts = [f"{host}:{port}" if ":" not in host.split("://")[1] else host]
            else:
                hosts = [f"http://{host}:{port}"]
        else:
            hosts = ["http://localhost:9200"]
        
        return cls(
            hosts=hosts,
            index=config.get("index", "records"),
            api_key=config.get("api_key"),
            basic_auth=config.get("basic_auth"),
            verify_certs=config.get("verify_certs", True),
            ca_certs=config.get("ca_certs"),
            client_cert=config.get("client_cert"),
            client_key=config.get("client_key"),
            ssl_show_warn=config.get("ssl_show_warn", True)
        )


async def create_async_elasticsearch_client(config: ElasticsearchPoolConfig):
    """Create an async Elasticsearch client."""
    from elasticsearch import AsyncElasticsearch
    
    # Build client configuration
    client_config = {
        "hosts": config.hosts,
    }
    
    # Add authentication if provided
    if config.api_key:
        client_config["api_key"] = config.api_key
    elif config.basic_auth:
        client_config["basic_auth"] = config.basic_auth
    
    # Add SSL configuration
    if config.ca_certs:
        client_config["ca_certs"] = config.ca_certs
    if config.client_cert:
        client_config["client_cert"] = config.client_cert
    if config.client_key:
        client_config["client_key"] = config.client_key
    
    client_config["verify_certs"] = config.verify_certs
    client_config["ssl_show_warn"] = config.ssl_show_warn
    
    # Create and return the client
    return AsyncElasticsearch(**client_config)


async def validate_elasticsearch_client(client) -> None:
    """Validate an Elasticsearch client by pinging it."""
    if not await client.ping():
        raise ConnectionError("Failed to ping Elasticsearch")


async def close_elasticsearch_client(client) -> None:
    """Properly close an Elasticsearch client and its underlying connections."""
    if client:
        try:
            await client.close()
        except Exception:
            pass  # Ignore errors during cleanup