#!/usr/bin/env python3
"""Helper script to check if services are available."""

import sys
import time
import os
import json
from typing import Dict, Any

def check_postgres(host: str = "postgres", port: int = 5432) -> bool:
    """Check if PostgreSQL is available."""
    # First try socket connection
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((host, port))
        s.close()
        # If we can connect to the port, assume PostgreSQL is ready
        # This is less accurate than using psycopg2 but works without dependencies
        return True
    except Exception:
        pass
    
    # Try psycopg2 if available
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=host,
            port=port,
            user="postgres",
            password="postgres",
            database="postgres",
            connect_timeout=3
        )
        conn.close()
        return True
    except ImportError:
        # psycopg2 not available, rely on socket check
        pass
    except Exception:
        return False
    
    return False

def check_elasticsearch(host: str = "elasticsearch", port: int = 9200) -> bool:
    """Check if Elasticsearch is available."""
    try:
        import requests
        response = requests.get(f"http://{host}:{port}/_cluster/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") in ["green", "yellow"]
    except Exception:
        pass
    
    # Fallback to urllib if requests is not available
    try:
        import urllib.request
        import urllib.error
        with urllib.request.urlopen(f"http://{host}:{port}/_cluster/health", timeout=3) as response:
            data = json.loads(response.read().decode())
            return data.get("status") in ["green", "yellow"]
    except Exception:
        return False

def check_localstack(host: str = "localstack", port: int = 4566) -> bool:
    """Check if LocalStack is available."""
    try:
        import requests
        response = requests.get(f"http://{host}:{port}/_localstack/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            services = data.get("services", {})
            return services.get("s3") == "available"
    except Exception:
        pass

    # Fallback to urllib if requests is not available
    try:
        import urllib.request
        import urllib.error
        with urllib.request.urlopen(f"http://{host}:{port}/_localstack/health", timeout=3) as response:
            data = json.loads(response.read().decode())
            services = data.get("services", {})
            return services.get("s3") == "available"
    except Exception:
        return False

def check_ollama(host: str = "localhost", port: int = 11434) -> bool:
    """Check if Ollama is available."""
    try:
        import requests
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        pass

    # Fallback to urllib if requests is not available
    try:
        import urllib.request
        import urllib.error
        with urllib.request.urlopen(f"http://{host}:{port}/api/tags", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: check-services.py <service> [host] [port]")
        sys.exit(1)
    
    service = sys.argv[1].lower()
    
    # Determine if we're in Docker
    in_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER")
    
    if service == "postgres":
        host = sys.argv[2] if len(sys.argv) > 2 else ("postgres" if in_docker else "localhost")
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5432
        sys.exit(0 if check_postgres(host, port) else 1)
    
    elif service == "elasticsearch":
        host = sys.argv[2] if len(sys.argv) > 2 else ("elasticsearch" if in_docker else "localhost")
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 9200
        sys.exit(0 if check_elasticsearch(host, port) else 1)
    
    elif service == "localstack":
        host = sys.argv[2] if len(sys.argv) > 2 else ("localstack" if in_docker else "localhost")
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 4566
        sys.exit(0 if check_localstack(host, port) else 1)

    elif service == "ollama":
        # Ollama always runs locally, not in Docker
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 11434
        sys.exit(0 if check_ollama(host, port) else 1)

    else:
        print(f"Unknown service: {service}")
        sys.exit(1)

if __name__ == "__main__":
    main()