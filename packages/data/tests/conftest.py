"""Pytest configuration for dataknobs_data tests."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add the package source to path for testing
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the event loop policy for the test session."""
    return asyncio.DefaultEventLoopPolicy()


def pytest_sessionfinish(session, exitstatus):
    """Cleanup connection pools at the end of the test session."""
    import asyncio
    
    async def cleanup():
        # Clean up any remaining pools
        try:
            from dataknobs_data.backends.elasticsearch_async import _client_manager as es_manager
            await es_manager.close_all()
        except Exception:
            pass  # Ignore cleanup errors
        
        try:
            from dataknobs_data.backends.s3_async import _session_manager as s3_manager
            await s3_manager.close_all()
        except Exception:
            pass  # Ignore cleanup errors
        
        try:
            from dataknobs_data.backends.postgres_native import _pool_manager as pg_manager
            await pg_manager.close_all()
        except Exception:
            pass  # Ignore cleanup errors
    
    # Run cleanup in a new event loop
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cleanup())
        loop.close()
    except Exception:
        pass  # Ignore cleanup errors
