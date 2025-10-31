"""Benchmarking suite for DataKnobs LLM package.

This package provides comprehensive benchmarking tools for:
- Template rendering performance
- RAG search operations
- Conversation management
- End-to-end prompt building
"""

from .benchmark_result import BenchmarkResult
from .prompt_benchmark import PromptBenchmark
from .rag_benchmark import RAGBenchmark

# Conversation benchmarks are optional (require dataknobs-common)
try:
    from .conversation_benchmark import ConversationBenchmark
    CONVERSATION_BENCHMARKS_AVAILABLE = True
except ImportError:
    ConversationBenchmark = None
    CONVERSATION_BENCHMARKS_AVAILABLE = False

__all__ = [
    "BenchmarkResult",
    "PromptBenchmark",
    "RAGBenchmark",
]

if CONVERSATION_BENCHMARKS_AVAILABLE:
    __all__.append("ConversationBenchmark")
