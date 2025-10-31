"""Benchmarks for conversation management operations."""

import time
import asyncio
from typing import List

try:
    from dataknobs_llm.conversations.manager import ConversationManager
    from dataknobs_llm.conversations.storage import DataknobsConversationStorage
    from dataknobs_llm.llm.providers import EchoProvider
    from dataknobs_llm.prompts.implementations.config_library import ConfigPromptLibrary
    from dataknobs_llm.prompts.builders.async_prompt_builder import AsyncPromptBuilder
    try:
        from dataknobs_common.databases.async_memory_database import AsyncMemoryDatabase
    except ImportError:
        # Fallback: use mock storage
        AsyncMemoryDatabase = None
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    ConversationManager = None
    DataknobsConversationStorage = None
    EchoProvider = None
    ConfigPromptLibrary = None
    AsyncPromptBuilder = None
    AsyncMemoryDatabase = None

try:
    from .benchmark_result import BenchmarkResult
except ImportError:
    from benchmark_result import BenchmarkResult


class ConversationBenchmark:
    """Benchmark conversation management operations.

    This class provides benchmarks for:
    - Adding messages to conversations
    - Creating conversation branches
    - Switching between nodes
    - Completing conversations
    - Middleware execution

    Note: Requires dataknobs-common dependency.
    """

    def __init__(self, iterations: int = 100):
        """Initialize conversation benchmark.

        Args:
            iterations: Number of iterations to run for each benchmark

        Raises:
            ImportError: If conversation dependencies are not available
        """
        if not CONVERSATION_AVAILABLE:
            raise ImportError(
                "Conversation benchmarks require dataknobs-common. "
                "Install it to run conversation benchmarks."
            )
        self.iterations = iterations

    async def _create_manager(self) -> ConversationManager:
        """Create a ConversationManager for benchmarking.

        Returns:
            Configured ConversationManager
        """
        # Simple config with test prompts
        config = {
            "system": {
                "test": {
                    "template": "You are a helpful assistant."
                }
            },
            "user": {
                "test": {
                    0: {"template": "User message: {{text}}"}
                }
            }
        }

        library = ConfigPromptLibrary(config)
        builder = AsyncPromptBuilder(library=library)
        llm = EchoProvider()

        # Create storage with AsyncMemoryDatabase backend
        backend = AsyncMemoryDatabase()
        storage = DataknobsConversationStorage(backend)

        manager = await ConversationManager.create(
            llm=llm,
            prompt_builder=builder,
            storage=storage,
            system_prompt_name="test"
        )

        return manager

    async def benchmark_add_message(self) -> BenchmarkResult:
        """Benchmark adding messages to a conversation."""
        manager = await self._create_manager()

        times = []
        for i in range(self.iterations):
            start = time.perf_counter()
            await manager.add_message(
                role="user",
                prompt_name="test",
                params={"text": f"Message {i}"}
            )
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Add Message", times)

    async def benchmark_branch_creation(self) -> BenchmarkResult:
        """Benchmark creating conversation branches."""
        manager = await self._create_manager()

        # Add some initial messages
        await manager.add_message("user", "test", {"text": "Hello"})
        await manager.complete()

        times = []
        for i in range(min(self.iterations, 50)):  # Limit branches to avoid memory issues
            start = time.perf_counter()
            await manager.complete(branch_name=f"branch_{i}")
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Branch Creation", times)

    async def benchmark_switch_node(self) -> BenchmarkResult:
        """Benchmark switching between conversation nodes."""
        manager = await self._create_manager()

        # Create a conversation with multiple nodes
        await manager.add_message("user", "test", {"text": "Message 1"})
        await manager.complete()
        node_1 = manager.current_node.id

        await manager.add_message("user", "test", {"text": "Message 2"})
        await manager.complete()
        node_2 = manager.current_node.id

        await manager.add_message("user", "test", {"text": "Message 3"})
        await manager.complete()
        node_3 = manager.current_node.id

        nodes = [node_1, node_2, node_3]

        times = []
        for i in range(self.iterations):
            node_id = nodes[i % len(nodes)]
            start = time.perf_counter()
            await manager.switch_to_node(node_id)
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Switch Node", times)

    async def benchmark_complete(self) -> BenchmarkResult:
        """Benchmark completing conversations (LLM call)."""
        times = []

        for _ in range(min(self.iterations, 50)):  # Limit to avoid too many managers
            manager = await self._create_manager()
            await manager.add_message("user", "test", {"text": "Hello"})

            start = time.perf_counter()
            await manager.complete()
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Complete (with LLM call)", times)

    async def benchmark_get_messages(self) -> BenchmarkResult:
        """Benchmark retrieving conversation messages."""
        manager = await self._create_manager()

        # Add 10 messages
        for i in range(10):
            await manager.add_message("user", "test", {"text": f"Message {i}"})
            await manager.complete()

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            messages = await manager.get_messages()
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult.from_times("Get Messages (10 msgs)", times)

    async def run_all_async(self) -> List[BenchmarkResult]:
        """Run all conversation benchmarks asynchronously.

        Returns:
            List of BenchmarkResult objects
        """
        print(f"Running conversation benchmarks ({self.iterations} iterations)...")
        print()

        results = []

        benchmarks = [
            ("Add Message", self.benchmark_add_message),
            ("Branch Creation", self.benchmark_branch_creation),
            ("Switch Node", self.benchmark_switch_node),
            ("Complete", self.benchmark_complete),
            ("Get Messages", self.benchmark_get_messages),
        ]

        for name, benchmark_func in benchmarks:
            print(f"Running {name}...")
            result = await benchmark_func()
            results.append(result)
            print(f"  {result.operations_per_second:.0f} ops/sec "
                  f"({result.mean_time * 1000:.3f}ms mean)")

        print()
        print("Conversation benchmarks complete!")
        return results

    def run_all(self) -> List[BenchmarkResult]:
        """Run all conversation benchmarks (synchronous wrapper).

        Returns:
            List of BenchmarkResult objects
        """
        return asyncio.run(self.run_all_async())
