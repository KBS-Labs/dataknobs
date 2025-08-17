#!/usr/bin/env python3
"""
Performance benchmarks for validation and migration modules.

Compares performance between different operations and provides insights
on optimization opportunities.
"""

import time
import random
import string
import statistics
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

from dataknobs_data.records import Record
from dataknobs_data.validation import (
    Schema,
    Required,
    Range,
    Length,
    Pattern,
    Enum,
    Unique,
    ValidationContext,
)
from dataknobs_data.migration import (
    Migration,
    AddField,
    RemoveField,
    RenameField,
    TransformField,
    Transformer,
    Migrator,
)
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_data.query import Query


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    operations: int
    total_time: float
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    ops_per_second: float


class Benchmark:
    """Base class for benchmarks."""
    
    def __init__(self, name: str, operations: int = 1000):
        self.name = name
        self.operations = operations
        self.times: List[float] = []
    
    def run(self) -> BenchmarkResult:
        """Run the benchmark and return results."""
        print(f"\nRunning {self.name}...")
        
        # Warmup
        for _ in range(min(10, self.operations // 10)):
            self.setup()
            self.execute()
            self.teardown()
        
        # Actual benchmark
        self.times = []
        total_start = time.perf_counter()
        
        for _ in range(self.operations):
            self.setup()
            
            start = time.perf_counter()
            self.execute()
            end = time.perf_counter()
            
            self.times.append(end - start)
            self.teardown()
        
        total_end = time.perf_counter()
        total_time = total_end - total_start
        
        return BenchmarkResult(
            name=self.name,
            operations=self.operations,
            total_time=total_time,
            mean_time=statistics.mean(self.times),
            median_time=statistics.median(self.times),
            min_time=min(self.times),
            max_time=max(self.times),
            ops_per_second=self.operations / total_time
        )
    
    def setup(self):
        """Setup before each operation."""
        pass
    
    def execute(self):
        """Execute the operation to benchmark."""
        raise NotImplementedError
    
    def teardown(self):
        """Cleanup after each operation."""
        pass


class ValidationBenchmark(Benchmark):
    """Benchmark schema validation."""
    
    def __init__(self, complexity: str = "simple"):
        super().__init__(f"Schema Validation ({complexity})", 10000)
        self.complexity = complexity
        self.schema = None
        self.record = None
    
    def setup(self):
        if self.complexity == "simple":
            self.schema = Schema("SimpleSchema")
            self.schema.field("id", "INTEGER", required=True)
            self.schema.field("name", "STRING", required=True)
            
            self.record = Record({
                "id": random.randint(1, 1000),
                "name": f"item_{random.randint(1, 100)}"
            })
            
        elif self.complexity == "complex":
            self.schema = Schema("ComplexSchema", strict=True)
            self.schema.field("id", "INTEGER", required=True, constraints=[Range(min=1, max=10000)])
            self.schema.field("username", "STRING", required=True, constraints=[
                Length(min=3, max=20),
                Pattern(r"^[a-zA-Z0-9_]+$")
            ])
            self.schema.field("email", "STRING", constraints=[
                Pattern(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
            ])
            self.schema.field("age", "INTEGER", constraints=[Range(min=0, max=150)])
            self.schema.field("status", "STRING", constraints=[
                Enum(["active", "inactive", "pending", "suspended"])
            ])
            
            self.record = Record({
                "id": random.randint(1, 10000),
                "username": ''.join(random.choices(string.ascii_letters + string.digits, k=10)),
                "email": f"user{random.randint(1, 1000)}@example.com",
                "age": random.randint(18, 65),
                "status": random.choice(["active", "inactive", "pending", "suspended"])
            })
    
    def execute(self):
        result = self.schema.validate(self.record)
        return result.valid


class UniqueConstraintBenchmark(Benchmark):
    """Benchmark unique constraint validation."""
    
    def __init__(self):
        super().__init__("Unique Constraint", 5000)
        self.schema = None
        self.context = None
        self.records = []
    
    def setup(self):
        self.schema = Schema("UniqueSchema")
        self.schema.field("id", "INTEGER", required=True, constraints=[Unique("id")])
        self.schema.field("username", "STRING", constraints=[Unique("username")])
        
        self.context = ValidationContext()
        self.records = []
        
        # Create records with some duplicates
        for i in range(100):
            id_val = i if i < 80 else random.randint(0, 79)  # 20% duplicates
            self.records.append(Record({
                "id": id_val,
                "username": f"user_{id_val}"
            }))
    
    def execute(self):
        valid_count = 0
        for record in self.records:
            result = self.schema.validate(record, context=self.context)
            if result.valid:
                valid_count += 1
        return valid_count


class MigrationBenchmark(Benchmark):
    """Benchmark migration operations."""
    
    def __init__(self, operations_count: int = 5):
        super().__init__(f"Migration ({operations_count} operations)", 5000)
        self.operations_count = operations_count
        self.migration = None
        self.record = None
    
    def setup(self):
        self.migration = Migration("v1", "v2")
        
        # Add various operations
        for i in range(self.operations_count):
            if i % 4 == 0:
                self.migration.add(AddField(f"field_{i}", f"value_{i}"))
            elif i % 4 == 1:
                self.migration.add(RenameField(f"old_{i}", f"new_{i}"))
            elif i % 4 == 2:
                self.migration.add(RemoveField(f"remove_{i}"))
            else:
                self.migration.add(TransformField(f"transform_{i}", lambda x: x * 2))
        
        # Create record with all necessary fields
        fields = {}
        for i in range(self.operations_count):
            if i % 4 == 1:
                fields[f"old_{i}"] = f"value_{i}"
            elif i % 4 == 2:
                fields[f"remove_{i}"] = f"value_{i}"
            elif i % 4 == 3:
                fields[f"transform_{i}"] = i
        
        self.record = Record(fields)
    
    def execute(self):
        migrated = self.migration.apply(self.record)
        return migrated


class TransformerBenchmark(Benchmark):
    """Benchmark transformer operations."""
    
    def __init__(self):
        super().__init__("Transformer", 10000)
        self.transformer = None
        self.record = None
    
    def setup(self):
        self.transformer = Transformer()
        self.transformer.map("old_id", "id")
        self.transformer.map("price", "price", lambda x: x * 1.1)
        self.transformer.exclude("password", "internal_id")
        self.transformer.add("processed", True)
        self.transformer.add("timestamp", lambda r: time.time())
        
        self.record = Record({
            "old_id": random.randint(1, 1000),
            "price": random.uniform(10.0, 1000.0),
            "password": "secret",
            "internal_id": "internal",
            "name": f"product_{random.randint(1, 100)}"
        })
    
    def execute(self):
        transformed = self.transformer.transform(self.record)
        return transformed


class DatabaseMigrationBenchmark(Benchmark):
    """Benchmark database migration."""
    
    def __init__(self, record_count: int = 1000):
        super().__init__(f"Database Migration ({record_count} records)", 10)
        self.record_count = record_count
        self.source = None
        self.target = None
        self.transformer = None
    
    def setup(self):
        # Create source database with test data
        self.source = SyncMemoryDatabase()
        for i in range(self.record_count):
            record = Record({
                "id": i,
                "name": f"item_{i}",
                "value": random.uniform(1.0, 100.0),
                "category": random.choice(["A", "B", "C"])
            })
            self.source.create(record)
        
        # Create target database
        self.target = SyncMemoryDatabase()
        
        # Create transformer
        self.transformer = Transformer()
        self.transformer.map("value", "price")
        self.transformer.add("currency", "USD")
        self.transformer.add("migrated_at", lambda r: "2024-01-01")
    
    def execute(self):
        migrator = Migrator()
        progress = migrator.migrate(
            source=self.source,
            target=self.target,
            transform=self.transformer,
            batch_size=100
        )
        return progress.succeeded
    
    def teardown(self):
        if self.source:
            self.source.close()
        if self.target:
            self.target.close()


class BatchValidationBenchmark(Benchmark):
    """Benchmark batch validation."""
    
    def __init__(self):
        super().__init__("Batch Validation (100 records)", 100)
        self.schema = None
        self.records = []
    
    def setup(self):
        self.schema = Schema("BatchSchema")
        self.schema.field("id", "INTEGER", required=True)
        self.schema.field("value", "FLOAT", constraints=[Range(min=0, max=1000)])
        self.schema.field("status", "STRING", constraints=[Enum(["active", "inactive"])])
        
        self.records = []
        for i in range(100):
            self.records.append(Record({
                "id": i,
                "value": random.uniform(0, 1000),
                "status": random.choice(["active", "inactive"])
            }))
    
    def execute(self):
        results = self.schema.validate_many(self.records)
        valid_count = sum(1 for r in results if r.valid)
        return valid_count


def format_result(result: BenchmarkResult) -> str:
    """Format benchmark result for display."""
    return f"""
{result.name}
{'=' * len(result.name)}
Operations:     {result.operations:,}
Total time:     {result.total_time:.3f}s
Mean time:      {result.mean_time * 1000:.3f}ms
Median time:    {result.median_time * 1000:.3f}ms
Min time:       {result.min_time * 1000:.3f}ms
Max time:       {result.max_time * 1000:.3f}ms
Ops/second:     {result.ops_per_second:,.0f}
"""


def run_benchmarks():
    """Run all benchmarks and display results."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    benchmarks = [
        # Validation benchmarks
        ValidationBenchmark("simple"),
        ValidationBenchmark("complex"),
        UniqueConstraintBenchmark(),
        BatchValidationBenchmark(),
        
        # Migration benchmarks
        MigrationBenchmark(operations_count=3),
        MigrationBenchmark(operations_count=10),
        TransformerBenchmark(),
        
        # Database migration benchmarks
        DatabaseMigrationBenchmark(record_count=100),
        DatabaseMigrationBenchmark(record_count=1000),
    ]
    
    results = []
    for benchmark in benchmarks:
        try:
            result = benchmark.run()
            results.append(result)
            print(format_result(result))
        except Exception as e:
            print(f"Failed to run {benchmark.name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nTop 5 Fastest Operations (ops/sec):")
    sorted_results = sorted(results, key=lambda r: r.ops_per_second, reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result.name}: {result.ops_per_second:,.0f} ops/sec")
    
    print("\nTop 5 Slowest Operations (mean time):")
    sorted_results = sorted(results, key=lambda r: r.mean_time, reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result.name}: {result.mean_time * 1000:.3f}ms")


if __name__ == "__main__":
    run_benchmarks()