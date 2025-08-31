#!/usr/bin/env python3
"""
Test runner for validating all vector store example scripts.

This script runs all example tests and validates that the examples work correctly
with the current DataKnobs implementation.

Usage:
    python run_example_tests.py [--verbose] [--filter PATTERN]
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple
import time


class ExampleTestRunner:
    """Runs and validates example script tests."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.tests_dir = Path(__file__).parent
        self.results = []
    
    def run_pytest(self, test_file: str, args: List[str] = None) -> Tuple[bool, str, float]:
        """Run pytest on a specific test file."""
        cmd = ["uv", "run", "pytest", test_file]
        
        if args:
            cmd.extend(args)
        
        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Add coverage if available
        cmd.extend(["--tb=short"])
        
        print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.tests_dir.parent.parent  # Run from package root
            )
            elapsed = time.time() - start_time
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            return success, output, elapsed
            
        except Exception as e:
            elapsed = time.time() - start_time
            return False, str(e), elapsed
    
    def run_all_tests(self, filter_pattern: str = None) -> bool:
        """Run all example tests."""
        test_files = [
            "test_basic_vector_search.py",
            "test_text_to_vector_sync.py",
            "test_migrate_existing_data.py",
            "test_hybrid_search.py",
            "test_examples_integration.py"
        ]
        
        if filter_pattern:
            test_files = [f for f in test_files if filter_pattern in f]
        
        print("\n" + "="*60)
        print("Running Example Tests")
        print("="*60)
        
        all_passed = True
        
        for test_file in test_files:
            test_path = self.tests_dir / test_file
            
            if not test_path.exists():
                print(f"\n❌ Test file not found: {test_file}")
                all_passed = False
                continue
            
            print(f"\n📝 Testing: {test_file}")
            print("-" * 40)
            
            success, output, elapsed = self.run_pytest(str(test_path))
            
            if success:
                print(f"✅ PASSED ({elapsed:.2f}s)")
                if self.verbose:
                    print(output)
            else:
                print(f"❌ FAILED ({elapsed:.2f}s)")
                print(output)
                all_passed = False
            
            self.results.append({
                "test": test_file,
                "passed": success,
                "time": elapsed
            })
        
        return all_passed
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])
        total_time = sum(r["time"] for r in self.results)
        
        print(f"\nResults: {passed} passed, {failed} failed")
        print(f"Total time: {total_time:.2f}s")
        
        if failed > 0:
            print("\nFailed tests:")
            for result in self.results:
                if not result["passed"]:
                    print(f"  ❌ {result['test']}")
        
        print("\nTest breakdown:")
        for result in self.results:
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {result['test']:40} {result['time']:.2f}s")
    
    def validate_examples(self) -> bool:
        """Validate that example scripts can be imported and run."""
        print("\n" + "="*60)
        print("Validating Example Scripts")
        print("="*60)
        
        examples_dir = self.tests_dir.parent.parent / "examples"
        example_files = [
            "basic_vector_search.py",
            "text_to_vector_sync.py",
            "migrate_existing_data.py",
            "hybrid_search.py"
        ]
        
        all_valid = True
        
        for example_file in example_files:
            example_path = examples_dir / example_file
            
            if not example_path.exists():
                print(f"❌ Example not found: {example_file}")
                all_valid = False
                continue
            
            # Try to import the example
            try:
                # Check syntax
                compile(example_path.read_text(), str(example_path), 'exec')
                print(f"✅ Valid: {example_file}")
            except SyntaxError as e:
                print(f"❌ Syntax error in {example_file}: {e}")
                all_valid = False
            except Exception as e:
                print(f"❌ Error validating {example_file}: {e}")
                all_valid = False
        
        return all_valid


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for DataKnobs vector store examples"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--filter", "-f",
        type=str,
        help="Filter tests by pattern"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate examples without running tests"
    )
    
    args = parser.parse_args()
    
    runner = ExampleTestRunner(verbose=args.verbose)
    
    # Validate examples
    examples_valid = runner.validate_examples()
    
    if args.validate_only:
        sys.exit(0 if examples_valid else 1)
    
    # Run tests
    tests_passed = runner.run_all_tests(filter_pattern=args.filter)
    
    # Print summary
    runner.print_summary()
    
    # Exit with appropriate code
    if tests_passed and examples_valid:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()