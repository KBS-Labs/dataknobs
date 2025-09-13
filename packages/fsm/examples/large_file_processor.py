#!/usr/bin/env python3
"""
Large file processor example using FSM with REFERENCE mode and streaming.

This example demonstrates:
1. REFERENCE mode for handling large files without loading into memory
2. Streaming processing for chunks of data
3. Parallel processing of file chunks
4. Progress tracking and statistics collection
5. Error handling for partial failures

The example processes a large CSV/JSONL file by:
- Splitting it into chunks
- Processing each chunk in parallel
- Aggregating results
- Handling errors gracefully
"""

import json
import csv
import io
from pathlib import Path
from typing import Dict, Any, Iterator, List
import hashlib
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode


def initialize_processing(state) -> Dict[str, Any]:
    """Initialize the file processing context."""
    data = state.data.copy()
    
    # Get file reference
    file_ref = data.get('file_reference')
    if not file_ref:
        # Mark initialization as failed
        data['initialization_failed'] = True
        data['error_message'] = "No file reference provided"
        data['processing'] = {
            'total_lines': 0,
            'processed_lines': 0,
            'failed_lines': 0,
            'chunks_processed': 0,
            'errors': ["No file reference provided"],
            'statistics': {
                'min_value': None,
                'max_value': None,
                'sum': 0,
                'count': 0
            }
        }
        return data
    
    # Initialize processing metadata
    data['processing'] = {
        'total_lines': 0,
        'processed_lines': 0,
        'failed_lines': 0,
        'chunks_processed': 0,
        'errors': [],
        'statistics': {
            'min_value': None,
            'max_value': None,
            'sum': 0,
            'count': 0
        }
    }
    
    # Determine file type from extension
    file_path = Path(file_ref)
    if not file_path.exists():
        data['initialization_failed'] = True
        data['error_message'] = f"File not found: {file_ref}"
        return data
    
    if file_path.suffix.lower() == '.csv':
        data['file_type'] = 'csv'
    elif file_path.suffix.lower() in ['.json', '.jsonl']:
        data['file_type'] = 'jsonl'
    else:
        data['file_type'] = 'text'
    
    data['file_path'] = str(file_path)
    data['file_name'] = file_path.name
    data['file_size'] = file_path.stat().st_size
    data['initialization_failed'] = False
    
    return data


def process_chunk(state) -> Dict[str, Any]:
    """Process a chunk of the file."""
    data = state.data.copy()
    
    # In REFERENCE mode, we get a reference to the chunk
    chunk_data = data.get('chunk_data', [])
    file_type = data.get('file_type', 'text')
    
    chunk_stats = {
        'lines_processed': 0,
        'lines_failed': 0,
        'chunk_errors': []
    }
    
    # Process based on file type
    if file_type == 'jsonl':
        for line_num, line in enumerate(chunk_data):
            try:
                record = json.loads(line) if isinstance(line, str) else line
                # Process the record (example: extract numeric value)
                value = record.get('value', 0) if isinstance(record, dict) else 0
                update_statistics(data['processing']['statistics'], value)
                chunk_stats['lines_processed'] += 1
            except Exception as e:
                chunk_stats['lines_failed'] += 1
                chunk_stats['chunk_errors'].append(f"Line {line_num}: {str(e)}")
    
    elif file_type == 'csv':
        for row_num, row in enumerate(chunk_data):
            try:
                # Process CSV row (example: extract numeric value from first column)
                if row and len(row) > 0:
                    value = float(row[0]) if row[0] else 0
                    update_statistics(data['processing']['statistics'], value)
                    chunk_stats['lines_processed'] += 1
            except Exception as e:
                chunk_stats['lines_failed'] += 1
                chunk_stats['chunk_errors'].append(f"Row {row_num}: {str(e)}")
    
    else:  # text file
        for line_num, line in enumerate(chunk_data):
            try:
                # Process text line (example: count characters)
                line_length = len(line.strip())
                update_statistics(data['processing']['statistics'], line_length)
                chunk_stats['lines_processed'] += 1
            except Exception as e:
                chunk_stats['lines_failed'] += 1
                chunk_stats['chunk_errors'].append(f"Line {line_num}: {str(e)}")
    
    # Update global statistics
    data['processing']['processed_lines'] += chunk_stats['lines_processed']
    data['processing']['failed_lines'] += chunk_stats['lines_failed']
    data['processing']['chunks_processed'] += 1
    
    if chunk_stats['chunk_errors']:
        data['processing']['errors'].extend(chunk_stats['chunk_errors'][:10])  # Keep first 10 errors
    
    data['last_chunk_stats'] = chunk_stats
    
    return data


def update_statistics(stats: Dict, value: float):
    """Update running statistics."""
    if value is not None:
        stats['count'] += 1
        stats['sum'] += value
        
        if stats['min_value'] is None or value < stats['min_value']:
            stats['min_value'] = value
        
        if stats['max_value'] is None or value > stats['max_value']:
            stats['max_value'] = value


def aggregate_results(state) -> Dict[str, Any]:
    """Aggregate all processing results."""
    data = state.data.copy()
    
    processing = data['processing']
    stats = processing['statistics']
    
    # Calculate final statistics
    if stats['count'] > 0:
        stats['average'] = stats['sum'] / stats['count']
    else:
        stats['average'] = 0
    
    # Calculate success rate
    total_lines = processing['processed_lines'] + processing['failed_lines']
    if total_lines > 0:
        processing['success_rate'] = processing['processed_lines'] / total_lines
    else:
        processing['success_rate'] = 0
    
    # Generate summary
    data['summary'] = {
        'file_name': data['file_name'],
        'file_size': data['file_size'],
        'file_type': data['file_type'],
        'total_lines': total_lines,
        'processed_lines': processing['processed_lines'],
        'failed_lines': processing['failed_lines'],
        'chunks_processed': processing['chunks_processed'],
        'success_rate': f"{processing['success_rate']:.2%}",
        'statistics': {
            'min': stats['min_value'],
            'max': stats['max_value'],
            'avg': stats['average'],
            'sum': stats['sum'],
            'count': stats['count']
        },
        'errors_sample': processing['errors'][:5]  # First 5 errors
    }
    
    # Generate file hash for verification
    if Path(data['file_path']).exists():
        data['file_hash'] = calculate_file_hash(data['file_path'])
    
    return data


def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def check_initialization_success(data: Dict[str, Any], context: Any) -> bool:
    """Check if initialization was successful."""
    return not data.get('initialization_failed', False)


def check_processing_complete(data: Dict[str, Any], context: Any) -> bool:
    """Check if all chunks have been processed."""
    # In a real implementation, this would check if all chunks are done
    # For this example, we'll consider it complete after processing
    return data.get('processing', {}).get('chunks_processed', 0) > 0


def mark_success(state) -> Dict[str, Any]:
    """Mark processing as successful."""
    data = state.data.copy()
    data['status'] = 'SUCCESS'
    data['message'] = f"Successfully processed {data['file_name']}"
    return data


def mark_failure(state) -> Dict[str, Any]:
    """Mark processing as failed."""
    data = state.data.copy()
    data['status'] = 'FAILED'
    data['message'] = f"Failed to process {data.get('file_name', 'unknown file')}"
    return data


# FSM configuration for large file processing
config = {
    "name": "LargeFileProcessor",
    "main_network": "main",
    "data_mode": {
        "default": "reference"  # Use REFERENCE mode for large files
    },
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "start",
                "is_start": True
            },
            {
                "name": "initialize",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "initialize_processing"
                    }
                }
            },
            {
                "name": "process_chunks",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "process_chunk"
                    }
                },
                "streaming": {
                    "enabled": True,
                    "chunk_size": 1000,
                    "parallel": True,
                    "max_workers": 4
                }
            },
            {
                "name": "aggregate",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "aggregate_results"
                    }
                }
            },
            {
                "name": "success",
                "is_end": True,
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "mark_success"
                    }
                }
            },
            {
                "name": "failure",
                "is_end": True,
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "mark_failure"
                    }
                }
            }
        ],
        "arcs": [
            {"from": "start", "to": "initialize"},
            {
                "from": "initialize",
                "to": "process_chunks",
                "condition": {
                    "type": "registered",
                    "name": "check_initialization_success"
                }
            },
            {
                "from": "initialize",
                "to": "failure",
                "condition": {
                    "type": "inline",
                    "code": "data.get('initialization_failed', False)"
                }
            },
            {"from": "process_chunks", "to": "aggregate"},
            {
                "from": "aggregate",
                "to": "success",
                "condition": {
                    "type": "registered",
                    "name": "check_processing_complete"
                }
            },
            {
                "from": "aggregate",
                "to": "failure",
                "condition": {
                    "type": "inline",
                    "code": "not check_processing_complete(data, context)"
                }
            }
        ]
    }]
}


def create_sample_file(file_path: Path, file_type: str, num_lines: int = 10000):
    """Create a sample file for testing."""
    if file_type == 'jsonl':
        with open(file_path, 'w') as f:
            for i in range(num_lines):
                record = {
                    'id': i,
                    'value': i * 1.5,
                    'name': f'record_{i}',
                    'category': f'cat_{i % 10}'
                }
                f.write(json.dumps(record) + '\n')
    
    elif file_type == 'csv':
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['value', 'name', 'category'])  # Header
            for i in range(num_lines):
                writer.writerow([i * 1.5, f'record_{i}', f'cat_{i % 10}'])
    
    else:  # text
        with open(file_path, 'w') as f:
            for i in range(num_lines):
                f.write(f"Line {i}: This is a sample text line with some content.\n")


def simulate_chunked_reading(file_path: Path, file_type: str, chunk_size: int = 1000):
    """Simulate reading file in chunks (for demo purposes)."""
    chunks = []
    
    if file_type == 'jsonl':
        with open(file_path, 'r') as f:
            chunk = []
            for line in f:
                chunk.append(line.strip())
                if len(chunk) >= chunk_size:
                    chunks.append(chunk)
                    chunk = []
            if chunk:
                chunks.append(chunk)
    
    elif file_type == 'csv':
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            chunk = []
            for row in reader:
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    chunks.append(chunk)
                    chunk = []
            if chunk:
                chunks.append(chunk)
    
    else:  # text
        with open(file_path, 'r') as f:
            chunk = []
            for line in f:
                chunk.append(line.strip())
                if len(chunk) >= chunk_size:
                    chunks.append(chunk)
                    chunk = []
            if chunk:
                chunks.append(chunk)
    
    return chunks


def main():
    """Run the large file processor example."""
    import tempfile
    
    # Create FSM with custom functions
    fsm = SimpleFSM(
        config,
        data_mode=DataHandlingMode.REFERENCE,  # Use REFERENCE mode
        custom_functions={
            'initialize_processing': initialize_processing,
            'process_chunk': process_chunk,
            'aggregate_results': aggregate_results,
            'check_initialization_success': check_initialization_success,
            'check_processing_complete': check_processing_complete,
            'mark_success': mark_success,
            'mark_failure': mark_failure
        }
    )
    
    print("Large File Processor Example")
    print("=" * 70)
    
    # Test with different file types
    file_types = ['jsonl', 'csv', 'text']
    
    for file_type in file_types:
        print(f"\nProcessing {file_type.upper()} file")
        print("-" * 50)
        
        # Create temporary sample file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'.{file_type if file_type != "text" else "txt"}',
            delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Create sample file
            create_sample_file(tmp_path, file_type, num_lines=5000)
            print(f"Created sample file: {tmp_path.name}")
            print(f"File size: {tmp_path.stat().st_size:,} bytes")
            
            # Simulate chunked processing
            chunks = simulate_chunked_reading(tmp_path, file_type, chunk_size=1000)
            print(f"Split into {len(chunks)} chunks")
            
            # Process the file
            # Note: In a real implementation, chunks would be streamed
            # For this demo, we'll process the first chunk
            result = fsm.process({
                'file_reference': str(tmp_path),
                'chunk_data': chunks[0] if chunks else []
            })
            
            if result['success']:
                print(f"✓ Processing succeeded")
                print(f"Final State: {result['final_state']}")
                
                summary = result['data'].get('summary', {})
                if summary:
                    print("\nProcessing Summary:")
                    print(f"  • File: {summary['file_name']}")
                    print(f"  • Type: {summary['file_type']}")
                    print(f"  • Size: {summary['file_size']:,} bytes")
                    print(f"  • Lines Processed: {summary['processed_lines']}")
                    print(f"  • Lines Failed: {summary['failed_lines']}")
                    print(f"  • Success Rate: {summary['success_rate']}")
                    
                    if summary.get('statistics'):
                        stats = summary['statistics']
                        print("\nStatistics:")
                        print(f"  • Min: {stats['min']}")
                        print(f"  • Max: {stats['max']}")
                        print(f"  • Avg: {stats['avg']:.2f}")
                        print(f"  • Sum: {stats['sum']:.2f}")
                        print(f"  • Count: {stats['count']}")
                
                if result['data'].get('file_hash'):
                    print(f"\nFile Hash: {result['data']['file_hash'][:16]}...")
            else:
                print(f"✗ Processing failed")
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()
                print(f"Cleaned up temporary file")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    
    # Demonstrate REFERENCE mode benefits
    print("\n📊 REFERENCE Mode Benefits:")
    print("  • Memory efficient - doesn't load entire file")
    print("  • Supports streaming processing")
    print("  • Can handle files larger than available RAM")
    print("  • Enables parallel chunk processing")
    print("  • Maintains file references instead of copying data")


if __name__ == "__main__":
    main()