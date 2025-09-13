#!/usr/bin/env python3
"""
Database ETL Pipeline Example using FSM with COPY mode.

This example demonstrates:
1. COPY mode for transaction safety and rollback capability
2. Multi-stage data extraction, transformation, and loading
3. Custom function registration for ETL operations
4. Error handling with proper routing to rollback states
5. Batch processing with configurable size
6. Data validation and quality checks
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode


def initialize_etl(state) -> Dict[str, Any]:
    """Initialize ETL pipeline with configuration and statistics."""
    data = state.data.copy()
    
    # Initialize ETL metadata
    data['etl_metadata'] = {
        'start_time': datetime.now().isoformat(),
        'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'source_table': data.get('source_table', 'sales_raw'),
        'target_table': data.get('target_table', 'sales_fact'),
        'batch_size': data.get('batch_size', 100),
        'mode': data.get('mode', 'incremental')
    }
    
    # Initialize statistics
    data['statistics'] = {
        'records_extracted': 0,
        'records_transformed': 0,
        'records_validated': 0,
        'records_loaded': 0,
        'records_failed': 0,
        'validation_errors': [],
        'transformation_errors': []
    }
    
    # Set up query parameters
    if data['etl_metadata']['mode'] == 'incremental':
        # For incremental, get last processed timestamp
        data['last_processed'] = data.get('last_processed', 
                                         (datetime.now() - timedelta(days=7)).isoformat())
    
    data['initialization_complete'] = True
    return data


def extract_data(state) -> Dict[str, Any]:
    """Extract data from source database."""
    data = state.data.copy()
    
    # In a real scenario, this would connect to a database
    # For demo, we'll simulate extraction
    source_table = data['etl_metadata']['source_table']
    batch_size = data['etl_metadata']['batch_size']
    
    # Simulate database extraction
    extracted_records = []
    for i in range(batch_size):
        record = {
            'id': i + 1,
            'order_id': f"ORD{random.randint(1000, 9999)}",
            'customer_id': f"CUST{random.randint(100, 999)}",
            'product_id': f"PROD{random.randint(10, 99)}",
            'quantity': random.randint(1, 10),
            'unit_price': round(random.uniform(10.0, 1000.0), 2),
            'order_date': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            'status': random.choice(['pending', 'completed', 'shipped', 'cancelled']),
            'region': random.choice(['North', 'South', 'East', 'West']),
            'raw_data': True  # Flag to indicate unprocessed data
        }
        extracted_records.append(record)
    
    data['extracted_records'] = extracted_records
    data['statistics']['records_extracted'] = len(extracted_records)
    
    # Log extraction
    data['extraction_log'] = {
        'timestamp': datetime.now().isoformat(),
        'source': source_table,
        'count': len(extracted_records),
        'status': 'success'
    }
    
    return data


def validate_records(state) -> Dict[str, Any]:
    """Validate extracted records for data quality."""
    data = state.data.copy()
    
    validated_records = []
    validation_errors = []
    
    for record in data.get('extracted_records', []):
        errors = []
        
        # Check required fields
        required_fields = ['order_id', 'customer_id', 'quantity', 'unit_price']
        for field in required_fields:
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate data types and ranges
        if 'quantity' in record and record['quantity'] <= 0:
            errors.append(f"Invalid quantity: {record['quantity']}")
        
        if 'unit_price' in record and record['unit_price'] <= 0:
            errors.append(f"Invalid unit_price: {record['unit_price']}")
        
        # Check for future dates
        if 'order_date' in record:
            order_date = datetime.fromisoformat(record['order_date'])
            if order_date > datetime.now():
                errors.append(f"Future order date: {record['order_date']}")
        
        if errors:
            validation_errors.append({
                'record_id': record.get('id'),
                'errors': errors
            })
            data['statistics']['records_failed'] += 1
        else:
            record['validated'] = True
            validated_records.append(record)
            data['statistics']['records_validated'] += 1
    
    data['validated_records'] = validated_records
    data['statistics']['validation_errors'] = validation_errors
    
    # Determine if validation passed based on error threshold
    error_rate = data['statistics']['records_failed'] / data['statistics']['records_extracted']
    data['validation_passed'] = error_rate <= 0.1  # Allow up to 10% errors
    
    return data


def transform_records(state) -> Dict[str, Any]:
    """Transform validated records with business logic."""
    data = state.data.copy()
    
    transformed_records = []
    
    for record in data.get('validated_records', []):
        try:
            # Calculate derived fields
            transformed = record.copy()
            
            # Calculate total amount
            transformed['total_amount'] = round(
                record['quantity'] * record['unit_price'], 2
            )
            
            # Apply discount based on quantity
            if record['quantity'] >= 5:
                transformed['discount_rate'] = 0.1
            elif record['quantity'] >= 3:
                transformed['discount_rate'] = 0.05
            else:
                transformed['discount_rate'] = 0.0
            
            transformed['discount_amount'] = round(
                transformed['total_amount'] * transformed['discount_rate'], 2
            )
            transformed['net_amount'] = round(
                transformed['total_amount'] - transformed['discount_amount'], 2
            )
            
            # Add time dimensions
            order_date = datetime.fromisoformat(record['order_date'])
            transformed['year'] = order_date.year
            transformed['month'] = order_date.month
            transformed['quarter'] = f"Q{(order_date.month - 1) // 3 + 1}"
            transformed['day_of_week'] = order_date.strftime('%A')
            
            # Customer segmentation
            if transformed['net_amount'] > 500:
                transformed['customer_segment'] = 'Premium'
            elif transformed['net_amount'] > 100:
                transformed['customer_segment'] = 'Standard'
            else:
                transformed['customer_segment'] = 'Basic'
            
            # Add ETL metadata
            transformed['etl_batch_id'] = data['etl_metadata']['batch_id']
            transformed['etl_timestamp'] = datetime.now().isoformat()
            transformed['transformed'] = True
            
            # Remove raw data flag
            transformed.pop('raw_data', None)
            
            transformed_records.append(transformed)
            data['statistics']['records_transformed'] += 1
            
        except Exception as e:
            data['statistics']['transformation_errors'].append({
                'record_id': record.get('id'),
                'error': str(e)
            })
    
    data['transformed_records'] = transformed_records
    data['transformation_complete'] = True
    
    return data


def load_to_staging(state) -> Dict[str, Any]:
    """Load transformed records to staging area."""
    data = state.data.copy()
    
    # In COPY mode, this creates a staging area before final commit
    staging_records = []
    
    for record in data.get('transformed_records', []):
        # Prepare record for database insertion
        staging_record = {
            'id': record['id'],
            'order_id': record['order_id'],
            'customer_id': record['customer_id'],
            'product_id': record['product_id'],
            'quantity': record['quantity'],
            'unit_price': record['unit_price'],
            'total_amount': record['total_amount'],
            'discount_amount': record['discount_amount'],
            'net_amount': record['net_amount'],
            'order_date': record['order_date'],
            'status': record['status'],
            'region': record['region'],
            'customer_segment': record['customer_segment'],
            'year': record['year'],
            'month': record['month'],
            'quarter': record['quarter'],
            'etl_batch_id': record['etl_batch_id'],
            'etl_timestamp': record['etl_timestamp']
        }
        staging_records.append(staging_record)
    
    data['staging_records'] = staging_records
    data['staging_complete'] = True
    data['ready_to_commit'] = True
    
    # Log staging
    data['staging_log'] = {
        'timestamp': datetime.now().isoformat(),
        'record_count': len(staging_records),
        'status': 'ready'
    }
    
    return data


def commit_to_target(state) -> Dict[str, Any]:
    """Commit staged records to target database."""
    data = state.data.copy()
    
    # Simulate database commit
    # In COPY mode, this is where the actual database transaction would occur
    try:
        records_to_load = data.get('staging_records', [])
        
        # Simulate batch insert
        data['loaded_records'] = records_to_load
        data['statistics']['records_loaded'] = len(records_to_load)
        
        # Mark as committed
        data['committed'] = True
        data['commit_timestamp'] = datetime.now().isoformat()
        
        # Clear staging area after successful commit
        data['staging_records'] = []
        
        # Update last processed timestamp for next incremental run
        data['last_processed'] = datetime.now().isoformat()
        
    except Exception as e:
        data['commit_error'] = str(e)
        data['committed'] = False
        raise
    
    return data


def rollback_staging(state) -> Dict[str, Any]:
    """Rollback staging area on failure."""
    data = state.data.copy()
    
    # Clear staging area
    data['staging_records'] = []
    data['staging_complete'] = False
    data['ready_to_commit'] = False
    
    # Log rollback
    data['rollback_log'] = {
        'timestamp': datetime.now().isoformat(),
        'reason': data.get('commit_error', 'Validation failed'),
        'records_rolled_back': data['statistics'].get('records_transformed', 0)
    }
    
    data['rollback_complete'] = True
    
    return data


def finalize_etl(state) -> Dict[str, Any]:
    """Finalize ETL pipeline and generate summary."""
    data = state.data.copy()
    
    end_time = datetime.now()
    start_time = datetime.fromisoformat(data['etl_metadata']['start_time'])
    duration = (end_time - start_time).total_seconds()
    
    # Generate summary
    data['etl_summary'] = {
        'batch_id': data['etl_metadata']['batch_id'],
        'start_time': data['etl_metadata']['start_time'],
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'source_table': data['etl_metadata']['source_table'],
        'target_table': data['etl_metadata']['target_table'],
        'records_extracted': data['statistics']['records_extracted'],
        'records_validated': data['statistics']['records_validated'],
        'records_transformed': data['statistics']['records_transformed'],
        'records_loaded': data['statistics']['records_loaded'],
        'records_failed': data['statistics']['records_failed'],
        'success_rate': (data['statistics']['records_loaded'] / 
                        data['statistics']['records_extracted'] * 100 
                        if data['statistics']['records_extracted'] > 0 else 0),
        'status': 'SUCCESS' if data.get('committed', False) else 'FAILED'
    }
    
    return data


def check_validation_passed(data: Dict[str, Any], context: Any) -> bool:
    """Check if validation passed with acceptable error threshold."""
    return data.get('validation_passed', False)


def check_ready_to_commit(data: Dict[str, Any], context: Any) -> bool:
    """Check if staging is complete and ready to commit."""
    return data.get('ready_to_commit', False)


# FSM configuration for ETL pipeline
etl_config = {
    "name": "DatabaseETLPipeline",
    "main_network": "main",
    "data_mode": {
        "default": "copy"  # Use COPY mode for transaction safety
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
                        "name": "initialize_etl"
                    }
                }
            },
            {
                "name": "extract",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "extract_data"
                    }
                }
            },
            {
                "name": "validate",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "validate_records"
                    }
                }
            },
            {
                "name": "transform",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "transform_records"
                    }
                }
            },
            {
                "name": "staging",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "load_to_staging"
                    }
                }
            },
            {
                "name": "commit",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "commit_to_target"
                    }
                }
            },
            {
                "name": "rollback",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "rollback_staging"
                    }
                }
            },
            {
                "name": "finalize",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "finalize_etl"
                    }
                }
            },
            {
                "name": "success",
                "is_end": True
            },
            {
                "name": "failure",
                "is_end": True
            }
        ],
        "arcs": [
            {"from": "start", "to": "initialize"},
            {"from": "initialize", "to": "extract"},
            {"from": "extract", "to": "validate"},
            {
                "from": "validate",
                "to": "transform",
                "condition": {
                    "type": "registered",
                    "name": "check_validation_passed"
                }
            },
            {
                "from": "validate",
                "to": "rollback",
                "condition": {
                    "type": "inline",
                    "code": "not data.get('validation_passed', False)"
                }
            },
            {"from": "transform", "to": "staging"},
            {
                "from": "staging",
                "to": "commit",
                "condition": {
                    "type": "registered",
                    "name": "check_ready_to_commit"
                }
            },
            {
                "from": "staging",
                "to": "rollback",
                "condition": {
                    "type": "inline",
                    "code": "not data.get('ready_to_commit', False)"
                }
            },
            {"from": "commit", "to": "finalize"},
            {"from": "rollback", "to": "finalize"},
            {
                "from": "finalize",
                "to": "success",
                "condition": {
                    "type": "inline",
                    "code": "data.get('committed', False)"
                }
            },
            {
                "from": "finalize",
                "to": "failure",
                "condition": {
                    "type": "inline",
                    "code": "not data.get('committed', False)"
                }
            }
        ]
    }]
}


def create_etl_fsm():
    """Create and configure ETL FSM."""
    return SimpleFSM(
        etl_config,
        data_mode=DataHandlingMode.COPY,  # Ensure COPY mode for transactions
        custom_functions={
            'initialize_etl': initialize_etl,
            'extract_data': extract_data,
            'validate_records': validate_records,
            'transform_records': transform_records,
            'load_to_staging': load_to_staging,
            'commit_to_target': commit_to_target,
            'rollback_staging': rollback_staging,
            'finalize_etl': finalize_etl,
            'check_validation_passed': check_validation_passed,
            'check_ready_to_commit': check_ready_to_commit
        }
    )


def simulate_database_operations():
    """Simulate database setup for the example."""
    # Create SQLite databases for demonstration
    source_db = sqlite3.connect(':memory:')
    target_db = sqlite3.connect(':memory:')
    
    # Create source table
    source_db.execute('''
        CREATE TABLE sales_raw (
            id INTEGER PRIMARY KEY,
            order_id TEXT,
            customer_id TEXT,
            product_id TEXT,
            quantity INTEGER,
            unit_price REAL,
            order_date TEXT,
            status TEXT,
            region TEXT
        )
    ''')
    
    # Insert sample data
    sample_data = []
    for i in range(50):
        sample_data.append((
            i + 1,
            f"ORD{1000 + i}",
            f"CUST{100 + (i % 20)}",
            f"PROD{10 + (i % 10)}",
            random.randint(1, 10),
            round(random.uniform(10.0, 500.0), 2),
            (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            random.choice(['pending', 'completed', 'shipped']),
            random.choice(['North', 'South', 'East', 'West'])
        ))
    
    source_db.executemany(
        'INSERT INTO sales_raw VALUES (?,?,?,?,?,?,?,?,?)',
        sample_data
    )
    source_db.commit()
    
    # Create target table
    target_db.execute('''
        CREATE TABLE sales_fact (
            id INTEGER PRIMARY KEY,
            order_id TEXT,
            customer_id TEXT,
            product_id TEXT,
            quantity INTEGER,
            unit_price REAL,
            total_amount REAL,
            discount_amount REAL,
            net_amount REAL,
            order_date TEXT,
            status TEXT,
            region TEXT,
            customer_segment TEXT,
            year INTEGER,
            month INTEGER,
            quarter TEXT,
            etl_batch_id TEXT,
            etl_timestamp TEXT
        )
    ''')
    target_db.commit()
    
    return source_db, target_db


def main():
    """Run the ETL pipeline example."""
    print("Database ETL Pipeline Example")
    print("=" * 70)
    print("Demonstrating:")
    print("- COPY mode for transaction safety")
    print("- Multi-stage ETL processing")
    print("- Data validation with error thresholds")
    print("- Staging and commit/rollback pattern")
    print("- Custom function registration")
    print("-" * 70)
    
    # Create ETL FSM
    fsm = create_etl_fsm()
    
    # Simulate databases (in production, these would be real connections)
    source_db, target_db = simulate_database_operations()
    
    # Test Case 1: Successful ETL
    print("\n📊 Test Case 1: Successful ETL Pipeline")
    print("-" * 50)
    
    result = fsm.process({
        'source_table': 'sales_raw',
        'target_table': 'sales_fact',
        'batch_size': 20,
        'mode': 'incremental'
    })
    
    if result['success']:
        print(f"✓ ETL Pipeline completed successfully")
        print(f"Final State: {result['final_state']}")
        print(f"Path: {' -> '.join(result['path'])}")
        
        summary = result['data'].get('etl_summary', {})
        print(f"\nETL Summary:")
        print(f"  • Batch ID: {summary.get('batch_id')}")
        print(f"  • Duration: {summary.get('duration_seconds', 0):.2f} seconds")
        print(f"  • Records Extracted: {summary.get('records_extracted', 0)}")
        print(f"  • Records Validated: {summary.get('records_validated', 0)}")
        print(f"  • Records Transformed: {summary.get('records_transformed', 0)}")
        print(f"  • Records Loaded: {summary.get('records_loaded', 0)}")
        print(f"  • Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"  • Status: {summary.get('status')}")
    else:
        print(f"✗ ETL Pipeline failed")
        print(f"Error: {result.get('error')}")
    
    # Test Case 2: ETL with validation failures (high error rate)
    print("\n📊 Test Case 2: ETL with Validation Failures")
    print("-" * 50)
    
    # Modify validate function to simulate failures
    def validate_with_failures(state):
        data = state.data.copy()
        # Force high error rate
        data['statistics']['records_extracted'] = 100
        data['statistics']['records_failed'] = 20  # 20% error rate
        data['statistics']['records_validated'] = 80
        data['validated_records'] = data.get('extracted_records', [])[:80]
        data['validation_passed'] = False  # Exceeds 10% threshold
        return data
    
    fsm_with_failures = SimpleFSM(
        etl_config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={
            'initialize_etl': initialize_etl,
            'extract_data': extract_data,
            'validate_records': validate_with_failures,  # Modified validator
            'transform_records': transform_records,
            'load_to_staging': load_to_staging,
            'commit_to_target': commit_to_target,
            'rollback_staging': rollback_staging,
            'finalize_etl': finalize_etl,
            'check_validation_passed': check_validation_passed,
            'check_ready_to_commit': check_ready_to_commit
        }
    )
    
    result = fsm_with_failures.process({
        'source_table': 'sales_raw',
        'target_table': 'sales_fact',
        'batch_size': 100
    })
    
    print(f"Final State: {result['final_state']}")
    print(f"Path: {' -> '.join(result['path'])}")
    
    if result['final_state'] == 'failure':
        print(f"✓ Correctly routed to failure after validation issues")
        rollback_log = result['data'].get('rollback_log', {})
        if rollback_log:
            print(f"\nRollback Log:")
            print(f"  • Timestamp: {rollback_log.get('timestamp')}")
            print(f"  • Reason: {rollback_log.get('reason')}")
            print(f"  • Records Rolled Back: {rollback_log.get('records_rolled_back', 0)}")
    
    # Test Case 3: Batch Processing
    print("\n📊 Test Case 3: Batch Processing Multiple Chunks")
    print("-" * 50)
    
    batches_processed = []
    for batch_num in range(3):
        print(f"\nProcessing batch {batch_num + 1}...")
        result = fsm.process({
            'source_table': 'sales_raw',
            'target_table': 'sales_fact',
            'batch_size': 10,
            'batch_number': batch_num + 1
        })
        
        if result['success']:
            summary = result['data'].get('etl_summary', {})
            batches_processed.append({
                'batch': batch_num + 1,
                'loaded': summary.get('records_loaded', 0),
                'success_rate': summary.get('success_rate', 0)
            })
            print(f"  ✓ Batch {batch_num + 1}: {summary.get('records_loaded', 0)} records")
    
    print(f"\nTotal batches processed: {len(batches_processed)}")
    total_records = sum(b['loaded'] for b in batches_processed)
    print(f"Total records loaded: {total_records}")
    
    print("\n" + "=" * 70)
    print("ETL Pipeline Example Complete!")
    print("\n📌 Key Features Demonstrated:")
    print("  • COPY mode ensures transactional integrity")
    print("  • Staging area allows for rollback on failure")
    print("  • Validation with configurable error thresholds")
    print("  • Multi-stage transformation pipeline")
    print("  • Proper error handling and routing")
    print("  • Batch processing capabilities")


if __name__ == "__main__":
    main()