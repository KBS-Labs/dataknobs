#!/usr/bin/env python3
"""
Database ETL Pipeline Example

This example demonstrates how to build a production-ready ETL pipeline
that extracts data from a source database, transforms it, and loads it
into a target database.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from dataknobs_fsm import SimpleFSM
from dataknobs_fsm.patterns import ETLPattern

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SalesETLPipeline:
    """Production-ready ETL pipeline for sales data."""
    
    def __init__(self, source_config: Dict[str, Any], target_config: Dict[str, Any]):
        """Initialize the ETL pipeline.
        
        Args:
            source_config: Source database configuration
            target_config: Target database configuration
        """
        self.source_config = source_config
        self.target_config = target_config
        self.etl = None
        self.stats = {
            "start_time": None,
            "end_time": None,
            "records_extracted": 0,
            "records_transformed": 0,
            "records_loaded": 0,
            "errors": []
        }
    
    def build_pipeline(self) -> ETLPattern:
        """Build the ETL pipeline with all transformations."""
        
        # Create ETL pattern
        self.etl = ETLPattern(
            name="sales_etl",
            source_config={
                "type": "database",
                **self.source_config
            },
            target_config={
                "type": "database",
                **self.target_config
            },
            options={
                "batch_size": 5000,
                "parallel": True,
                "workers": 4,
                "checkpoint_interval": 10000,
                "error_threshold": 0.01,  # Allow 1% errors
                "on_error": "log_and_continue"
            }
        )
        
        # Add transformations
        self.etl.add_transformation(self.clean_data)
        self.etl.add_transformation(self.validate_data)
        self.etl.add_transformation(self.enrich_data)
        self.etl.add_transformation(self.calculate_metrics)
        
        return self.etl
    
    def clean_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize data.
        
        Args:
            row: Input data row
            
        Returns:
            Cleaned data row
        """
        # Remove whitespace
        for key, value in row.items():
            if isinstance(value, str):
                row[key] = value.strip()
        
        # Standardize date formats
        if 'order_date' in row and row['order_date']:
            if isinstance(row['order_date'], str):
                row['order_date'] = datetime.fromisoformat(row['order_date'])
        
        # Handle missing values
        row['customer_segment'] = row.get('customer_segment', 'Unknown')
        row['region'] = row.get('region', 'Unknown')
        
        return row
    
    def validate_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality.
        
        Args:
            row: Input data row
            
        Returns:
            Validated data row
            
        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        required_fields = ['order_id', 'customer_id', 'order_date', 'amount']
        for field in required_fields:
            if field not in row or row[field] is None:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types and ranges
        if not isinstance(row['order_id'], (int, str)):
            raise ValueError(f"Invalid order_id: {row['order_id']}")
        
        if row['amount'] < 0:
            raise ValueError(f"Invalid amount: {row['amount']}")
        
        # Validate business rules
        if row['order_date'] > datetime.now():
            raise ValueError(f"Future order date: {row['order_date']}")
        
        row['_validated'] = True
        return row
    
    def enrich_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with calculated fields.
        
        Args:
            row: Input data row
            
        Returns:
            Enriched data row
        """
        # Add time-based dimensions
        order_date = row['order_date']
        row['year'] = order_date.year
        row['quarter'] = f"Q{(order_date.month - 1) // 3 + 1}"
        row['month'] = order_date.month
        row['week'] = order_date.isocalendar()[1]
        row['day_of_week'] = order_date.strftime('%A')
        
        # Calculate derived metrics
        row['revenue'] = row['amount'] * 1.1  # Add tax
        row['discount_amount'] = row.get('discount_pct', 0) * row['amount'] / 100
        row['net_amount'] = row['amount'] - row['discount_amount']
        
        # Add processing metadata
        row['etl_timestamp'] = datetime.now()
        row['etl_batch_id'] = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return row
    
    def calculate_metrics(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business metrics.
        
        Args:
            row: Input data row
            
        Returns:
            Row with calculated metrics
        """
        # Customer lifetime value indicator
        if row.get('total_orders', 0) > 10:
            row['customer_value'] = 'High'
        elif row.get('total_orders', 0) > 5:
            row['customer_value'] = 'Medium'
        else:
            row['customer_value'] = 'Low'
        
        # Order size classification
        if row['amount'] > 1000:
            row['order_size'] = 'Large'
        elif row['amount'] > 100:
            row['order_size'] = 'Medium'
        else:
            row['order_size'] = 'Small'
        
        # Seasonality indicator
        month = row['month']
        if month in [11, 12]:
            row['season'] = 'Holiday'
        elif month in [6, 7, 8]:
            row['season'] = 'Summer'
        elif month in [3, 4, 5]:
            row['season'] = 'Spring'
        elif month in [9, 10]:
            row['season'] = 'Fall'
        else:
            row['season'] = 'Winter'
        
        return row
    
    async def run_async(self, start_date: datetime = None) -> Dict[str, Any]:
        """Run the ETL pipeline asynchronously.
        
        Args:
            start_date: Start date for incremental load
            
        Returns:
            Execution results
        """
        if not self.etl:
            self.build_pipeline()
        
        self.stats['start_time'] = datetime.now()
        logger.info(f"Starting ETL pipeline at {self.stats['start_time']}")
        
        # Set up query parameters
        if not start_date:
            start_date = datetime.now() - timedelta(days=1)
        
        input_data = {
            "source_query": f"""
                SELECT 
                    o.order_id,
                    o.customer_id,
                    o.order_date,
                    o.amount,
                    o.discount_pct,
                    c.customer_name,
                    c.customer_segment,
                    c.region,
                    c.total_orders
                FROM orders o
                LEFT JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_date >= '{start_date.isoformat()}'
                ORDER BY o.order_date
            """,
            "target_table": "fact_sales",
            "mode": "upsert",
            "key_columns": ["order_id"]
        }
        
        try:
            # Run ETL with progress monitoring
            result = await self.etl.run_async(
                input_data,
                on_progress=self.progress_callback
            )
            
            self.stats['end_time'] = datetime.now()
            self.stats['records_extracted'] = result.get('records_extracted', 0)
            self.stats['records_transformed'] = result.get('records_transformed', 0)
            self.stats['records_loaded'] = result.get('records_loaded', 0)
            
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            logger.info(f"ETL completed successfully in {duration:.2f} seconds")
            logger.info(f"Records processed: {self.stats['records_loaded']}")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            self.stats['errors'].append(str(e))
            raise
    
    def progress_callback(self, progress: Dict[str, Any]):
        """Handle progress updates.
        
        Args:
            progress: Progress information
        """
        logger.info(
            f"Progress: {progress['processed']}/{progress['total']} records "
            f"({progress['percent']:.1f}%) - "
            f"Rate: {progress['records_per_second']:.0f} rec/s"
        )
        
        if progress.get('errors'):
            logger.warning(f"Errors encountered: {progress['errors']}")


def create_simple_etl_fsm() -> SimpleFSM:
    """Create a simple ETL FSM for demonstration."""
    
    fsm = SimpleFSM(name="simple_etl_workflow")
    
    # Define states
    fsm.add_state("start", initial=True)
    fsm.add_state("extract")
    fsm.add_state("transform")
    fsm.add_state("load")
    fsm.add_state("complete", terminal=True)
    fsm.add_state("error", terminal=True)
    
    # Add database resource
    fsm.add_resource("source_db", {
        "type": "database",
        "provider": "sqlite",
        "config": {"database": "source.db"}
    })
    
    fsm.add_resource("target_db", {
        "type": "database",
        "provider": "sqlite",
        "config": {"database": "warehouse.db"}
    })
    
    # Define transitions
    async def extract_data(data, resources):
        """Extract data from source."""
        source_db = resources["source_db"]
        async with source_db.connect() as conn:
            result = await conn.execute("SELECT * FROM sales")
            rows = await result.fetchall()
            return {**data, "raw_data": rows}
    
    def transform_data(data):
        """Transform extracted data."""
        transformed = []
        for row in data["raw_data"]:
            transformed.append({
                "id": row["id"],
                "amount": row["amount"] * 1.1,  # Add tax
                "date": datetime.now().isoformat()
            })
        return {**data, "transformed_data": transformed}
    
    async def load_data(data, resources):
        """Load data to target."""
        target_db = resources["target_db"]
        async with target_db.connect() as conn:
            for row in data["transformed_data"]:
                await conn.execute(
                    "INSERT INTO fact_sales (id, amount, date) VALUES (?, ?, ?)",
                    [row["id"], row["amount"], row["date"]]
                )
            await conn.commit()
        return {**data, "loaded": len(data["transformed_data"])}
    
    # Add transitions with error handling
    fsm.add_transition("start", "extract")
    fsm.add_transition("extract", "transform", 
                      function=extract_data, 
                      on_error="error")
    fsm.add_transition("transform", "load", 
                      function=transform_data,
                      on_error="error")
    fsm.add_transition("load", "complete", 
                      function=load_data,
                      on_error="error")
    
    return fsm


async def main():
    """Main execution function."""
    
    # Example 1: Using the ETL Pattern
    print("=" * 60)
    print("Example 1: ETL Pattern Pipeline")
    print("=" * 60)
    
    pipeline = SalesETLPipeline(
        source_config={
            "provider": "sqlite",
            "config": {"database": "sales.db"}
        },
        target_config={
            "provider": "sqlite",
            "config": {"database": "warehouse.db"}
        }
    )
    
    try:
        results = await pipeline.run_async(
            start_date=datetime.now() - timedelta(days=30)
        )
        
        print(f"\nETL Results:")
        print(f"- Duration: {(results['end_time'] - results['start_time']).total_seconds():.2f}s")
        print(f"- Records Extracted: {results['records_extracted']}")
        print(f"- Records Transformed: {results['records_transformed']}")
        print(f"- Records Loaded: {results['records_loaded']}")
        print(f"- Errors: {len(results['errors'])}")
        
    except Exception as e:
        print(f"ETL failed: {e}")
    
    # Example 2: Using SimpleFSM
    print("\n" + "=" * 60)
    print("Example 2: Simple FSM ETL Workflow")
    print("=" * 60)
    
    fsm = create_simple_etl_fsm()
    
    # Validate FSM
    errors = fsm.validate()
    if errors:
        print(f"Validation errors: {errors}")
        return
    
    # Run FSM
    try:
        result = await fsm.run_async({"batch_id": "batch_001"})
        print(f"\nFSM Results:")
        print(f"- Batch ID: {result['batch_id']}")
        print(f"- Records Loaded: {result.get('loaded', 0)}")
        
    except Exception as e:
        print(f"FSM execution failed: {e}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())