# Pandas Integration Tutorial

This tutorial demonstrates how to leverage pandas DataFrames with the DataKnobs Data package for powerful data manipulation, analysis, and bulk operations.

## Prerequisites

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataknobs_data.pandas import (
    DataFrameConverter, BatchOperations, ChunkedProcessor,
    ConversionOptions, BatchConfig
)
from dataknobs_data import Record, MemoryDatabase, Query, FieldType
from dataknobs_data.validation import Schema, Range
```

## Part 1: Basic DataFrame Conversion

### Records to DataFrame

Convert DataKnobs records to pandas DataFrames for analysis:

```python
# Create sample records
records = [
    Record(data={
        "id": 1,
        "name": "Alice",
        "age": 30,
        "department": "Engineering",
        "salary": 75000,
        "joined_date": "2020-01-15"
    }),
    Record(data={
        "id": 2,
        "name": "Bob",
        "age": 25,
        "department": "Marketing",
        "salary": 55000,
        "joined_date": "2021-03-20"
    }),
    Record(data={
        "id": 3,
        "name": "Charlie",
        "age": 35,
        "department": "Engineering",
        "salary": 85000,
        "joined_date": "2019-06-10"
    })
]

# Initialize converter
converter = DataFrameConverter()

# Convert to DataFrame
df = converter.records_to_dataframe(records)

print("DataFrame from records:")
print(df)
print(f"\nData types:")
print(df.dtypes)

# Basic DataFrame operations
print(f"\nAverage salary by department:")
print(df.groupby("department")["salary"].mean())
```

### DataFrame to Records

Convert DataFrames back to DataKnobs records:

```python
# Create a DataFrame
data = {
    "product_id": [101, 102, 103, 104],
    "product_name": ["Laptop", "Mouse", "Keyboard", "Monitor"],
    "price": [999.99, 29.99, 79.99, 299.99],
    "stock": [50, 200, 150, 75],
    "category": ["Electronics", "Accessories", "Accessories", "Electronics"]
}
df = pd.DataFrame(data)

# Convert to records
records = converter.dataframe_to_records(df)

# Verify conversion
for record in records[:2]:
    print(f"Record: {record.data}")
```

### Conversion Options

Customize how data is converted:

```python
# Advanced conversion options
options = ConversionOptions(
    include_metadata=True,        # Include record metadata
    flatten_nested=True,          # Flatten nested dictionaries
    preserve_index=True,          # Keep DataFrame index
    datetime_format="iso",        # ISO format for datetime
    null_handling="skip"          # Skip null values
)

# Convert with options
df_with_options = converter.records_to_dataframe(records, options)

# Convert back with index preservation
options = ConversionOptions(
    use_index_as_id=True,        # Use DataFrame index as record ID
    metadata_columns=["_metadata"], # Columns containing metadata
)

records_from_df = converter.dataframe_to_records(df, options)
```

## Part 2: Batch Database Operations

### Setting Up Database and Data

```python
# Initialize database and batch operations
db = MemoryDatabase()
batch_ops = BatchOperations(db)

# Create sample sales data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"]
regions = ["North", "South", "East", "West"]

sales_data = []
for _ in range(1000):
    sales_data.append({
        "date": np.random.choice(dates).strftime("%Y-%m-%d"),
        "product": np.random.choice(products),
        "region": np.random.choice(regions),
        "quantity": np.random.randint(1, 20),
        "price": np.random.uniform(10, 1000),
        "customer_id": np.random.randint(1000, 2000)
    })

sales_df = pd.DataFrame(sales_data)
```

### Bulk Insert from DataFrame

```python
# Configure batch insertion
config = BatchConfig(
    chunk_size=100,              # Process 100 records at a time
    parallel=True,               # Use parallel processing
    num_workers=4,               # Number of parallel workers
    error_handling="log",        # Log errors but continue
    validate=True,               # Validate before insert
    progress_callback=lambda current, total: print(f"Inserted {current}/{total} records", end="\r")
)

# Bulk insert DataFrame into database
result = batch_ops.bulk_insert_dataframe(sales_df, config)

print(f"\nBulk insert results:")
print(f"  Inserted: {result['inserted']}")
print(f"  Failed: {result['failed']}")
print(f"  Duration: {result['duration']:.2f} seconds")
print(f"  Rate: {result['rate']:.0f} records/second")
```

### Query as DataFrame

```python
# Query database and get results as DataFrame
# Get all laptop sales
laptop_df = batch_ops.query_as_dataframe(
    Query().filter("product", "=", "Laptop")
)

print(f"\nLaptop sales: {len(laptop_df)} records")
print(laptop_df.head())

# Complex query with multiple filters
high_value_df = batch_ops.query_as_dataframe(
    Query()
    .filter("price", ">", 500)
    .filter("quantity", ">=", 5)
    .sort("date", descending=True)
    .limit(50)
)

print(f"\nHigh-value sales: {len(high_value_df)} records")
print(high_value_df.describe())
```

## Part 3: Advanced Analytics

### Aggregations and Grouping

```python
# Perform aggregations directly on database data
agg_df = batch_ops.aggregate(
    Query(),
    aggregations={
        "quantity": ["sum", "mean", "std"],
        "price": ["mean", "min", "max"],
        "customer_id": "nunique"  # Count unique customers
    },
    group_by=["product", "region"]
)

print("\nSales aggregation by product and region:")
print(agg_df.head(10))

# Pivot table for better visualization
pivot_table = agg_df.pivot_table(
    values="quantity_sum",
    index="product",
    columns="region",
    fill_value=0
)

print("\nQuantity sold by product and region:")
print(pivot_table)
```

### Transform and Save

Apply complex transformations and save back to database:

```python
def analyze_and_enrich(df):
    """Add analytics fields to sales data"""
    # Calculate total revenue
    df["revenue"] = df["quantity"] * df["price"]
    
    # Add day of week
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    
    # Calculate moving average of price
    df = df.sort_values("date")
    df["price_ma7"] = df.groupby("product")["price"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Flag high-value transactions
    df["is_high_value"] = (df["revenue"] > df["revenue"].quantile(0.9))
    
    # Categorize quantity
    df["quantity_category"] = pd.cut(
        df["quantity"],
        bins=[0, 5, 10, 20, float('inf')],
        labels=["Low", "Medium", "High", "Very High"]
    )
    
    return df

# Transform and save enriched data
stats = batch_ops.transform_and_save(
    Query(),
    transform_func=analyze_and_enrich,
    config=BatchConfig(chunk_size=200)
)

print(f"\nTransformation results:")
print(f"  Records processed: {stats['processed']}")
print(f"  Records saved: {stats['saved']}")
print(f"  Errors: {stats['errors']}")

# Verify enriched data
enriched_df = batch_ops.query_as_dataframe(Query().limit(5))
print("\nEnriched data sample:")
print(enriched_df[["product", "revenue", "day_of_week", "is_high_value", "quantity_category"]])
```

## Part 4: Memory-Efficient Processing

### Chunked Processing

Process large datasets without loading everything into memory:

```python
# Initialize chunked processor
processor = ChunkedProcessor(chunk_size=100)

def process_chunk(chunk_df):
    """Process each chunk of data"""
    # Calculate statistics for the chunk
    stats = {
        "count": len(chunk_df),
        "total_revenue": (chunk_df["quantity"] * chunk_df["price"]).sum(),
        "avg_quantity": chunk_df["quantity"].mean(),
        "unique_products": chunk_df["product"].nunique(),
        "date_range": (chunk_df["date"].min(), chunk_df["date"].max())
    }
    return stats

# Process large DataFrame in chunks
chunk_results = processor.process_dataframe(
    sales_df,
    process_func=process_chunk,
    combine_func=lambda results: pd.DataFrame(results)
)

print("\nChunk processing results:")
print(chunk_results.describe())
```

### Streaming from Database

Stream and process records in batches:

```python
def calculate_daily_metrics(chunk_df):
    """Calculate daily sales metrics"""
    daily_stats = chunk_df.groupby("date").agg({
        "quantity": "sum",
        "price": "mean",
        "customer_id": "nunique"
    }).rename(columns={
        "quantity": "total_quantity",
        "price": "avg_price",
        "customer_id": "unique_customers"
    })
    
    daily_stats["revenue"] = (
        chunk_df.groupby("date")
        .apply(lambda x: (x["quantity"] * x["price"]).sum())
    )
    
    return daily_stats

# Stream data from database and process
all_daily_stats = []

for chunk_df in batch_ops.stream_as_dataframe(Query(), chunk_size=200):
    daily_stats = calculate_daily_metrics(chunk_df)
    all_daily_stats.append(daily_stats)

# Combine results
final_daily_stats = pd.concat(all_daily_stats).groupby(level=0).sum()

print("\nDaily sales metrics:")
print(final_daily_stats.head(10))
```

## Part 5: Data Export and Import

### Export to Files

```python
# Export to CSV
csv_stats = batch_ops.export_to_csv(
    Query().filter("region", "=", "North"),
    "north_region_sales.csv",
    config=BatchConfig(chunk_size=500)
)
print(f"Exported {csv_stats['records']} records to CSV")

# Export to Parquet (better for large datasets)
parquet_stats = batch_ops.export_to_parquet(
    Query(),
    "all_sales.parquet",
    compression="snappy"
)
print(f"Exported {parquet_stats['records']} records to Parquet")

# Export to Excel with multiple sheets
with pd.ExcelWriter("sales_report.xlsx") as writer:
    # Sheet 1: Summary by product
    product_summary = batch_ops.aggregate(
        Query(),
        aggregations={"quantity": "sum", "price": "mean"},
        group_by=["product"]
    )
    product_summary.to_excel(writer, sheet_name="Product Summary", index=False)
    
    # Sheet 2: Top customers
    top_customers_df = batch_ops.query_as_dataframe(Query())
    top_customers = (top_customers_df.groupby("customer_id")
                     .agg({"quantity": "sum", "price": "sum"})
                     .nlargest(20, "price"))
    top_customers.to_excel(writer, sheet_name="Top Customers")

print("Exported Excel report with multiple sheets")
```

### Import from Files

```python
# Read CSV in chunks and import to database
def import_csv_to_db(filepath, db, batch_ops):
    """Import CSV file to database in chunks"""
    
    processor = ChunkedProcessor(chunk_size=500)
    
    def process_and_import(chunk_df):
        # Clean and validate data
        chunk_df = chunk_df.dropna()
        chunk_df["imported_at"] = datetime.now().isoformat()
        
        # Import to database
        result = batch_ops.bulk_insert_dataframe(chunk_df)
        return result["inserted"]
    
    results = processor.read_csv_chunked(
        filepath,
        process_func=process_and_import
    )
    
    total_imported = sum(results)
    print(f"Imported {total_imported} records from {filepath}")

# Example import
# import_csv_to_db("external_sales_data.csv", db, batch_ops)
```

## Part 6: Real-World Examples

### Example 1: Time Series Analysis

```python
# Prepare time series data
ts_df = batch_ops.query_as_dataframe(Query())
ts_df["date"] = pd.to_datetime(ts_df["date"])
ts_df = ts_df.set_index("date")

# Daily sales trend
daily_sales = ts_df.groupby(ts_df.index)["quantity"].sum()

# Calculate moving averages
ma_df = pd.DataFrame({
    "daily_sales": daily_sales,
    "ma_3": daily_sales.rolling(window=3).mean(),
    "ma_7": daily_sales.rolling(window=7).mean(),
    "ma_14": daily_sales.rolling(window=14).mean()
})

print("\nSales trend analysis:")
print(ma_df.tail(10))

# Detect anomalies using z-score
from scipy import stats

z_scores = np.abs(stats.zscore(daily_sales.dropna()))
anomalies = daily_sales[z_scores > 2]

print(f"\nDetected {len(anomalies)} anomalous days:")
for date, value in anomalies.items():
    print(f"  {date.strftime('%Y-%m-%d')}: {value} sales")
```

### Example 2: Customer Segmentation

```python
# Prepare customer data
customer_df = batch_ops.query_as_dataframe(Query())

# Calculate customer metrics
customer_metrics = customer_df.groupby("customer_id").agg({
    "quantity": "sum",
    "price": ["mean", "sum", "count"],
    "date": lambda x: (pd.to_datetime(x).max() - pd.to_datetime(x).min()).days
}).round(2)

# Flatten column names
customer_metrics.columns = ["_".join(col).strip() for col in customer_metrics.columns]
customer_metrics.columns = ["total_quantity", "avg_price", "total_spent", "purchase_count", "customer_lifetime_days"]

# Perform RFM segmentation
def calculate_rfm(df):
    """Calculate RFM (Recency, Frequency, Monetary) scores"""
    current_date = pd.to_datetime("2024-02-01")
    
    rfm = customer_df.groupby("customer_id").agg({
        "date": lambda x: (current_date - pd.to_datetime(x).max()).days,
        "quantity": "count",
        "price": "sum"
    })
    
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    
    # Create segments based on quantiles
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=["4", "3", "2", "1"])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=["1", "2", "3", "4"])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=["1", "2", "3", "4"])
    
    # Combine scores
    rfm["RFM_Segment"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    
    # Define segment names
    def segment_name(row):
        if row["RFM_Segment"] == "444":
            return "Champions"
        elif row["R_Score"] == "4" and row["F_Score"] in ["3", "4"]:
            return "Loyal Customers"
        elif row["R_Score"] == "3" and row["M_Score"] == "4":
            return "Big Spenders"
        elif row["R_Score"] == "1":
            return "Lost Customers"
        else:
            return "Regular Customers"
    
    rfm["Segment"] = rfm.apply(segment_name, axis=1)
    
    return rfm

rfm_df = calculate_rfm(customer_df)
print("\nCustomer Segmentation Results:")
print(rfm_df["Segment"].value_counts())

# Save segments back to database
segment_records = converter.dataframe_to_records(rfm_df.reset_index())
for record in segment_records:
    # Update customer records with segments
    pass
```

### Example 3: Data Pipeline

```python
class SalesDataPipeline:
    """Complete data pipeline for sales analysis"""
    
    def __init__(self, db):
        self.db = db
        self.batch_ops = BatchOperations(db)
        self.converter = DataFrameConverter()
    
    def extract(self, date_range=None):
        """Extract data from database"""
        query = Query()
        if date_range:
            query = query.filter("date", ">=", date_range[0])
            query = query.filter("date", "<=", date_range[1])
        
        return self.batch_ops.query_as_dataframe(query)
    
    def transform(self, df):
        """Apply transformations"""
        # Clean data
        df = df.dropna()
        df["date"] = pd.to_datetime(df["date"])
        
        # Add derived columns
        df["revenue"] = df["quantity"] * df["price"]
        df["profit_margin"] = np.random.uniform(0.1, 0.3, len(df))  # Simulated
        df["profit"] = df["revenue"] * df["profit_margin"]
        
        # Add time-based features
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["week"] = df["date"].dt.isocalendar().week
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
        
        # Categorize products
        df["product_category"] = df["product"].map({
            "Laptop": "Computing",
            "Mouse": "Accessories",
            "Keyboard": "Accessories",
            "Monitor": "Display",
            "Headphones": "Audio"
        })
        
        return df
    
    def analyze(self, df):
        """Perform analysis"""
        analysis = {
            "summary": df.describe(),
            "by_product": df.groupby("product").agg({
                "revenue": ["sum", "mean"],
                "quantity": "sum",
                "profit": "sum"
            }),
            "by_region": df.groupby("region").agg({
                "revenue": "sum",
                "customer_id": "nunique"
            }),
            "by_time": df.groupby([df["date"].dt.to_period("W")])["revenue"].sum(),
            "top_products": df.groupby("product")["revenue"].sum().nlargest(5),
            "customer_distribution": df["customer_id"].value_counts().describe()
        }
        
        return analysis
    
    def load(self, df, target_table="processed_sales"):
        """Load processed data"""
        # Create records from DataFrame
        records = self.converter.dataframe_to_records(df)
        
        # Bulk insert to target table
        config = BatchConfig(chunk_size=1000, parallel=True)
        result = self.batch_ops.bulk_insert_dataframe(df, config)
        
        return result
    
    def run(self, date_range=None):
        """Run complete pipeline"""
        print("Starting data pipeline...")
        
        # Extract
        print("Extracting data...")
        raw_df = self.extract(date_range)
        print(f"  Extracted {len(raw_df)} records")
        
        # Transform
        print("Transforming data...")
        processed_df = self.transform(raw_df)
        print(f"  Processed {len(processed_df)} records")
        
        # Analyze
        print("Analyzing data...")
        analysis = self.analyze(processed_df)
        
        # Load
        print("Loading processed data...")
        load_result = self.load(processed_df)
        print(f"  Loaded {load_result['inserted']} records")
        
        return {
            "processed_records": len(processed_df),
            "analysis": analysis,
            "load_result": load_result
        }

# Run the pipeline
pipeline = SalesDataPipeline(db)
results = pipeline.run(date_range=["2024-01-01", "2024-01-31"])

print("\nPipeline Results:")
print(f"Total processed: {results['processed_records']}")
print("\nTop Products by Revenue:")
print(results["analysis"]["top_products"])
```

## Part 7: Performance Optimization

### Optimizing Large Operations

```python
# Performance comparison for different approaches

def benchmark_operations(db, num_records=10000):
    """Benchmark different data operations"""
    import time
    
    # Generate test data
    test_df = pd.DataFrame({
        "id": range(num_records),
        "value": np.random.randn(num_records),
        "category": np.random.choice(["A", "B", "C", "D"], num_records),
        "timestamp": pd.date_range("2024-01-01", periods=num_records, freq="1min")
    })
    
    batch_ops = BatchOperations(db)
    results = {}
    
    # Method 1: Individual inserts
    start = time.time()
    for _, row in test_df.head(100).iterrows():
        db.insert(Record(data=row.to_dict()))
    results["individual_inserts"] = time.time() - start
    
    # Method 2: Bulk insert
    start = time.time()
    batch_ops.bulk_insert_dataframe(
        test_df.head(1000),
        BatchConfig(chunk_size=100)
    )
    results["bulk_insert"] = time.time() - start
    
    # Method 3: Parallel bulk insert
    start = time.time()
    batch_ops.bulk_insert_dataframe(
        test_df,
        BatchConfig(chunk_size=1000, parallel=True, num_workers=4)
    )
    results["parallel_insert"] = time.time() - start
    
    return results

# Run benchmark
# benchmark_results = benchmark_operations(db)
# print("\nPerformance Benchmark:")
# for method, time_taken in benchmark_results.items():
#     print(f"  {method}: {time_taken:.3f} seconds")
```

### Memory Management

```python
# Memory-efficient data processing
def memory_efficient_aggregation(db, batch_ops):
    """Perform aggregation with minimal memory usage"""
    
    # Use iterator pattern
    aggregator = {}
    
    for chunk_df in batch_ops.stream_as_dataframe(Query(), chunk_size=100):
        # Process each chunk
        chunk_agg = chunk_df.groupby("product")["revenue"].sum()
        
        # Merge with existing aggregation
        for product, revenue in chunk_agg.items():
            if product in aggregator:
                aggregator[product] += revenue
            else:
                aggregator[product] = revenue
    
    # Convert to DataFrame
    result_df = pd.DataFrame(
        list(aggregator.items()),
        columns=["product", "total_revenue"]
    )
    
    return result_df

# Example usage
# efficient_result = memory_efficient_aggregation(db, batch_ops)
```

## Best Practices

1. **Use Chunking for Large Data**: Process data in chunks to manage memory
2. **Leverage DataFrame Operations**: Use pandas' optimized operations
3. **Batch Database Operations**: Use bulk operations instead of individual inserts
4. **Choose Right Data Types**: Use appropriate dtypes to reduce memory usage
5. **Index Strategically**: Set proper indexes for better query performance
6. **Parallelize When Possible**: Use parallel processing for independent operations
7. **Cache Computed Results**: Store frequently used aggregations
8. **Monitor Memory Usage**: Track memory consumption during processing
9. **Use Parquet for Storage**: Better compression and faster I/O than CSV
10. **Profile Your Code**: Identify bottlenecks with profiling tools

## Summary

You've learned how to:

- Convert between DataKnobs records and pandas DataFrames
- Perform bulk database operations with DataFrames
- Use memory-efficient processing for large datasets
- Build analytics pipelines with pandas
- Export and import data in various formats
- Optimize performance for large-scale operations

This integration provides the best of both worlds: DataKnobs' robust data management with pandas' powerful analytics capabilities.