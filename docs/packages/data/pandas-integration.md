# Pandas Integration

The DataKnobs data package provides seamless integration with pandas DataFrames, enabling efficient data analysis, transformation, and bulk operations.

## Overview

The pandas integration enables:

- **Bidirectional conversion**: Convert between Records and DataFrames
- **Type preservation**: Maintain field types during conversion
- **Metadata handling**: Preserve record metadata and IDs
- **Bulk operations**: Efficient batch insert/update from DataFrames
- **Query integration**: Return query results as DataFrames
- **Analysis workflows**: Leverage pandas for data analysis

## Core Components

### DataFrameConverter

The `DataFrameConverter` class handles conversion between Records and DataFrames.

```python
from dataknobs_data.pandas import DataFrameConverter
import pandas as pd

# Create converter
converter = DataFrameConverter()

# Convert records to DataFrame
records = database.search(Query())
df = converter.records_to_dataframe(records)

# Convert DataFrame back to records
new_records = converter.dataframe_to_records(df)
```

### Conversion Options

Control conversion behavior with `ConversionOptions`:

```python
from dataknobs_data.pandas import ConversionOptions, MetadataStrategy

options = ConversionOptions(
    preserve_types=True,           # Maintain field types
    include_metadata=True,          # Include metadata columns
    metadata_strategy=MetadataStrategy.SEPARATE,  # How to handle metadata
    flatten_nested=True,           # Flatten nested structures
    parse_json=True,               # Parse JSON fields
    datetime_format="%Y-%m-%d",    # Date format
    null_handling="preserve"       # How to handle nulls
)

df = converter.records_to_dataframe(records, options=options)
```

## Type Mapping

### Field Type to Pandas dtype

The converter automatically maps DataKnobs field types to appropriate pandas dtypes:

```python
from dataknobs_data.fields import FieldType
import pandas as pd
import numpy as np

# Type mapping
TYPE_MAPPING = {
    FieldType.STRING: "string",           # pd.StringDtype()
    FieldType.INTEGER: "Int64",           # pd.Int64Dtype() - nullable
    FieldType.FLOAT: "Float64",           # pd.Float64Dtype() - nullable
    FieldType.BOOLEAN: "boolean",         # pd.BooleanDtype() - nullable
    FieldType.DATETIME: "datetime64[ns]", # pd.DatetimeTZDtype()
    FieldType.JSON: "object",             # Python objects
    FieldType.BINARY: "object",           # Bytes objects
}

# Example conversion with type preservation
records = [
    Record(fields={
        "id": Field("id", FieldType.STRING, "user_001"),
        "age": Field("age", FieldType.INTEGER, 25),
        "score": Field("score", FieldType.FLOAT, 98.5),
        "active": Field("active", FieldType.BOOLEAN, True),
        "created": Field("created", FieldType.DATETIME, datetime.now())
    })
]

df = converter.records_to_dataframe(records)
print(df.dtypes)
# id        string
# age       Int64
# score     Float64
# active    boolean
# created   datetime64[ns]
```

### Custom Type Converters

Define custom type conversion logic:

```python
from dataknobs_data.pandas import TypeConverter

class CustomTypeConverter(TypeConverter):
    """Custom type conversion logic"""
    
    def to_pandas_value(self, field):
        """Convert field value to pandas-compatible value"""
        if field.type == FieldType.JSON:
            # Convert JSON to string for DataFrame
            return json.dumps(field.value) if field.value else None
        elif field.type == "custom_type":
            # Handle custom type
            return str(field.value)
        return super().to_pandas_value(field)
    
    def from_pandas_value(self, value, field_type):
        """Convert pandas value back to field value"""
        if field_type == FieldType.JSON:
            # Parse JSON string
            return json.loads(value) if value else None
        elif field_type == "custom_type":
            # Handle custom type
            return CustomType(value)
        return super().from_pandas_value(value, field_type)

# Use custom converter
converter = DataFrameConverter(type_converter=CustomTypeConverter())
```

## Metadata Preservation

### Metadata Strategies

Different strategies for handling record metadata:

```python
from dataknobs_data.pandas import MetadataStrategy

# Strategy 1: Include metadata as columns
options = ConversionOptions(
    metadata_strategy=MetadataStrategy.COLUMNS
)
df = converter.records_to_dataframe(records, options=options)
# DataFrame includes: _id, _metadata_created, _metadata_updated, etc.

# Strategy 2: Separate metadata DataFrame
options = ConversionOptions(
    metadata_strategy=MetadataStrategy.SEPARATE
)
df, metadata_df = converter.records_to_dataframe(records, options=options)
# df: Contains only field data
# metadata_df: Contains record IDs and metadata

# Strategy 3: Ignore metadata
options = ConversionOptions(
    metadata_strategy=MetadataStrategy.IGNORE
)
df = converter.records_to_dataframe(records, options=options)
# DataFrame contains only field values
```

### ID Preservation

Preserve record IDs during conversion:

```python
# Convert with ID preservation
df = converter.records_to_dataframe(records, preserve_ids=True)
print(df.index)  # Record IDs as index

# Convert back preserving IDs
records = converter.dataframe_to_records(df, use_index_as_id=True)
for record in records:
    print(record.id)  # Original IDs preserved
```

## Batch Operations

### Bulk Insert from DataFrame

Efficiently insert DataFrame data into database:

```python
from dataknobs_data.pandas import BatchOperations

# Create batch operations handler
batch_ops = BatchOperations(database)

# Bulk insert from DataFrame
df = pd.read_csv("large_dataset.csv")
result = batch_ops.bulk_insert_dataframe(
    df,
    batch_size=1000,
    parallel=True,
    validate=True  # Validate against schema
)

print(f"Inserted: {result.successful}")
print(f"Failed: {result.failed}")
if result.errors:
    print("Errors:", result.errors)
```

### Bulk Update

Update existing records from DataFrame:

```python
# Update records matching DataFrame index
df_updates = pd.DataFrame({
    "status": ["active", "inactive", "active"],
    "last_login": [datetime.now()] * 3
}, index=["id1", "id2", "id3"])  # Record IDs as index

result = batch_ops.bulk_update_dataframe(
    df_updates,
    id_column=None,  # Use index as ID
    merge_strategy="update"  # or "replace"
)
```

### Upsert Operations

Insert or update based on existence:

```python
# Upsert: Update if exists, insert if new
result = batch_ops.bulk_upsert_dataframe(
    df,
    id_column="user_id",  # Column to use as record ID
    batch_size=500
)

print(f"Inserted: {result.inserted}")
print(f"Updated: {result.updated}")
```

## Query Integration

### Query Results as DataFrame

Get query results directly as DataFrame:

```python
from dataknobs_data.pandas import PandasDatabase

# Wrap database with pandas support
pandas_db = PandasDatabase(database)

# Query returning DataFrame
df = pandas_db.search_dataframe(
    Query(
        filters=[Filter("status", "=", "active")],
        sort=[Sort("created_at", "desc")],
        limit=1000
    )
)

# Use pandas for analysis
summary = df.groupby("category").agg({
    "price": ["mean", "std", "count"],
    "quantity": "sum"
})
```

### Aggregation Queries

Perform aggregations with pandas:

```python
# Get all data as DataFrame
df = pandas_db.all_as_dataframe()

# Complex aggregation
result = df.groupby(["category", "status"]).agg({
    "revenue": "sum",
    "quantity": "sum",
    "customer_id": "nunique",
    "created_at": ["min", "max"]
}).round(2)

# Time-based aggregation
df["created_at"] = pd.to_datetime(df["created_at"])
daily_stats = df.set_index("created_at").resample("D").agg({
    "revenue": "sum",
    "orders": "count"
})
```

## Data Analysis Workflows

### ETL Pipeline

Extract, Transform, Load workflow:

```python
class DataPipeline:
    """ETL pipeline using pandas"""
    
    def __init__(self, source_db, target_db):
        self.source_db = PandasDatabase(source_db)
        self.target_db = PandasDatabase(target_db)
        self.converter = DataFrameConverter()
    
    def run(self, query=None):
        # Extract
        df = self.source_db.search_dataframe(query or Query())
        
        # Transform
        df = self.transform(df)
        
        # Load
        result = self.target_db.bulk_insert_dataframe(df)
        return result
    
    def transform(self, df):
        """Apply transformations"""
        # Clean data
        df = df.dropna(subset=["required_field"])
        df["email"] = df["email"].str.lower().str.strip()
        
        # Add computed columns
        df["full_name"] = df["first_name"] + " " + df["last_name"]
        df["age_group"] = pd.cut(df["age"], 
                                 bins=[0, 18, 30, 50, 100],
                                 labels=["youth", "young", "middle", "senior"])
        
        # Aggregate if needed
        if "daily_summary" in self.target_db.name:
            df = df.groupby(["date", "category"]).agg({
                "sales": "sum",
                "customers": "nunique"
            }).reset_index()
        
        return df

# Run pipeline
pipeline = DataPipeline(source_db, target_db)
result = pipeline.run()
```

### Data Cleaning

Clean and validate data using pandas:

```python
def clean_dataset(database):
    """Clean and validate dataset"""
    
    # Load data
    pandas_db = PandasDatabase(database)
    df = pandas_db.all_as_dataframe()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["email"], keep="first")
    
    # Fix data types
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    
    # Handle missing values
    df["status"] = df["status"].fillna("unknown")
    df["score"] = df["score"].fillna(df["score"].median())
    
    # Validate ranges
    df.loc[df["age"] < 0, "age"] = None
    df.loc[df["age"] > 150, "age"] = None
    
    # Standardize text
    df["category"] = df["category"].str.lower().str.strip()
    df["category"] = df["category"].replace({
        "electronic": "electronics",
        "clothes": "clothing"
    })
    
    # Save cleaned data
    batch_ops = BatchOperations(database)
    batch_ops.bulk_upsert_dataframe(df, id_column="id")
    
    return df
```

### Statistical Analysis

Perform statistical analysis on data:

```python
def analyze_dataset(database):
    """Statistical analysis of dataset"""
    
    pandas_db = PandasDatabase(database)
    df = pandas_db.all_as_dataframe()
    
    # Basic statistics
    print("Dataset Overview:")
    print(df.describe())
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()
    
    # Find strong correlations
    strong_corr = correlations[abs(correlations) > 0.7]
    print("\nStrong Correlations:")
    print(strong_corr)
    
    # Distribution analysis
    for col in numeric_cols:
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        print(f"\n{col}:")
        print(f"  Skewness: {skewness:.2f}")
        print(f"  Kurtosis: {kurtosis:.2f}")
    
    # Outlier detection
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                (df[numeric_cols] > (Q3 + 1.5 * IQR)))
    
    print("\nOutliers per column:")
    print(outliers.sum())
    
    return {
        "summary": df.describe(),
        "correlations": correlations,
        "outliers": outliers
    }
```

## Performance Optimization

### Chunked Processing

Process large datasets in chunks:

```python
def process_large_dataset(database, chunk_size=10000):
    """Process large dataset in chunks"""
    
    total = database.count()
    processed = 0
    
    while processed < total:
        # Get chunk
        query = Query(offset=processed, limit=chunk_size)
        records = database.search(query)
        
        # Convert to DataFrame
        df = converter.records_to_dataframe(records)
        
        # Process chunk
        df = process_chunk(df)
        
        # Save results
        batch_ops.bulk_upsert_dataframe(df)
        
        processed += len(records)
        print(f"Processed {processed}/{total} records")
        
        # Clear memory
        del df
        gc.collect()
```

### Parallel Processing

Use parallel processing for better performance:

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def parallel_transform(database, transform_fn, num_workers=4):
    """Transform data in parallel"""
    
    # Split data into partitions
    total = database.count()
    partition_size = total // num_workers
    
    def process_partition(offset, limit):
        """Process a single partition"""
        query = Query(offset=offset, limit=limit)
        records = database.search(query)
        df = converter.records_to_dataframe(records)
        df = transform_fn(df)
        return df
    
    # Process partitions in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            offset = i * partition_size
            limit = partition_size if i < num_workers - 1 else total - offset
            futures.append(executor.submit(process_partition, offset, limit))
        
        # Combine results
        results = pd.concat([f.result() for f in futures])
    
    return results
```

### Memory Management

Optimize memory usage for large DataFrames:

```python
def optimize_dataframe_memory(df):
    """Reduce DataFrame memory usage"""
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            if col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif col_type == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            # Convert string columns to category if appropriate
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (initial_memory - final_memory) / initial_memory * 100
    
    print(f"Memory usage reduced by {reduction:.1f}%")
    print(f"From {initial_memory:.1f} MB to {final_memory:.1f} MB")
    
    return df
```

## Integration Examples

### Data Export

Export data to various formats:

```python
def export_data(database, format="csv", query=None):
    """Export data in various formats"""
    
    # Get data as DataFrame
    pandas_db = PandasDatabase(database)
    df = pandas_db.search_dataframe(query or Query())
    
    if format == "csv":
        df.to_csv("export.csv", index=False)
    elif format == "excel":
        with pd.ExcelWriter("export.xlsx") as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
            # Add summary sheet
            summary = df.describe()
            summary.to_excel(writer, sheet_name="Summary")
    elif format == "parquet":
        df.to_parquet("export.parquet", compression="snappy")
    elif format == "json":
        df.to_json("export.json", orient="records", indent=2)
    elif format == "html":
        df.to_html("export.html", index=False)
    
    print(f"Exported {len(df)} records to {format}")
```

### Data Import

Import data from various sources:

```python
def import_data(database, file_path, format="auto"):
    """Import data from file"""
    
    # Detect format
    if format == "auto":
        ext = file_path.split(".")[-1].lower()
        format = ext
    
    # Read file into DataFrame
    if format == "csv":
        df = pd.read_csv(file_path)
    elif format in ["xlsx", "xls"]:
        df = pd.read_excel(file_path)
    elif format == "parquet":
        df = pd.read_parquet(file_path)
    elif format == "json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Import into database
    batch_ops = BatchOperations(database)
    result = batch_ops.bulk_insert_dataframe(df, validate=True)
    
    print(f"Imported {result.successful} records")
    if result.failed:
        print(f"Failed: {result.failed}")
    
    return result
```

### Real-time Analytics

Combine database queries with pandas analytics:

```python
class RealTimeAnalytics:
    """Real-time analytics using pandas"""
    
    def __init__(self, database):
        self.db = PandasDatabase(database)
        self.cache = {}
        self.cache_ttl = 60  # seconds
    
    def get_metrics(self, time_range="1h"):
        """Get real-time metrics"""
        
        cache_key = f"metrics_{time_range}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        # Query recent data
        cutoff = datetime.now() - pd.Timedelta(time_range)
        query = Query(filters=[
            Filter("created_at", ">=", cutoff)
        ])
        
        df = self.db.search_dataframe(query)
        
        # Calculate metrics
        metrics = {
            "total_records": len(df),
            "unique_users": df["user_id"].nunique(),
            "average_value": df["value"].mean(),
            "total_revenue": df["revenue"].sum(),
            "top_categories": df["category"].value_counts().head(5).to_dict(),
            "hourly_trend": df.set_index("created_at").resample("H")["value"].sum().to_dict()
        }
        
        # Cache results
        self.cache[cache_key] = (time.time(), metrics)
        
        return metrics
```

## Best Practices

1. **Choose appropriate strategies**: Select metadata and type preservation strategies based on use case
2. **Optimize memory usage**: Use appropriate dtypes and consider chunking for large datasets
3. **Validate data**: Validate DataFrames before inserting into database
4. **Handle nulls explicitly**: Define clear null handling strategy
5. **Use batch operations**: Leverage batch operations for better performance
6. **Consider parallelization**: Use parallel processing for large transformations
7. **Cache when appropriate**: Cache frequently accessed DataFrames
8. **Monitor performance**: Track conversion and operation times
9. **Test round-trip conversion**: Ensure data integrity in conversions
10. **Document transformations**: Keep clear records of data transformations

## See Also

- [Batch Operations](backends.md#batch-operations) - Batch operations in backends
- [Migration Utilities](migration.md) - Data migration tools
- [Query System](query.md) - Query construction and execution
- [Pandas Tutorial](tutorials/pandas-tutorial.md) - Step-by-step pandas guide