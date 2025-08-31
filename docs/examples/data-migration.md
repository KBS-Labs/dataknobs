# Data Migration Examples

This document provides comprehensive examples of data migration scenarios using the DataKnobs Data package, from simple schema changes to complex multi-system migrations.

## Example 1: E-commerce Platform Migration

### Scenario
Migrating an e-commerce platform from legacy schema (v1) to modern microservices architecture (v2).

### Legacy Schema (v1)
```python
# Old monolithic structure
legacy_order = {
    "order_id": "ORD-2024-001",
    "customer_name": "John Smith",
    "customer_email": "john.smith@example.com",
    "customer_phone": "555-0123",
    "customer_address": "123 Main St, Boston, MA 02101",
    "product_ids": "PROD-101,PROD-102,PROD-103",
    "quantities": "2,1,3",
    "prices": "29.99,49.99,19.99",
    "order_total": 189.94,
    "order_status": 1,  # 1=pending, 2=shipped, 3=delivered
    "created_date": "2024-01-15 10:30:00"
}
```

### Target Schema (v2)
```python
# New microservices structure
modern_order = {
    "id": "ORD-2024-001",
    "customer": {
        "name": {"first": "John", "last": "Smith"},
        "contact": {
            "email": "john.smith@example.com",
            "phone": "+1-555-0123"
        },
        "address": {
            "street": "123 Main St",
            "city": "Boston",
            "state": "MA",
            "zip": "02101"
        }
    },
    "items": [
        {"product_id": "PROD-101", "quantity": 2, "unit_price": 29.99},
        {"product_id": "PROD-102", "quantity": 1, "unit_price": 49.99},
        {"product_id": "PROD-103", "quantity": 3, "unit_price": 19.99}
    ],
    "pricing": {
        "subtotal": 169.95,
        "tax": 13.99,
        "shipping": 6.00,
        "total": 189.94
    },
    "status": "pending",
    "timestamps": {
        "created": "2024-01-15T10:30:00Z",
        "updated": "2024-01-15T10:30:00Z"
    }
}
```

### Implementation

```python
from dataknobs_data import Record, MemoryDatabase, Migration, Migrator
from dataknobs_data.migration import Transformer, CompositeOperation
from dataknobs_data.validation import Schema, Pattern, Range, Enum
from datetime import datetime
import re

class EcommerceMigration:
    """Migrate e-commerce orders from v1 to v2 schema"""
    
    def __init__(self):
        self.source_db = MemoryDatabase()
        self.target_db = MemoryDatabase()
        self.setup_validation_schemas()
    
    def setup_validation_schemas(self):
        """Define validation schemas for v2"""
        self.order_schema = (Schema("ModernOrder")
            .field("id", "STRING", required=True, 
                   constraints=[Pattern(r"^ORD-\d{4}-\d{3}$")])
            .field("customer", "DICT", required=True)
            .field("items", "LIST", required=True)
            .field("pricing", "DICT", required=True)
            .field("status", "STRING", required=True,
                   constraints=[Enum(["pending", "processing", "shipped", "delivered", "cancelled"])])
            .field("timestamps", "DICT", required=True)
        )
    
    def create_transformer(self):
        """Create transformation logic"""
        return OrderTransformer()
    
    def migrate(self):
        """Perform the migration"""
        transformer = self.create_transformer()
        migrator = Migrator()
        
        # Track progress
        def on_progress(progress):
            print(f"Migration progress: {progress.percent:.1f}% "
                  f"({progress.processed}/{progress.total} records)")
            if progress.failed > 0:
                print(f"  Failed: {progress.failed} records")
        
        # Perform migration with validation
        progress = migrator.migrate(
            source=self.source_db,
            target=self.target_db,
            transform=transformer,
            batch_size=100,
            on_progress=on_progress
        )
        
        print(f"\nMigration complete:")
        print(f"  Total processed: {progress.processed}")
        print(f"  Success rate: {(1 - progress.failed/progress.total)*100:.1f}%")
        print(f"  Duration: {progress.elapsed_time:.2f} seconds")
        
        return progress

class OrderTransformer:
    """Transform orders from v1 to v2 schema"""
    
    def transform_many(self, records):
        transformed = []
        for record in records:
            try:
                new_record = self.transform_order(record)
                transformed.append(new_record)
            except Exception as e:
                print(f"Failed to transform record {record.data.get('order_id')}: {e}")
        return transformed
    
    def transform_order(self, record):
        """Transform a single order record"""
        old_data = record.data
        
        # Parse customer name
        name_parts = old_data.get("customer_name", "").split(" ", 1)
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        # Parse address
        address_parts = old_data.get("customer_address", "").split(", ")
        street = address_parts[0] if address_parts else ""
        city = address_parts[1] if len(address_parts) > 1 else ""
        state_zip = address_parts[2] if len(address_parts) > 2 else " "
        state, zip_code = state_zip.split(" ") if " " in state_zip else ("", "")
        
        # Parse items
        product_ids = old_data.get("product_ids", "").split(",")
        quantities = [int(q) for q in old_data.get("quantities", "").split(",")]
        prices = [float(p) for p in old_data.get("prices", "").split(",")]
        
        items = []
        for i in range(len(product_ids)):
            if i < len(quantities) and i < len(prices):
                items.append({
                    "product_id": product_ids[i].strip(),
                    "quantity": quantities[i],
                    "unit_price": prices[i]
                })
        
        # Calculate pricing
        subtotal = sum(item["quantity"] * item["unit_price"] for item in items)
        tax = subtotal * 0.0825  # 8.25% tax rate
        shipping = 6.00 if subtotal < 100 else 0  # Free shipping over $100
        
        # Convert status
        status_map = {1: "pending", 2: "shipped", 3: "delivered"}
        status = status_map.get(old_data.get("order_status", 1), "pending")
        
        # Parse and format timestamp
        created_str = old_data.get("created_date", "")
        try:
            created_dt = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
            created_iso = created_dt.isoformat() + "Z"
        except:
            created_iso = datetime.now().isoformat() + "Z"
        
        # Build new structure
        new_data = {
            "id": old_data.get("order_id"),
            "customer": {
                "name": {"first": first_name, "last": last_name},
                "contact": {
                    "email": old_data.get("customer_email", ""),
                    "phone": self.format_phone(old_data.get("customer_phone", ""))
                },
                "address": {
                    "street": street,
                    "city": city,
                    "state": state,
                    "zip": zip_code
                }
            },
            "items": items,
            "pricing": {
                "subtotal": round(subtotal, 2),
                "tax": round(tax, 2),
                "shipping": round(shipping, 2),
                "total": round(subtotal + tax + shipping, 2)
            },
            "status": status,
            "timestamps": {
                "created": created_iso,
                "updated": created_iso
            }
        }
        
        return Record(data=new_data, metadata={"migrated_from": "v1"})
    
    def format_phone(self, phone):
        """Format phone number to international format"""
        # Remove non-digits
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        return phone

# Run the migration
def run_ecommerce_migration():
    migration = EcommerceMigration()
    
    # Load sample data
    sample_orders = [
        {
            "order_id": "ORD-2024-001",
            "customer_name": "John Smith",
            "customer_email": "john.smith@example.com",
            "customer_phone": "555-0123",
            "customer_address": "123 Main St, Boston, MA 02101",
            "product_ids": "PROD-101,PROD-102,PROD-103",
            "quantities": "2,1,3",
            "prices": "29.99,49.99,19.99",
            "order_total": 189.94,
            "order_status": 1,
            "created_date": "2024-01-15 10:30:00"
        },
        # Add more sample orders...
    ]
    
    for order_data in sample_orders:
        migration.source_db.insert(Record(data=order_data))
    
    # Perform migration
    progress = migration.migrate()
    
    # Verify results
    print("\nSample migrated record:")
    migrated_records = list(migration.target_db.search(Query().limit(1)))
    if migrated_records:
        import json
        print(json.dumps(migrated_records[0].data, indent=2))
    
    return migration

# Execute
if __name__ == "__main__":
    run_ecommerce_migration()
```

## Example 2: Healthcare Data Migration

### Scenario
Migrating patient records from multiple hospital systems to a unified healthcare platform.

```python
from dataknobs_data import Record, Query
from dataknobs_data.migration import Migrator, Transformer
from dataknobs_data.validation import Schema, Pattern, Custom
import hashlib
from datetime import datetime, date

class HealthcareDataMigration:
    """Migrate and consolidate healthcare records from multiple sources"""
    
    def __init__(self):
        self.hospital_a_db = MemoryDatabase()  # Hospital A format
        self.hospital_b_db = MemoryDatabase()  # Hospital B format
        self.clinic_db = MemoryDatabase()      # Clinic format
        self.unified_db = MemoryDatabase()     # Unified platform
    
    def migrate_hospital_a(self):
        """Migrate Hospital A records (uses SSN as ID)"""
        
        class HospitalATransformer:
            def transform_many(self, records):
                transformed = []
                for record in records:
                    # Hospital A specific transformation
                    data = record.data
                    
                    # Generate secure patient ID from SSN
                    patient_id = self.generate_patient_id(data.get("ssn"))
                    
                    # Standardize medical record
                    unified = {
                        "patient_id": patient_id,
                        "demographics": {
                            "name": {
                                "first": data.get("first_name"),
                                "last": data.get("last_name"),
                                "middle": data.get("middle_initial", "")
                            },
                            "dob": self.parse_date(data.get("birth_date")),
                            "gender": self.standardize_gender(data.get("sex")),
                            "contact": {
                                "phone": data.get("phone"),
                                "email": data.get("email"),
                                "address": {
                                    "street": data.get("address_line1"),
                                    "city": data.get("city"),
                                    "state": data.get("state"),
                                    "zip": data.get("zip")
                                }
                            }
                        },
                        "medical": {
                            "mrn": data.get("medical_record_number"),
                            "blood_type": data.get("blood_type"),
                            "allergies": self.parse_list(data.get("allergies")),
                            "conditions": self.parse_list(data.get("diagnoses")),
                            "medications": self.parse_medications(data.get("current_meds"))
                        },
                        "insurance": {
                            "provider": data.get("insurance_company"),
                            "policy_number": data.get("policy_id"),
                            "group_number": data.get("group_id")
                        },
                        "source": {
                            "system": "Hospital_A",
                            "original_id": data.get("patient_id"),
                            "imported_at": datetime.now().isoformat()
                        }
                    }
                    
                    transformed.append(Record(data=unified))
                return transformed
            
            def generate_patient_id(self, ssn):
                """Generate secure patient ID from SSN"""
                if not ssn:
                    return f"TEMP_{datetime.now().timestamp()}"
                # Hash SSN for security
                hash_obj = hashlib.sha256(ssn.encode())
                return f"PAT-{hash_obj.hexdigest()[:12].upper()}"
            
            def parse_date(self, date_str):
                """Parse various date formats"""
                if not date_str:
                    return None
                
                formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt).date().isoformat()
                    except:
                        continue
                return date_str
            
            def standardize_gender(self, gender):
                """Standardize gender values"""
                if not gender:
                    return "unknown"
                
                gender = gender.lower()
                if gender in ["m", "male"]:
                    return "male"
                elif gender in ["f", "female"]:
                    return "female"
                else:
                    return "other"
            
            def parse_list(self, list_str):
                """Parse comma-separated list"""
                if not list_str:
                    return []
                return [item.strip() for item in list_str.split(",")]
            
            def parse_medications(self, meds_str):
                """Parse medication list with dosages"""
                if not meds_str:
                    return []
                
                meds = []
                for med in self.parse_list(meds_str):
                    parts = med.split("-")
                    meds.append({
                        "name": parts[0].strip() if parts else med,
                        "dosage": parts[1].strip() if len(parts) > 1 else "",
                        "frequency": parts[2].strip() if len(parts) > 2 else ""
                    })
                return meds
        
        # Perform migration
        migrator = Migrator()
        return migrator.migrate(
            source=self.hospital_a_db,
            target=self.unified_db,
            transform=HospitalATransformer(),
            batch_size=500
        )
    
    def migrate_hospital_b(self):
        """Migrate Hospital B records (uses different format)"""
        
        class HospitalBTransformer:
            def transform_many(self, records):
                # Hospital B specific transformation
                # Similar structure but different field names
                pass
        
        # Implementation similar to Hospital A
        pass
    
    def deduplicate_patients(self):
        """Identify and merge duplicate patient records"""
        
        from collections import defaultdict
        
        # Group by similar demographics
        patient_groups = defaultdict(list)
        
        for record in self.unified_db.search(Query()):
            demo = record.data.get("demographics", {})
            name = demo.get("name", {})
            
            # Create matching key
            key = (
                name.get("first", "").lower(),
                name.get("last", "").lower(),
                demo.get("dob", "")
            )
            
            patient_groups[key].append(record)
        
        # Merge duplicates
        merged_count = 0
        for key, records in patient_groups.items():
            if len(records) > 1:
                # Merge records, keeping most complete data
                merged = self.merge_patient_records(records)
                
                # Delete duplicates
                for record in records:
                    self.unified_db.delete(record.id)
                
                # Insert merged record
                self.unified_db.insert(merged)
                merged_count += 1
        
        print(f"Merged {merged_count} duplicate patient records")
    
    def merge_patient_records(self, records):
        """Merge multiple patient records into one"""
        merged_data = {}
        
        # Implement intelligent merging logic
        # Prefer non-empty fields, most recent updates, etc.
        
        return Record(data=merged_data)
    
    def validate_migrated_data(self):
        """Validate all migrated healthcare records"""
        
        # Define validation schema
        healthcare_schema = (Schema("UnifiedHealthRecord")
            .field("patient_id", "STRING", required=True,
                   constraints=[Pattern(r"^PAT-[A-Z0-9]{12}$")])
            .field("demographics", "DICT", required=True)
            .field("medical", "DICT", required=True)
            .field("source", "DICT", required=True)
        )
        
        # Validate all records
        invalid_count = 0
        for record in self.unified_db.search(Query()):
            result = healthcare_schema.validate(record)
            if not result.valid:
                invalid_count += 1
                print(f"Invalid record {record.data.get('patient_id')}: {result.errors}")
        
        total = self.unified_db.search(Query()).count()
        print(f"Validation complete: {total - invalid_count}/{total} valid records")
    
    def run_full_migration(self):
        """Execute complete healthcare migration"""
        print("Starting healthcare data migration...")
        
        # Migrate from each source
        print("\n1. Migrating Hospital A...")
        self.migrate_hospital_a()
        
        print("\n2. Migrating Hospital B...")
        self.migrate_hospital_b()
        
        print("\n3. Migrating Clinic data...")
        # self.migrate_clinic()
        
        # Post-processing
        print("\n4. Deduplicating patient records...")
        self.deduplicate_patients()
        
        print("\n5. Validating migrated data...")
        self.validate_migrated_data()
        
        # Generate report
        total = len(list(self.unified_db.search(Query())))
        print(f"\n✅ Migration complete: {total} unified patient records")
```

## Example 3: Financial Data Migration

### Scenario
Migrating from legacy banking system to modern fintech platform with real-time processing.

```python
from decimal import Decimal
from dataknobs_data import Record
from dataknobs_data.migration import Migrator, Migration, AddField, TransformField
import uuid

class FinancialDataMigration:
    """Migrate financial transactions with precision and compliance"""
    
    def __init__(self):
        self.legacy_db = MemoryDatabase()
        self.modern_db = MemoryDatabase()
        self.audit_db = MemoryDatabase()  # Audit trail
    
    def create_migration_pipeline(self):
        """Create multi-stage migration pipeline"""
        
        # Stage 1: Basic transformation
        stage1 = Migration("legacy", "v1", "Basic field mapping")
        stage1.add(RenameField("acct_no", "account_number"))
        stage1.add(RenameField("tx_date", "transaction_date"))
        stage1.add(RenameField("tx_amt", "amount"))
        
        # Stage 2: Data enrichment
        stage2 = Migration("v1", "v2", "Add computed fields")
        stage2.add(AddField("transaction_id", lambda: str(uuid.uuid4())))
        stage2.add(TransformField("amount", self.convert_to_decimal))
        stage2.add(AddField("currency", "USD"))
        stage2.add(AddField("status", "completed"))
        
        # Stage 3: Compliance additions
        stage3 = Migration("v2", "v3", "Add compliance fields")
        stage3.add(AddField("aml_checked", False))
        stage3.add(AddField("risk_score", self.calculate_risk_score))
        stage3.add(AddField("reporting_required", self.check_reporting_requirement))
        
        return [stage1, stage2, stage3]
    
    def convert_to_decimal(self, amount):
        """Convert amount to Decimal for precision"""
        if isinstance(amount, str):
            amount = amount.replace("$", "").replace(",", "")
        return str(Decimal(amount).quantize(Decimal("0.01")))
    
    def calculate_risk_score(self, record):
        """Calculate transaction risk score"""
        amount = Decimal(record.get("amount", 0))
        
        # Simple risk scoring
        if amount > 10000:
            return "high"
        elif amount > 5000:
            return "medium"
        else:
            return "low"
    
    def check_reporting_requirement(self, record):
        """Check if transaction requires regulatory reporting"""
        amount = Decimal(record.get("amount", 0))
        return amount >= 10000  # CTR requirement
    
    def migrate_with_audit(self):
        """Perform migration with full audit trail"""
        
        class AuditingTransformer:
            def __init__(self, audit_db):
                self.audit_db = audit_db
            
            def transform_many(self, records):
                transformed = []
                
                for record in records:
                    # Create audit entry
                    audit_entry = Record(data={
                        "original_record": record.data.copy(),
                        "migration_timestamp": datetime.now().isoformat(),
                        "source_system": "legacy_banking",
                        "target_system": "modern_fintech"
                    })
                    
                    # Apply transformations
                    # ... transformation logic ...
                    
                    # Record the transformation
                    audit_entry.data["transformed_record"] = transformed_record.data
                    audit_entry.data["transformation_status"] = "success"
                    
                    self.audit_db.insert(audit_entry)
                    transformed.append(transformed_record)
                
                return transformed
        
        # Execute migration with auditing
        migrator = Migrator()
        transformer = AuditingTransformer(self.audit_db)
        
        progress = migrator.migrate(
            source=self.legacy_db,
            target=self.modern_db,
            transform=transformer,
            batch_size=1000
        )
        
        return progress
    
    def reconcile_balances(self):
        """Reconcile account balances post-migration"""
        
        # Calculate balances in legacy system
        legacy_balances = {}
        for record in self.legacy_db.search(Query()):
            acct = record.data.get("acct_no")
            amt = Decimal(record.data.get("tx_amt", 0))
            legacy_balances[acct] = legacy_balances.get(acct, Decimal(0)) + amt
        
        # Calculate balances in modern system
        modern_balances = {}
        for record in self.modern_db.search(Query()):
            acct = record.data.get("account_number")
            amt = Decimal(record.data.get("amount", 0))
            modern_balances[acct] = modern_balances.get(acct, Decimal(0)) + amt
        
        # Compare and report discrepancies
        discrepancies = []
        for acct, legacy_bal in legacy_balances.items():
            modern_bal = modern_balances.get(acct, Decimal(0))
            if abs(legacy_bal - modern_bal) > Decimal("0.01"):
                discrepancies.append({
                    "account": acct,
                    "legacy_balance": str(legacy_bal),
                    "modern_balance": str(modern_bal),
                    "difference": str(legacy_bal - modern_bal)
                })
        
        if discrepancies:
            print(f"⚠️ Found {len(discrepancies)} balance discrepancies")
            for disc in discrepancies[:5]:
                print(f"  Account {disc['account']}: ${disc['difference']} difference")
        else:
            print("✅ All balances reconciled successfully")
        
        return discrepancies
```

## Example 4: IoT Sensor Data Migration

### Scenario
Migrating time-series IoT sensor data with data compression and aggregation.

```python
from dataknobs_data.pandas import DataFrameConverter, BatchOperations
import pandas as pd
import numpy as np

class IoTDataMigration:
    """Migrate and optimize IoT sensor data"""
    
    def __init__(self):
        self.raw_sensor_db = MemoryDatabase()
        self.optimized_db = MemoryDatabase()
        self.converter = DataFrameConverter()
        self.batch_ops = BatchOperations(self.optimized_db)
    
    def migrate_with_aggregation(self):
        """Migrate sensor data with time-based aggregation"""
        
        # Read raw sensor data
        raw_df = self.batch_ops.query_as_dataframe(Query())
        
        # Convert timestamp column
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
        raw_df = raw_df.set_index('timestamp')
        
        # Resample to 5-minute intervals
        aggregated = raw_df.resample('5T').agg({
            'temperature': ['mean', 'min', 'max', 'std'],
            'humidity': ['mean', 'min', 'max'],
            'pressure': 'mean',
            'sensor_id': 'first'  # Keep sensor ID
        })
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        
        # Add data quality metrics
        aggregated['sample_count'] = raw_df.resample('5T').size()
        aggregated['data_quality'] = aggregated['sample_count'].apply(
            lambda x: 'good' if x >= 50 else 'poor'
        )
        
        # Detect anomalies
        aggregated['temperature_anomaly'] = (
            np.abs(aggregated['temperature_mean'] - aggregated['temperature_mean'].rolling(12).mean()) 
            > 2 * aggregated['temperature_mean'].rolling(12).std()
        )
        
        # Convert back to records and save
        aggregated_records = self.converter.dataframe_to_records(
            aggregated.reset_index()
        )
        
        for record in aggregated_records:
            self.optimized_db.insert(record)
        
        print(f"Compressed {len(raw_df)} raw readings to {len(aggregated)} aggregated records")
        print(f"Compression ratio: {len(raw_df)/len(aggregated):.1f}:1")
    
    def migrate_with_partitioning(self):
        """Partition data by sensor and time for efficient querying"""
        
        class PartitioningTransformer:
            def transform_many(self, records):
                transformed = []
                
                for record in records:
                    # Add partition keys
                    timestamp = datetime.fromisoformat(record.data['timestamp'])
                    
                    enhanced = record.data.copy()
                    enhanced['partition_year'] = timestamp.year
                    enhanced['partition_month'] = timestamp.month
                    enhanced['partition_day'] = timestamp.day
                    enhanced['partition_hour'] = timestamp.hour
                    enhanced['partition_sensor'] = record.data['sensor_id'] % 10
                    
                    transformed.append(Record(data=enhanced))
                
                return transformed
        
        # Migrate with partitioning
        migrator = Migrator()
        progress = migrator.migrate(
            source=self.raw_sensor_db,
            target=self.optimized_db,
            transform=PartitioningTransformer(),
            batch_size=10000
        )
        
        return progress
```

## Example 5: Multi-Database Synchronization

### Scenario
Keeping multiple databases synchronized with different schemas.

```python
class MultiDatabaseSync:
    """Synchronize data across multiple database systems"""
    
    def __init__(self):
        self.postgres_db = MemoryDatabase()  # Primary
        self.mongodb_db = MemoryDatabase()   # Document store
        self.redis_db = MemoryDatabase()     # Cache
        self.elasticsearch_db = MemoryDatabase()  # Search
    
    def create_sync_pipeline(self):
        """Create transformers for each target database"""
        
        class MongoTransformer:
            """Transform relational data to document format"""
            def transform_many(self, records):
                documents = []
                for record in records:
                    # Convert to nested document structure
                    doc = {
                        "_id": record.data.get("id"),
                        "data": record.data,
                        "metadata": {
                            "source": "postgres",
                            "synced_at": datetime.now().isoformat()
                        }
                    }
                    documents.append(Record(data=doc))
                return documents
        
        class RedisTransformer:
            """Transform for Redis key-value storage"""
            def transform_many(self, records):
                kv_pairs = []
                for record in records:
                    # Create key-value structure
                    key = f"record:{record.data.get('id')}"
                    value = json.dumps(record.data)
                    
                    kv_pairs.append(Record(data={
                        "key": key,
                        "value": value,
                        "ttl": 3600  # 1 hour TTL
                    }))
                return kv_pairs
        
        class ElasticsearchTransformer:
            """Transform for Elasticsearch indexing"""
            def transform_many(self, records):
                documents = []
                for record in records:
                    # Add search-specific fields
                    doc = record.data.copy()
                    doc["_index"] = "records"
                    doc["_type"] = "_doc"
                    doc["suggest"] = {
                        "input": [record.data.get("name", "")],
                        "weight": 1
                    }
                    documents.append(Record(data=doc))
                return documents
        
        return {
            "mongodb": MongoTransformer(),
            "redis": RedisTransformer(),
            "elasticsearch": ElasticsearchTransformer()
        }
    
    def sync_databases(self, source_query=None):
        """Synchronize all databases from primary source"""
        
        if source_query is None:
            source_query = Query()  # Sync all records
        
        transformers = self.create_sync_pipeline()
        migrator = Migrator()
        
        results = {}
        
        # Sync to MongoDB
        results["mongodb"] = migrator.migrate(
            source=self.postgres_db,
            target=self.mongodb_db,
            transform=transformers["mongodb"],
            batch_size=500
        )
        
        # Sync to Redis (cache recent data only)
        recent_query = Query().filter("updated_at", ">", 
                                     (datetime.now() - timedelta(hours=24)).isoformat())
        results["redis"] = migrator.migrate(
            source=self.postgres_db,
            target=self.redis_db,
            transform=transformers["redis"],
            query=recent_query,
            batch_size=100
        )
        
        # Sync to Elasticsearch
        results["elasticsearch"] = migrator.migrate(
            source=self.postgres_db,
            target=self.elasticsearch_db,
            transform=transformers["elasticsearch"],
            batch_size=1000
        )
        
        # Print sync summary
        print("\nDatabase Synchronization Summary:")
        for db, progress in results.items():
            print(f"  {db}: {progress.processed} records synced in {progress.elapsed_time:.2f}s")
        
        return results
    
    def verify_consistency(self):
        """Verify data consistency across all databases"""
        
        # Get record counts
        counts = {
            "postgres": len(list(self.postgres_db.search(Query()))),
            "mongodb": len(list(self.mongodb_db.search(Query()))),
            "redis": len(list(self.redis_db.search(Query()))),
            "elasticsearch": len(list(self.elasticsearch_db.search(Query())))
        }
        
        print("\nConsistency Check:")
        print(f"  PostgreSQL: {counts['postgres']} records")
        print(f"  MongoDB: {counts['mongodb']} records")
        print(f"  Redis: {counts['redis']} records (recent only)")
        print(f"  Elasticsearch: {counts['elasticsearch']} records")
        
        # Sample and compare records
        sample_size = min(10, counts["postgres"])
        sample_ids = [r.data.get("id") for r in self.postgres_db.search(Query().limit(sample_size))]
        
        inconsistencies = []
        for record_id in sample_ids:
            # Check if record exists in all databases
            # ... verification logic ...
            pass
        
        if not inconsistencies:
            print("✅ All sampled records are consistent across databases")
        else:
            print(f"⚠️ Found {len(inconsistencies)} inconsistencies")
        
        return counts, inconsistencies
```

## Best Practices Summary

1. **Always validate data** before and after migration
2. **Maintain audit trails** for compliance and debugging
3. **Use batch processing** for large datasets
4. **Implement reconciliation** to verify data integrity
5. **Handle errors gracefully** with retry logic
6. **Test migrations** on sample data first
7. **Monitor progress** with callbacks and logging
8. **Document transformations** for future reference
9. **Use appropriate data types** (e.g., Decimal for financial data)
10. **Plan for rollback** scenarios