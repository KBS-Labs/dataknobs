# Pandas Workflow Examples

This document provides comprehensive examples of using pandas DataFrames with the DataKnobs Data package for analytics, ETL pipelines, and data science workflows.

## Example 1: Real-Time Analytics Dashboard

### Scenario
Build a real-time analytics dashboard for e-commerce data using pandas integration.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataknobs_data.pandas import DataFrameConverter, BatchOperations, ChunkedProcessor
from dataknobs_data import MemoryDatabase, Query, Record
from typing import Dict, List, Optional
import json

class EcommerceAnalyticsDashboard:
    """Real-time analytics dashboard for e-commerce data"""
    
    def __init__(self):
        self.db = MemoryDatabase()
        self.batch_ops = BatchOperations(self.db)
        self.converter = DataFrameConverter()
        self.cache = {}  # Simple cache for computed metrics
    
    def generate_sample_data(self, num_orders: int = 10000):
        """Generate realistic e-commerce data"""
        
        np.random.seed(42)
        
        # Configuration
        start_date = datetime.now() - timedelta(days=90)
        products = [
            {"id": "PROD-001", "name": "Laptop", "category": "Electronics", "price": 999.99},
            {"id": "PROD-002", "name": "Mouse", "category": "Accessories", "price": 29.99},
            {"id": "PROD-003", "name": "Keyboard", "category": "Accessories", "price": 79.99},
            {"id": "PROD-004", "name": "Monitor", "category": "Electronics", "price": 399.99},
            {"id": "PROD-005", "name": "Headphones", "category": "Audio", "price": 149.99},
            {"id": "PROD-006", "name": "Webcam", "category": "Accessories", "price": 89.99},
            {"id": "PROD-007", "name": "Tablet", "category": "Electronics", "price": 599.99},
            {"id": "PROD-008", "name": "Phone", "category": "Electronics", "price": 799.99},
        ]
        
        countries = ["US", "UK", "DE", "FR", "JP", "CA", "AU", "BR"]
        customer_segments = ["New", "Returning", "VIP", "Inactive"]
        payment_methods = ["Credit Card", "PayPal", "Bank Transfer", "Crypto"]
        
        orders = []
        for i in range(num_orders):
            # Generate temporal pattern (more orders on weekends and evenings)
            order_date = start_date + timedelta(
                days=np.random.randint(0, 90),
                hours=np.random.choice([9, 10, 11, 14, 15, 16, 19, 20, 21]),
                minutes=np.random.randint(0, 60)
            )
            
            # Customer info
            customer_id = f"CUST-{np.random.randint(1000, 5000):04d}"
            customer_segment = np.random.choice(customer_segments, 
                                              p=[0.3, 0.5, 0.1, 0.1])
            
            # Order composition
            num_items = np.random.poisson(2) + 1  # Poisson distribution for item count
            order_products = np.random.choice(products, min(num_items, len(products)), replace=False)
            
            # Calculate order value
            subtotal = sum(p["price"] * np.random.randint(1, 4) for p in order_products)
            tax_rate = 0.08 if np.random.choice(countries) == "US" else 0.20
            shipping = 0 if subtotal > 100 else 9.99
            total = subtotal + (subtotal * tax_rate) + shipping
            
            # Conversion metrics
            session_duration = np.random.gamma(2, 2) * 60  # In seconds
            pages_viewed = np.random.poisson(5) + 1
            
            order = {
                "order_id": f"ORD-{i+1:06d}",
                "order_date": order_date.isoformat(),
                "customer_id": customer_id,
                "customer_segment": customer_segment,
                "country": np.random.choice(countries),
                "products": json.dumps([p["id"] for p in order_products]),
                "product_names": ", ".join([p["name"] for p in order_products]),
                "categories": ", ".join(list(set([p["category"] for p in order_products]))),
                "num_items": num_items,
                "subtotal": round(subtotal, 2),
                "tax": round(subtotal * tax_rate, 2),
                "shipping": round(shipping, 2),
                "total": round(total, 2),
                "payment_method": np.random.choice(payment_methods),
                "session_duration": round(session_duration, 2),
                "pages_viewed": pages_viewed,
                "conversion_rate": round(1 / pages_viewed * 100, 2),
                "is_mobile": np.random.choice([True, False], p=[0.6, 0.4]),
                "has_discount": np.random.choice([True, False], p=[0.3, 0.7]),
                "discount_amount": round(subtotal * 0.1, 2) if np.random.random() < 0.3 else 0
            }
            
            orders.append(order)
        
        # Convert to DataFrame and load into database
        df = pd.DataFrame(orders)
        df["order_date"] = pd.to_datetime(df["order_date"])
        
        # Bulk insert
        result = self.batch_ops.bulk_insert_dataframe(df)
        print(f"Loaded {result['inserted']} orders into database")
        
        return df
    
    def calculate_key_metrics(self) -> Dict:
        """Calculate key business metrics"""
        
        # Query all orders
        df = self.batch_ops.query_as_dataframe(Query())
        df["order_date"] = pd.to_datetime(df["order_date"])
        
        # Current period (last 30 days)
        current_date = datetime.now()
        current_start = current_date - timedelta(days=30)
        current_df = df[df["order_date"] >= current_start]
        
        # Previous period (30 days before that)
        previous_start = current_start - timedelta(days=30)
        previous_df = df[(df["order_date"] >= previous_start) & 
                        (df["order_date"] < current_start)]
        
        # Calculate metrics
        metrics = {
            "current_period": {
                "total_revenue": float(current_df["total"].sum()),
                "total_orders": len(current_df),
                "average_order_value": float(current_df["total"].mean()),
                "unique_customers": current_df["customer_id"].nunique(),
                "conversion_rate": float(current_df["conversion_rate"].mean()),
                "mobile_percentage": float((current_df["is_mobile"].sum() / len(current_df)) * 100)
            },
            "previous_period": {
                "total_revenue": float(previous_df["total"].sum()),
                "total_orders": len(previous_df),
                "average_order_value": float(previous_df["total"].mean()),
                "unique_customers": previous_df["customer_id"].nunique()
            }
        }
        
        # Calculate growth rates
        metrics["growth"] = {
            "revenue_growth": ((metrics["current_period"]["total_revenue"] - 
                              metrics["previous_period"]["total_revenue"]) / 
                              metrics["previous_period"]["total_revenue"] * 100),
            "order_growth": ((metrics["current_period"]["total_orders"] - 
                            metrics["previous_period"]["total_orders"]) / 
                            metrics["previous_period"]["total_orders"] * 100),
            "aov_growth": ((metrics["current_period"]["average_order_value"] - 
                          metrics["previous_period"]["average_order_value"]) / 
                          metrics["previous_period"]["average_order_value"] * 100)
        }
        
        return metrics
    
    def analyze_product_performance(self) -> pd.DataFrame:
        """Analyze product performance metrics"""
        
        df = self.batch_ops.query_as_dataframe(Query())
        
        # Explode products (since they're stored as JSON)
        df["product_list"] = df["products"].apply(json.loads)
        exploded = df.explode("product_list")
        
        # Group by product
        product_metrics = exploded.groupby("product_list").agg({
            "order_id": "count",
            "total": "sum",
            "customer_id": "nunique",
            "conversion_rate": "mean"
        }).rename(columns={
            "order_id": "units_sold",
            "total": "total_revenue",
            "customer_id": "unique_customers",
            "conversion_rate": "avg_conversion_rate"
        })
        
        # Calculate additional metrics
        product_metrics["revenue_per_customer"] = (
            product_metrics["total_revenue"] / product_metrics["unique_customers"]
        )
        
        # Rank products
        product_metrics["revenue_rank"] = product_metrics["total_revenue"].rank(
            ascending=False, method="dense"
        )
        
        return product_metrics.sort_values("total_revenue", ascending=False)
    
    def customer_segmentation_analysis(self) -> pd.DataFrame:
        """Perform RFM customer segmentation"""
        
        df = self.batch_ops.query_as_dataframe(Query())
        df["order_date"] = pd.to_datetime(df["order_date"])
        
        current_date = df["order_date"].max()
        
        # Calculate RFM metrics
        rfm = df.groupby("customer_id").agg({
            "order_date": lambda x: (current_date - x.max()).days,  # Recency
            "order_id": "count",  # Frequency
            "total": "sum"  # Monetary
        }).rename(columns={
            "order_date": "recency",
            "order_id": "frequency",
            "total": "monetary"
        })
        
        # Create RFM segments
        rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=["4", "3", "2", "1"])
        rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, 
                                 labels=["1", "2", "3", "4"])
        rfm["m_score"] = pd.qcut(rfm["monetary"], 4, labels=["1", "2", "3", "4"])
        
        # Combine scores
        rfm["rfm_segment"] = rfm["r_score"].astype(str) + \
                             rfm["f_score"].astype(str) + \
                             rfm["m_score"].astype(str)
        
        # Define segment names
        def segment_customers(row):
            if row["rfm_segment"] == "444":
                return "Champions"
            elif row["rfm_segment"] == "434":
                return "Loyal Customers"
            elif row["rfm_segment"] == "443":
                return "Potential Loyalists"
            elif row["rfm_segment"] == "344":
                return "New Customers"
            elif row["rfm_segment"] == "144":
                return "At Risk"
            elif row["rfm_segment"] == "111":
                return "Lost Customers"
            else:
                return "Regular"
        
        rfm["segment"] = rfm.apply(segment_customers, axis=1)
        
        # Calculate segment statistics
        segment_stats = rfm.groupby("segment").agg({
            "recency": "mean",
            "frequency": "mean",
            "monetary": "mean"
        }).round(2)
        
        segment_stats["customer_count"] = rfm.groupby("segment").size()
        segment_stats["revenue_contribution"] = (
            rfm.groupby("segment")["monetary"].sum() / rfm["monetary"].sum() * 100
        ).round(2)
        
        return segment_stats
    
    def time_series_analysis(self) -> Dict:
        """Perform time series analysis and forecasting"""
        
        df = self.batch_ops.query_as_dataframe(Query())
        df["order_date"] = pd.to_datetime(df["order_date"])
        
        # Daily aggregation
        daily_sales = df.groupby(df["order_date"].dt.date).agg({
            "total": "sum",
            "order_id": "count",
            "customer_id": "nunique"
        }).rename(columns={
            "total": "revenue",
            "order_id": "orders",
            "customer_id": "customers"
        })
        
        # Calculate moving averages
        daily_sales["ma_7"] = daily_sales["revenue"].rolling(window=7).mean()
        daily_sales["ma_30"] = daily_sales["revenue"].rolling(window=30).mean()
        
        # Calculate growth rates
        daily_sales["revenue_growth"] = daily_sales["revenue"].pct_change()
        
        # Detect anomalies using z-score
        from scipy import stats
        z_scores = np.abs(stats.zscore(daily_sales["revenue"].dropna()))
        threshold = 2.5
        anomalies = daily_sales.index[np.where(z_scores > threshold)[0]]
        
        # Simple trend analysis
        x = np.arange(len(daily_sales))
        y = daily_sales["revenue"].values
        z = np.polyfit(x, y, 1)
        trend = "increasing" if z[0] > 0 else "decreasing"
        
        # Weekly patterns
        df["weekday"] = df["order_date"].dt.day_name()
        weekly_pattern = df.groupby("weekday")["total"].mean().sort_values(ascending=False)
        
        # Hourly patterns
        df["hour"] = df["order_date"].dt.hour
        hourly_pattern = df.groupby("hour")["total"].mean().sort_values(ascending=False)
        
        return {
            "daily_stats": daily_sales.describe().to_dict(),
            "trend": trend,
            "trend_slope": float(z[0]),
            "anomaly_dates": [str(d) for d in anomalies],
            "best_day": weekly_pattern.index[0],
            "worst_day": weekly_pattern.index[-1],
            "peak_hour": int(hourly_pattern.index[0]),
            "current_ma7": float(daily_sales["ma_7"].iloc[-1]),
            "current_ma30": float(daily_sales["ma_30"].iloc[-1])
        }
    
    def cohort_analysis(self) -> pd.DataFrame:
        """Perform cohort analysis for customer retention"""
        
        df = self.batch_ops.query_as_dataframe(Query())
        df["order_date"] = pd.to_datetime(df["order_date"])
        
        # Get first purchase date for each customer
        df["first_purchase"] = df.groupby("customer_id")["order_date"].transform("min")
        
        # Create cohort period
        df["cohort_period"] = df["first_purchase"].dt.to_period("M")
        
        # Calculate periods between first purchase and order
        df["period_number"] = (df["order_date"].dt.to_period("M") - 
                               df["cohort_period"]).apply(lambda x: x.n)
        
        # Create cohort matrix
        cohort_data = df.groupby(["cohort_period", "period_number"]).agg(
            n_customers=("customer_id", "nunique")
        ).reset_index()
        
        # Pivot to create matrix
        cohort_matrix = cohort_data.pivot(index="cohort_period", 
                                          columns="period_number", 
                                          values="n_customers")
        
        # Calculate retention rates
        cohort_sizes = cohort_matrix.iloc[:, 0]
        retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100
        
        return retention_matrix.round(1)
    
    def create_dashboard_summary(self) -> Dict:
        """Create comprehensive dashboard summary"""
        
        print("Generating dashboard summary...")
        
        # Key metrics
        metrics = self.calculate_key_metrics()
        
        # Product performance
        product_performance = self.analyze_product_performance()
        top_products = product_performance.head(5)[["units_sold", "total_revenue"]].to_dict()
        
        # Customer segmentation
        customer_segments = self.customer_segmentation_analysis()
        
        # Time series analysis
        time_analysis = self.time_series_analysis()
        
        # Cohort analysis
        cohort_matrix = self.cohort_analysis()
        
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "key_metrics": metrics,
            "top_products": top_products,
            "customer_segments": customer_segments.to_dict(),
            "time_analysis": time_analysis,
            "retention_rate_month_1": float(cohort_matrix.iloc[-1, 1]) if len(cohort_matrix) > 0 else 0
        }
        
        return dashboard
    
    def export_reports(self):
        """Export analytics reports to various formats"""
        
        # Create reports directory
        import os
        os.makedirs("reports", exist_ok=True)
        
        # Export key metrics to Excel
        with pd.ExcelWriter("reports/analytics_report.xlsx") as writer:
            # Product performance
            product_df = self.analyze_product_performance()
            product_df.to_excel(writer, sheet_name="Product Performance")
            
            # Customer segmentation
            segment_df = self.customer_segmentation_analysis()
            segment_df.to_excel(writer, sheet_name="Customer Segments")
            
            # Cohort analysis
            cohort_df = self.cohort_analysis()
            cohort_df.to_excel(writer, sheet_name="Cohort Analysis")
        
        # Export to CSV for further analysis
        df = self.batch_ops.query_as_dataframe(Query())
        df.to_csv("reports/raw_orders.csv", index=False)
        
        # Export summary to JSON
        summary = self.create_dashboard_summary()
        with open("reports/dashboard_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("Reports exported successfully!")

# Usage example
def run_analytics_dashboard():
    dashboard = EcommerceAnalyticsDashboard()
    
    # Generate sample data
    dashboard.generate_sample_data(10000)
    
    # Create dashboard summary
    summary = dashboard.create_dashboard_summary()
    
    # Display key metrics
    print("\n" + "="*60)
    print("E-COMMERCE ANALYTICS DASHBOARD")
    print("="*60)
    
    current = summary["key_metrics"]["current_period"]
    growth = summary["key_metrics"]["growth"]
    
    print(f"\n📊 KEY METRICS (Last 30 Days)")
    print(f"  Revenue: ${current['total_revenue']:,.2f} ({growth['revenue_growth']:+.1f}%)")
    print(f"  Orders: {current['total_orders']:,} ({growth['order_growth']:+.1f}%)")
    print(f"  AOV: ${current['average_order_value']:.2f} ({growth['aov_growth']:+.1f}%)")
    print(f"  Customers: {current['unique_customers']:,}")
    print(f"  Conversion: {current['conversion_rate']:.2f}%")
    print(f"  Mobile: {current['mobile_percentage']:.1f}%")
    
    print(f"\n📈 TRENDS")
    print(f"  Overall trend: {summary['time_analysis']['trend']}")
    print(f"  Best day: {summary['time_analysis']['best_day']}")
    print(f"  Peak hour: {summary['time_analysis']['peak_hour']}:00")
    
    # Export reports
    dashboard.export_reports()

if __name__ == "__main__":
    run_analytics_dashboard()
```

## Example 2: Machine Learning Pipeline

### Scenario
Build a machine learning pipeline using pandas integration for feature engineering and model training.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from dataknobs_data.pandas import DataFrameConverter, BatchOperations
from dataknobs_data import MemoryDatabase, Query, Record
from dataknobs_data.validation import Schema, Range
import joblib

class MLPipeline:
    """Machine learning pipeline with DataKnobs integration"""
    
    def __init__(self):
        self.db = MemoryDatabase()
        self.batch_ops = BatchOperations(self.db)
        self.converter = DataFrameConverter()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and prepare training data"""
        
        # Query training data from database
        df = self.batch_ops.query_as_dataframe(Query())
        
        # Basic data cleaning
        df = df.dropna()
        df = df.drop_duplicates()
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform feature engineering"""
        
        print("Performing feature engineering...")
        
        # Temporal features
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["month"] = df["timestamp"].dt.month
            df["quarter"] = df["timestamp"].dt.quarter
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["is_business_hour"] = df["hour"].between(9, 17).astype(int)
        
        # Numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Log transformation for skewed features
            if df[col].skew() > 1:
                df[f"{col}_log"] = np.log1p(df[col])
            
            # Polynomial features
            if col in ["age", "income", "price"]:
                df[f"{col}_squared"] = df[col] ** 2
                df[f"{col}_sqrt"] = np.sqrt(df[col])
        
        # Interaction features
        if "quantity" in df.columns and "price" in df.columns:
            df["total_value"] = df["quantity"] * df["price"]
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=["object"]).columns
        
        for col in categorical_cols:
            # Frequency encoding for high cardinality
            if df[col].nunique() > 10:
                freq_encoding = df[col].value_counts().to_dict()
                df[f"{col}_frequency"] = df[col].map(freq_encoding)
            
            # Target encoding (simplified - in practice use cross-validation)
            if "target" in df.columns:
                target_means = df.groupby(col)["target"].mean().to_dict()
                df[f"{col}_target_mean"] = df[col].map(target_means)
        
        # Aggregation features
        if "customer_id" in df.columns:
            customer_agg = df.groupby("customer_id").agg({
                "order_id": "count",
                "total": ["sum", "mean", "std"],
                "timestamp": lambda x: (x.max() - x.min()).days
            })
            customer_agg.columns = ["_".join(col).strip() for col in customer_agg.columns]
            df = df.merge(customer_agg, left_on="customer_id", right_index=True, how="left")
        
        return df
    
    def train_classification_model(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Train a classification model"""
        
        print(f"Training classification model for {target_col}...")
        
        # Prepare features and target
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_col] = scaler
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        self.models[target_col] = model
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        # Store results in database
        results = {
            "model_type": "classification",
            "target": target_col,
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "top_features": feature_importance.head(10).to_dict("records"),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        # Save model metadata
        self.db.insert(Record(data={
            "type": "model_metadata",
            "model_id": f"clf_{target_col}",
            **results
        }))
        
        return results
    
    def train_regression_model(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Train a regression model"""
        
        print(f"Training regression model for {target_col}...")
        
        # Prepare features and target
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_col] = scaler
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        self.models[target_col] = model
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Feature importance
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        # Store results
        results = {
            "model_type": "regression",
            "target": target_col,
            "mae": float(mae),
            "mape": float(mape),
            "r2_score": float(model.score(X_test_scaled, y_test)),
            "top_features": feature_importance.head(10).to_dict("records"),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        # Save model metadata
        self.db.insert(Record(data={
            "type": "model_metadata",
            "model_id": f"reg_{target_col}",
            **results
        }))
        
        return results
    
    def batch_predict(self, df: pd.DataFrame, model_id: str) -> pd.DataFrame:
        """Perform batch predictions"""
        
        print(f"Performing batch predictions with model {model_id}...")
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        scaler = self.scalers.get(model_id)
        
        # Prepare features
        X = df.copy()
        
        # Apply encoding
        for col, encoder in self.encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Add predictions to dataframe
        df["prediction"] = predictions
        
        # If classification, add probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)
            for i, class_label in enumerate(model.classes_):
                df[f"prob_{class_label}"] = probabilities[:, i]
        
        return df
    
    def save_models(self, path: str = "models"):
        """Save trained models to disk"""
        
        import os
        os.makedirs(path, exist_ok=True)
        
        for model_id, model in self.models.items():
            joblib.dump(model, f"{path}/{model_id}.joblib")
            
        for scaler_id, scaler in self.scalers.items():
            joblib.dump(scaler, f"{path}/scaler_{scaler_id}.joblib")
            
        for encoder_id, encoder in self.encoders.items():
            joblib.dump(encoder, f"{path}/encoder_{encoder_id}.joblib")
        
        print(f"Models saved to {path}/")
    
    def load_models(self, path: str = "models"):
        """Load models from disk"""
        
        import os
        
        for file in os.listdir(path):
            if file.endswith(".joblib"):
                if file.startswith("scaler_"):
                    model_id = file.replace("scaler_", "").replace(".joblib", "")
                    self.scalers[model_id] = joblib.load(f"{path}/{file}")
                elif file.startswith("encoder_"):
                    col_name = file.replace("encoder_", "").replace(".joblib", "")
                    self.encoders[col_name] = joblib.load(f"{path}/{file}")
                else:
                    model_id = file.replace(".joblib", "")
                    self.models[model_id] = joblib.load(f"{path}/{file}")
        
        print(f"Loaded {len(self.models)} models")

# Usage example
def run_ml_pipeline():
    pipeline = MLPipeline()
    
    # Generate sample data
    sample_data = generate_sample_ml_data()
    pipeline.batch_ops.bulk_insert_dataframe(sample_data)
    
    # Load and prepare data
    df = pipeline.load_training_data()
    
    # Feature engineering
    df_engineered = pipeline.feature_engineering(df)
    
    # Train models
    if "churn" in df_engineered.columns:
        classification_results = pipeline.train_classification_model(df_engineered, "churn")
        print(f"Classification model results: {classification_results}")
    
    if "lifetime_value" in df_engineered.columns:
        regression_results = pipeline.train_regression_model(df_engineered, "lifetime_value")
        print(f"Regression model results: {regression_results}")
    
    # Save models
    pipeline.save_models()
    
    # Batch predictions
    new_data = generate_sample_ml_data(100)
    predictions = pipeline.batch_predict(new_data, "churn")
    
    # Store predictions in database
    pipeline.batch_ops.bulk_insert_dataframe(predictions)

def generate_sample_ml_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample data for ML pipeline"""
    
    np.random.seed(42)
    
    data = {
        "customer_id": [f"CUST-{i:04d}" for i in range(n_samples)],
        "age": np.random.randint(18, 80, n_samples),
        "income": np.random.lognormal(10.5, 0.6, n_samples),
        "tenure_months": np.random.randint(1, 120, n_samples),
        "products_purchased": np.random.poisson(3, n_samples),
        "support_calls": np.random.poisson(1, n_samples),
        "payment_delay_days": np.random.exponential(2, n_samples),
        "satisfaction_score": np.random.uniform(1, 5, n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "region": np.random.choice(["North", "South", "East", "West"], n_samples),
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H")
    }
    
    df = pd.DataFrame(data)
    
    # Add target variables
    # Churn (classification)
    churn_probability = 1 / (1 + np.exp(
        -(-3 + 0.02 * df["support_calls"] - 0.01 * df["tenure_months"] + 
          0.1 * df["payment_delay_days"] - 0.5 * df["satisfaction_score"])
    ))
    df["churn"] = (np.random.random(n_samples) < churn_probability).astype(int)
    
    # Lifetime value (regression)
    df["lifetime_value"] = (
        df["income"] * 0.01 * df["tenure_months"] * 
        df["products_purchased"] * df["satisfaction_score"] / 5 +
        np.random.normal(0, 1000, n_samples)
    ).clip(0, None)
    
    return df
```

## Example 3: ETL Pipeline with Data Quality Monitoring

### Scenario
Build a comprehensive ETL pipeline with data quality monitoring and automated reporting.

```python
from dataknobs_data.pandas import DataFrameConverter, BatchOperations, ChunkedProcessor
from dataknobs_data import MemoryDatabase, Query
from dataknobs_data.validation import Schema, Range, Pattern
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime

class ETLPipeline:
    """Comprehensive ETL pipeline with data quality monitoring"""
    
    def __init__(self):
        self.source_db = MemoryDatabase()
        self.staging_db = MemoryDatabase()
        self.warehouse_db = MemoryDatabase()
        self.batch_ops = BatchOperations(self.warehouse_db)
        self.converter = DataFrameConverter()
        self.quality_metrics = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_from_sources(self) -> Dict[str, pd.DataFrame]:
        """Extract data from multiple sources"""
        
        self.logger.info("Starting data extraction...")
        
        extracted_data = {}
        
        # Extract from CSV files
        try:
            csv_processor = ChunkedProcessor(chunk_size=1000)
            csv_data = []
            
            def process_csv_chunk(chunk):
                # Clean and validate chunk
                chunk = chunk.dropna(subset=["id"])
                chunk["source"] = "csv"
                chunk["extracted_at"] = datetime.now()
                return chunk
            
            # Process large CSV in chunks
            for chunk in csv_processor.read_csv_chunked("data/sales.csv", process_csv_chunk):
                csv_data.append(chunk)
            
            if csv_data:
                extracted_data["csv_sales"] = pd.concat(csv_data, ignore_index=True)
                self.logger.info(f"Extracted {len(extracted_data['csv_sales'])} records from CSV")
        except Exception as e:
            self.logger.error(f"CSV extraction failed: {e}")
        
        # Extract from database
        try:
            db_data = self.batch_ops.query_as_dataframe(Query())
            if not db_data.empty:
                db_data["source"] = "database"
                db_data["extracted_at"] = datetime.now()
                extracted_data["database"] = db_data
                self.logger.info(f"Extracted {len(db_data)} records from database")
        except Exception as e:
            self.logger.error(f"Database extraction failed: {e}")
        
        # Extract from API (simulated)
        try:
            api_data = self.extract_from_api()
            if not api_data.empty:
                api_data["source"] = "api"
                api_data["extracted_at"] = datetime.now()
                extracted_data["api"] = api_data
                self.logger.info(f"Extracted {len(api_data)} records from API")
        except Exception as e:
            self.logger.error(f"API extraction failed: {e}")
        
        return extracted_data
    
    def extract_from_api(self) -> pd.DataFrame:
        """Simulate API data extraction"""
        
        # Simulated API data
        n_records = 500
        data = {
            "transaction_id": [f"TXN-{i:06d}" for i in range(n_records)],
            "amount": np.random.exponential(100, n_records),
            "currency": np.random.choice(["USD", "EUR", "GBP"], n_records),
            "status": np.random.choice(["completed", "pending", "failed"], 
                                      n_records, p=[0.8, 0.15, 0.05]),
            "timestamp": pd.date_range("2024-01-01", periods=n_records, freq="15min")
        }
        
        return pd.DataFrame(data)
    
    def transform_data(self, extracted_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Transform and clean extracted data"""
        
        self.logger.info("Starting data transformation...")
        
        transformed_frames = []
        
        for source_name, df in extracted_data.items():
            self.logger.info(f"Transforming {source_name} data...")
            
            # Common transformations
            df = self.apply_common_transformations(df)
            
            # Source-specific transformations
            if source_name == "csv_sales":
                df = self.transform_sales_data(df)
            elif source_name == "database":
                df = self.transform_database_data(df)
            elif source_name == "api":
                df = self.transform_api_data(df)
            
            # Add data quality metrics
            quality_metrics = self.assess_data_quality(df, source_name)
            self.quality_metrics.append(quality_metrics)
            
            transformed_frames.append(df)
        
        # Combine all transformed data
        if transformed_frames:
            combined_df = pd.concat(transformed_frames, ignore_index=True)
            
            # Remove duplicates
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["transaction_id"], keep="last")
            after_dedup = len(combined_df)
            
            self.logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
            
            # Final transformations
            combined_df = self.apply_business_rules(combined_df)
            
            return combined_df
        else:
            return pd.DataFrame()
    
    def apply_common_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply common transformations to all data sources"""
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        string_columns = df.select_dtypes(include=["object"]).columns
        df[string_columns] = df[string_columns].fillna("")
        
        # Standardize date formats
        date_columns = [col for col in df.columns if "date" in col or "time" in col]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
        
        # Remove extra whitespace
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].str.strip()
        
        return df
    
    def transform_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform sales-specific data"""
        
        # Calculate derived metrics
        if "quantity" in df.columns and "unit_price" in df.columns:
            df["total_amount"] = df["quantity"] * df["unit_price"]
        
        # Categorize sales
        if "total_amount" in df.columns:
            df["sale_category"] = pd.cut(df["total_amount"], 
                                         bins=[0, 100, 500, 1000, float('inf')],
                                         labels=["Small", "Medium", "Large", "Enterprise"])
        
        # Add fiscal period
        if "timestamp" in df.columns:
            df["fiscal_quarter"] = df["timestamp"].dt.to_period("Q")
            df["fiscal_year"] = df["timestamp"].dt.year
        
        return df
    
    def transform_database_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform database-specific data"""
        
        # Specific transformations for database data
        return df
    
    def transform_api_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform API-specific data"""
        
        # Currency conversion (simplified)
        conversion_rates = {"USD": 1.0, "EUR": 1.1, "GBP": 1.3}
        
        if "currency" in df.columns and "amount" in df.columns:
            df["amount_usd"] = df.apply(
                lambda x: x["amount"] * conversion_rates.get(x["currency"], 1.0),
                axis=1
            )
        
        return df
    
    def apply_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules and validations"""
        
        # Example business rules
        
        # Rule 1: Flag suspicious transactions
        if "amount" in df.columns:
            df["suspicious"] = (df["amount"] > df["amount"].quantile(0.99)) | \
                              (df["amount"] < 0)
        
        # Rule 2: Validate status transitions
        valid_statuses = ["pending", "processing", "completed", "failed", "cancelled"]
        if "status" in df.columns:
            df["status"] = df["status"].apply(
                lambda x: x if x in valid_statuses else "unknown"
            )
        
        # Rule 3: Add audit fields
        df["processed_at"] = datetime.now()
        df["etl_version"] = "1.0.0"
        
        return df
    
    def assess_data_quality(self, df: pd.DataFrame, source_name: str) -> Dict:
        """Assess data quality metrics"""
        
        metrics = {
            "source": source_name,
            "timestamp": datetime.now().isoformat(),
            "record_count": len(df),
            "column_count": len(df.columns),
            "completeness": {},
            "validity": {},
            "uniqueness": {},
            "consistency": {}
        }
        
        # Completeness metrics
        for col in df.columns:
            null_count = df[col].isnull().sum()
            metrics["completeness"][col] = {
                "non_null_count": len(df) - null_count,
                "null_count": null_count,
                "completeness_rate": (len(df) - null_count) / len(df) * 100
            }
        
        # Validity metrics
        if "email" in df.columns:
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            valid_emails = df["email"].str.match(email_pattern).sum()
            metrics["validity"]["email"] = {
                "valid_count": valid_emails,
                "invalid_count": len(df) - valid_emails,
                "validity_rate": valid_emails / len(df) * 100
            }
        
        # Uniqueness metrics
        for col in ["transaction_id", "order_id", "customer_id"]:
            if col in df.columns:
                unique_count = df[col].nunique()
                metrics["uniqueness"][col] = {
                    "unique_count": unique_count,
                    "duplicate_count": len(df) - unique_count,
                    "uniqueness_rate": unique_count / len(df) * 100
                }
        
        # Consistency metrics
        if "amount" in df.columns:
            metrics["consistency"]["amount"] = {
                "min": float(df["amount"].min()),
                "max": float(df["amount"].max()),
                "mean": float(df["amount"].mean()),
                "std": float(df["amount"].std()),
                "negative_count": (df["amount"] < 0).sum()
            }
        
        return metrics
    
    def load_to_warehouse(self, df: pd.DataFrame) -> Dict:
        """Load transformed data to data warehouse"""
        
        self.logger.info("Loading data to warehouse...")
        
        if df.empty:
            self.logger.warning("No data to load")
            return {"status": "skipped", "records_loaded": 0}
        
        # Validate before loading
        validation_schema = (Schema("WarehouseRecord")
            .field("transaction_id", "STRING", required=True)
            .field("amount", "FLOAT", constraints=[Range(min=0)])
            .field("processed_at", "STRING", required=True)
        )
        
        # Convert to records for validation
        records = self.converter.dataframe_to_records(df)
        valid_records = []
        invalid_records = []
        
        for record in records:
            result = validation_schema.validate(record)
            if result.valid:
                valid_records.append(record)
            else:
                invalid_records.append(record)
        
        # Load valid records
        if valid_records:
            valid_df = self.converter.records_to_dataframe(valid_records)
            load_result = self.batch_ops.bulk_insert_dataframe(valid_df)
            
            self.logger.info(f"Loaded {load_result['inserted']} records to warehouse")
            
            # Archive invalid records
            if invalid_records:
                invalid_df = self.converter.records_to_dataframe(invalid_records)
                invalid_df.to_csv("data/invalid_records.csv", index=False)
                self.logger.warning(f"Archived {len(invalid_records)} invalid records")
            
            return {
                "status": "success",
                "records_loaded": load_result["inserted"],
                "records_failed": len(invalid_records),
                "duration": load_result["duration"]
            }
        else:
            return {"status": "failed", "records_loaded": 0}
    
    def generate_quality_report(self) -> pd.DataFrame:
        """Generate data quality report"""
        
        if not self.quality_metrics:
            return pd.DataFrame()
        
        # Aggregate quality metrics
        report_data = []
        
        for metrics in self.quality_metrics:
            source = metrics["source"]
            
            # Calculate overall scores
            completeness_scores = [m["completeness_rate"] 
                                  for m in metrics["completeness"].values()]
            avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
            
            validity_scores = [m["validity_rate"] 
                             for m in metrics["validity"].values()]
            avg_validity = np.mean(validity_scores) if validity_scores else 100
            
            uniqueness_scores = [m["uniqueness_rate"] 
                               for m in metrics["uniqueness"].values()]
            avg_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 100
            
            report_data.append({
                "source": source,
                "timestamp": metrics["timestamp"],
                "record_count": metrics["record_count"],
                "completeness_score": avg_completeness,
                "validity_score": avg_validity,
                "uniqueness_score": avg_uniqueness,
                "overall_quality_score": np.mean([avg_completeness, avg_validity, avg_uniqueness])
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_df.to_csv("data/quality_report.csv", index=False)
        
        return report_df
    
    def run_pipeline(self) -> Dict:
        """Run the complete ETL pipeline"""
        
        self.logger.info("Starting ETL pipeline...")
        start_time = datetime.now()
        
        try:
            # Extract
            extracted_data = self.extract_from_sources()
            
            # Transform
            transformed_data = self.transform_data(extracted_data)
            
            # Load
            load_result = self.load_to_warehouse(transformed_data)
            
            # Generate quality report
            quality_report = self.generate_quality_report()
            
            # Calculate pipeline metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            pipeline_result = {
                "status": "success",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "records_processed": load_result.get("records_loaded", 0),
                "quality_score": quality_report["overall_quality_score"].mean() if not quality_report.empty else 0,
                "load_result": load_result
            }
            
            self.logger.info(f"ETL pipeline completed successfully in {duration:.2f} seconds")
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Usage example
def run_etl_pipeline():
    pipeline = ETLPipeline()
    
    # Run the pipeline
    result = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("ETL PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Duration: {result.get('duration_seconds', 0):.2f} seconds")
    print(f"Records Processed: {result.get('records_processed', 0):,}")
    print(f"Quality Score: {result.get('quality_score', 0):.1f}%")
    
    # Display quality report
    quality_report = pipeline.generate_quality_report()
    if not quality_report.empty:
        print("\nData Quality Report:")
        print(quality_report.to_string(index=False))

if __name__ == "__main__":
    run_etl_pipeline()
```

## Best Practices Summary

1. **Use chunked processing** for large datasets to manage memory
2. **Implement data quality checks** at each stage of the pipeline
3. **Cache frequently accessed data** to improve performance
4. **Use appropriate pandas data types** to optimize memory usage
5. **Parallelize operations** when processing independent data chunks
6. **Monitor pipeline performance** and track key metrics
7. **Implement error handling** and recovery mechanisms
8. **Document transformations** for maintainability
9. **Version your pipelines** to track changes over time
10. **Create comprehensive logging** for debugging and auditing