"""
Real-world example: Cleaning an e-commerce orders dataset
This demonstrates how to use the pipeline for a production scenario
"""

import pandas as pd
import numpy as np
from data_cleaning_pipeline import DataCleaningPipeline
from datetime import datetime, timedelta

# Create realistic sample data with common data quality issues
def create_sample_ecommerce_data():
    """Generate sample e-commerce data with intentional quality issues"""
    np.random.seed(42)
    n_records = 1000
    
    data = {
        'order_id': range(1, n_records + 1),
        'customer_name': [
            '  John Doe  ', 'jane SMITH', 'Bob Johnson', 'ALICE WONG',
            'John Doe', 'Charlie Brown', None, 'david LEE '
        ] * 125,
        'customer_email': [
            'john@example.com', 'invalid-email', 'bob@test.com',
            'alice@company.co', 'john@example.com', 'charlie@email.com',
            None, 'david@site.org'
        ] * 125,
        'order_date': pd.date_range('2023-01-01', periods=n_records, freq='6H'),
        'product_category': np.random.choice(
            ['Electronics', 'Clothing', 'Books', 'Home & Garden', None],
            n_records
        ),
        'quantity': np.random.randint(1, 10, n_records),
        'price': np.concatenate([
            np.random.uniform(10, 500, 980),
            [9999, 0.01, -50, 10000, 0, 15000, -100, 8888, 0.001, 99999],  # Outliers
            np.random.uniform(20, 300, 10)
        ]),
        'discount_percent': np.random.choice([0, 5, 10, 15, 20, None], n_records),
        'shipping_city': [
            'new york', 'LOS ANGELES', '  Chicago  ', 'Houston',
            'Phoenix!', 'Philadelphia@', 'San#Antonio', 'San Diego'
        ] * 125,
        'customer_phone': [
            '(555) 123-4567', '5551234567', '555-123-4567', None,
            '(555)1234567', '555.123.4567', 'invalid', '1234567890'
        ] * 125
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some duplicate orders
    duplicates = df.iloc[:20].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df


def main():
    print("="*80)
    print("E-COMMERCE DATA CLEANING PIPELINE - REAL WORLD EXAMPLE")
    print("="*80)
    print()
    
    # Generate sample data
    print("📊 Generating sample e-commerce data with quality issues...")
    df = create_sample_ecommerce_data()
    
    print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    print("Sample of raw data:")
    print(df.head(10))
    print()
    
    # Show data quality issues
    print("🔍 Data Quality Issues Detected:")
    print(f"  • Missing values: {df.isna().sum().sum()}")
    print(f"  • Duplicate rows: {df.duplicated().sum()}")
    print(f"  • Text formatting issues: Mixed case, extra spaces, special characters")
    print(f"  • Potential price outliers: {(df['price'] > 5000).sum()} records")
    print()
    
    # Initialize pipeline
    print("🔧 Starting Data Cleaning Pipeline...")
    print()
    
    pipeline = DataCleaningPipeline(df)
    
    # Execute cleaning steps
    cleaned_df = (pipeline
        # Step 1: Remove duplicate orders
        .remove_duplicates(subset=['customer_email', 'order_date', 'price'])
        
        # Step 2: Handle missing values
        .handle_missing_values({
            'customer_name': 'drop',           # Critical field
            'customer_email': 'drop',          # Critical field
            'product_category': 'mode',        # Fill with most common
            'discount_percent': ('constant', 0), # No discount = 0
            'customer_phone': ('constant', 'N/A')  # Optional field
        })
        
        # Step 3: Remove price and quantity outliers
        .remove_outliers(['price'], method='iqr', threshold=2.0)
        .remove_outliers(['quantity'], method='iqr', threshold=1.5)
        
        # Step 4: Standardize text fields
        .standardize_text(
            columns=['customer_name', 'shipping_city', 'product_category'],
            lowercase=False,  # Keep proper case for names
            strip=True,
            remove_special_chars=True
        )
        
        # Step 5: Lowercase cities for consistency
        .standardize_text(
            columns=['shipping_city'],
            lowercase=True,
            strip=True
        )
        
        # Step 6: Validate email addresses
        .validate_email('customer_email', remove_invalid=True)
        
        # Step 7: Convert data types
        .convert_data_types({
            'order_date': 'datetime',
            'quantity': 'int',
            'price': 'float',
            'discount_percent': 'float'
        })
        
        # Step 8: Clean up index
        .reset_index()
        .get_cleaned_data()
    )
    
    # Display cleaning report
    print("✅ CLEANING REPORT:")
    print("-" * 80)
    for i, step in enumerate(pipeline.get_cleaning_report(), 1):
        print(f"{i}. {step}")
    print()
    
    # Display summary statistics
    print("📈 SUMMARY STATISTICS:")
    print("-" * 80)
    summary = pipeline.get_summary()
    print(f"Original shape:          {summary['original_shape']}")
    print(f"Cleaned shape:           {summary['cleaned_shape']}")
    print(f"Rows removed:            {summary['rows_removed']} ({summary['rows_removed']/summary['original_shape'][0]*100:.1f}%)")
    print(f"Columns removed:         {summary['columns_removed']}")
    print(f"Missing values before:   {summary['missing_values_before']}")
    print(f"Missing values after:    {summary['missing_values_after']}")
    print(f"Data quality improvement: {(1 - summary['missing_values_after']/max(summary['missing_values_before'], 1))*100:.1f}%")
    print()
    
    # Display sample of cleaned data
    print("✨ Sample of cleaned data:")
    print(cleaned_df.head(10))
    print()
    
    # Calculate business metrics
    print("💰 BUSINESS METRICS (After Cleaning):")
    print("-" * 80)
    print(f"Total orders:            {len(cleaned_df)}")
    print(f"Total revenue:           ${cleaned_df['price'].sum():,.2f}")
    print(f"Average order value:     ${cleaned_df['price'].mean():.2f}")
    print(f"Unique customers:        {cleaned_df['customer_email'].nunique()}")
    print(f"Top category:            {cleaned_df['product_category'].mode()[0]}")
    print(f"Top city:                {cleaned_df['shipping_city'].mode()[0]}")
    print()
    
    # Save cleaned data
    output_file = 'cleaned_ecommerce_orders.csv'
    cleaned_df.to_csv(output_file, index=False)
    print(f"💾 Cleaned data saved to: {output_file}")
    print()
    
    # Data types after cleaning
    print("📋 Final Data Types:")
    print(cleaned_df.dtypes)
    print()
    
    print("="*80)
    print("✅ DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()