import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    """
    A comprehensive data cleaning pipeline that handles common data quality issues.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the pipeline with a DataFrame."""
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_report = []
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaningPipeline':
        """Remove duplicate rows."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        
        self.cleaning_report.append(f"Removed {removed} duplicate rows")
        logger.info(f"Removed {removed} duplicate rows")
        return self
    
    def handle_missing_values(self, strategy: Dict[str, str]) -> 'DataCleaningPipeline':
        """
        Handle missing values with different strategies per column.
        
        Strategies: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'constant'
        Example: {'age': 'mean', 'name': 'drop', 'category': 'mode'}
        """
        for col, method in strategy.items():
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
                
            missing_count = self.df[col].isna().sum()
            
            if method == 'drop':
                self.df = self.df.dropna(subset=[col])
            elif method == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif method == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif method == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif method == 'forward_fill':
                self.df[col].fillna(method='ffill', inplace=True)
            elif method == 'backward_fill':
                self.df[col].fillna(method='bfill', inplace=True)
            elif isinstance(method, tuple) and method[0] == 'constant':
                self.df[col].fillna(method[1], inplace=True)
            
            self.cleaning_report.append(f"Handled {missing_count} missing values in '{col}' using {method}")
            
        return self
    
    def remove_outliers(self, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaningPipeline':
        """
        Remove outliers using IQR or Z-score method.
        
        Args:
            columns: List of column names to check for outliers
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier (default 1.5) or Z-score threshold (default 3)
        """
        initial_rows = len(self.df)
        
        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
        
        removed = initial_rows - len(self.df)
        self.cleaning_report.append(f"Removed {removed} outliers using {method} method")
        logger.info(f"Removed {removed} outliers")
        return self
    
    def standardize_text(self, columns: List[str], lowercase: bool = True, 
                        remove_special_chars: bool = False, strip: bool = True) -> 'DataCleaningPipeline':
        """Standardize text columns."""
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if strip:
                self.df[col] = self.df[col].astype(str).str.strip()
            if lowercase:
                self.df[col] = self.df[col].astype(str).str.lower()
            if remove_special_chars:
                self.df[col] = self.df[col].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        
        self.cleaning_report.append(f"Standardized text in columns: {', '.join(columns)}")
        return self
    
    def convert_data_types(self, type_mapping: Dict[str, str]) -> 'DataCleaningPipeline':
        """
        Convert column data types.
        
        Example: {'age': 'int', 'price': 'float', 'date': 'datetime'}
        """
        for col, dtype in type_mapping.items():
            if col not in self.df.columns:
                continue
                
            try:
                if dtype == 'datetime':
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                else:
                    self.df[col] = self.df[col].astype(dtype)
                self.cleaning_report.append(f"Converted '{col}' to {dtype}")
            except Exception as e:
                logger.error(f"Failed to convert '{col}' to {dtype}: {str(e)}")
        
        return self
    
    def remove_columns(self, columns: List[str]) -> 'DataCleaningPipeline':
        """Remove specified columns."""
        existing_cols = [c for c in columns if c in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        self.cleaning_report.append(f"Removed columns: {', '.join(existing_cols)}")
        return self
    
    def rename_columns(self, mapping: Dict[str, str]) -> 'DataCleaningPipeline':
        """Rename columns."""
        self.df = self.df.rename(columns=mapping)
        self.cleaning_report.append(f"Renamed {len(mapping)} columns")
        return self
    
    def apply_custom_function(self, column: str, func: Callable, new_column: Optional[str] = None) -> 'DataCleaningPipeline':
        """Apply a custom function to a column."""
        target_col = new_column if new_column else column
        self.df[target_col] = self.df[column].apply(func)
        self.cleaning_report.append(f"Applied custom function to '{column}'")
        return self
    
    def validate_email(self, column: str, remove_invalid: bool = True) -> 'DataCleaningPipeline':
        """Validate email addresses."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        valid_mask = self.df[column].astype(str).str.match(email_pattern, na=False)
        
        invalid_count = (~valid_mask).sum()
        
        if remove_invalid:
            self.df = self.df[valid_mask]
            self.cleaning_report.append(f"Removed {invalid_count} rows with invalid emails in '{column}'")
        else:
            self.df[f'{column}_valid'] = valid_mask
            self.cleaning_report.append(f"Added validation column for '{column}'")
        
        return self
    
    def reset_index(self) -> 'DataCleaningPipeline':
        """Reset DataFrame index."""
        self.df = self.df.reset_index(drop=True)
        return self
    
    def get_cleaning_report(self) -> List[str]:
        """Get the cleaning report."""
        return self.cleaning_report
    
    def get_summary(self) -> Dict:
        """Get summary statistics before and after cleaning."""
        return {
            'original_shape': self.original_df.shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': self.original_df.shape[0] - self.df.shape[0],
            'columns_removed': self.original_df.shape[1] - self.df.shape[1],
            'missing_values_before': self.original_df.isna().sum().sum(),
            'missing_values_after': self.df.isna().sum().sum(),
            'cleaning_steps': len(self.cleaning_report)
        }
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        return self.df


# Example usage
if __name__ == "__main__":
    # Create sample dirty data
    data = {
        'name': ['John Doe', 'jane smith', '  Bob Johnson  ', 'Alice Wong', 'John Doe', None],
        'age': [25, 30, 150, 28, 25, 35],  # 150 is an outlier
        'email': ['john@example.com', 'invalid-email', 'bob@test.com', 'alice@company.co', 'john@example.com', 'test@test.com'],
        'salary': [50000, 60000, None, 70000, 50000, 80000],
        'join_date': ['2020-01-15', '2019-05-20', '2021-03-10', None, '2020-01-15', '2022-07-01']
    }
    
    df = pd.DataFrame(data)
    
    print("Original Data:")
    print(df)
    print("\n" + "="*80 + "\n")
    
    # Create and run the cleaning pipeline
    pipeline = DataCleaningPipeline(df)
    
    cleaned_df = (pipeline
                  .remove_duplicates()
                  .handle_missing_values({
                      'salary': 'median',
                      'join_date': 'drop',
                      'name': 'drop'
                  })
                  .remove_outliers(['age'], method='iqr')
                  .standardize_text(['name'], lowercase=True, strip=True)
                  .validate_email('email', remove_invalid=True)
                  .convert_data_types({'join_date': 'datetime'})
                  .reset_index()
                  .get_cleaned_data())
    
    print("Cleaned Data:")
    print(cleaned_df)
    print("\n" + "="*80 + "\n")
    
    print("Cleaning Report:")
    for step in pipeline.get_cleaning_report():
        print(f"  • {step}")
    
    print("\n" + "="*80 + "\n")
    
    print("Summary Statistics:")
    summary = pipeline.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")