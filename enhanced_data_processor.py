import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import re
from sklearn.preprocessing import (LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, 
                                 QuantileTransformer, PowerTransformer)
# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

class IntelligentDataProcessor:
    """Enhanced data processor with smart cleaning strategies and no emoji output."""
    
    def __init__(self):
        self.processed_data = None
        self.original_data = None
        self.processing_log = []
        self.metadata = {}
        self.cleaning_strategies = {}
        self.feature_engineering_applied = []
        self.data_quality_metrics = {}
        
        # Smart cleaning configuration
        self.smart_config = {
            'missing_threshold_drop': 0.90,  # Drop columns with >90% missing
            'missing_threshold_impute': 0.50,  # Special imputation for >50% missing
            'outlier_threshold': 0.05,  # 5% outlier threshold
            'correlation_threshold': 0.95,  # Remove highly correlated features
            'cardinality_threshold': 1000,  # High cardinality threshold
            'preserve_missing_info': True,  # Preserve missing patterns if informative
        }
    
    def smart_clean_data(self, data: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """Intelligently clean data based on advanced heuristics."""
        self.original_data = data.copy()
        self.processed_data = data.copy()
        
        self.log("[START] Initiating intelligent data cleaning process")
        
        # Step 1: Initial data assessment
        self._assess_data_quality()
        
        # Step 2: Smart missing value handling
        self._smart_missing_value_treatment(target_column)
        
        # Step 3: Intelligent outlier detection and treatment
        self._smart_outlier_treatment()
        
        # Step 4: Advanced text cleaning (remove emojis, normalize)
        self._advanced_text_cleaning()
        
        # Step 5: Smart feature type optimization
        self._optimize_feature_types()
        
        # Step 6: Intelligent duplicate handling
        self._smart_duplicate_handling()
        
        # Step 7: Feature correlation analysis
        self._analyze_feature_correlations()
        
        # Step 8: Data consistency improvements
        self._improve_data_consistency()
        
        # Step 9: Final quality assessment
        self._final_quality_assessment()
        
        self.log("[COMPLETE] Intelligent data cleaning completed successfully")
        return self.processed_data
    
    def _assess_data_quality(self):
        """Comprehensive initial data quality assessment."""
        self.log("[ASSESS] Analyzing initial data quality")
        
        data = self.processed_data
        assessment = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_cells': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime64']).columns)
        }
        
        assessment['missing_percentage'] = (assessment['missing_cells'] / 
                                          (assessment['total_rows'] * assessment['total_columns'])) * 100
        
        self.data_quality_metrics['initial'] = assessment
        
        self.log(f"[INFO] Dataset: {assessment['total_rows']} rows, {assessment['total_columns']} columns")
        self.log(f"[INFO] Missing data: {assessment['missing_percentage']:.2f}%")
        self.log(f"[INFO] Duplicates: {assessment['duplicate_rows']} rows")
    
    def _smart_missing_value_treatment(self, target_column: str = None):
        """Intelligent missing value treatment with preservation of informative patterns."""
        self.log("[CLEAN] Applying smart missing value treatment")
        
        data = self.processed_data
        missing_analysis = {}
        
        for column in data.columns:
            if column == target_column:
                continue
                
            missing_count = data[column].isnull().sum()
            missing_pct = missing_count / len(data)
            
            if missing_count == 0:
                continue
                
            missing_analysis[column] = {
                'count': missing_count,
                'percentage': missing_pct,
                'strategy': self._determine_missing_strategy(column, missing_pct, target_column)
            }
        
        # Apply strategies
        for column, analysis in missing_analysis.items():
            strategy = analysis['strategy']
            
            if strategy == 'drop_column':
                data = data.drop(columns=[column])
                self.log(f"[DROP] Column '{column}' - {analysis['percentage']:.1%} missing")
                
            elif strategy == 'preserve_as_feature':
                # Create missing indicator and fill with special value
                data[f'{column}_was_missing'] = data[column].isnull().astype(int)
                if data[column].dtype == 'object':
                    data[column] = data[column].fillna('MISSING_VALUE')
                else:
                    data[column] = data[column].fillna(-999999)  # Distinctive value
                self.log(f"[PRESERVE] Column '{column}' - missing pattern preserved as feature")
                
            elif strategy == 'iterative_imputation':
                if data[column].dtype in ['int64', 'float64']:
                    imputer = IterativeImputer(random_state=42, max_iter=10)
                    data[column] = imputer.fit_transform(data[[column]]).flatten()
                    self.log(f"[IMPUTE] Column '{column}' - iterative imputation applied")
                else:
                    # For categorical, use mode
                    mode_val = data[column].mode().iloc[0] if len(data[column].mode()) > 0 else 'Unknown'
                    data[column] = data[column].fillna(mode_val)
                    self.log(f"[IMPUTE] Column '{column}' - mode imputation applied")
                    
            elif strategy == 'knn_imputation':
                if data[column].dtype in ['int64', 'float64']:
                    # Only use other numeric columns for KNN
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 1:
                        imputer = KNNImputer(n_neighbors=5)
                        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                        self.log(f"[IMPUTE] Column '{column}' - KNN imputation applied")
                    else:
                        # Fall back to median
                        data[column] = data[column].fillna(data[column].median())
                        
            elif strategy == 'simple_imputation':
                if data[column].dtype in ['int64', 'float64']:
                    # Use median for numeric
                    data[column] = data[column].fillna(data[column].median())
                    self.log(f"[IMPUTE] Column '{column}' - median imputation")
                else:
                    # Use mode for categorical
                    mode_val = data[column].mode().iloc[0] if len(data[column].mode()) > 0 else 'Unknown'
                    data[column] = data[column].fillna(mode_val)
                    self.log(f"[IMPUTE] Column '{column}' - mode imputation")
        
        self.processed_data = data
        self.cleaning_strategies['missing_values'] = missing_analysis
    
    def _determine_missing_strategy(self, column: str, missing_pct: float, target_column: str = None) -> str:
        """Determine the best strategy for handling missing values in a column."""
        data = self.processed_data
        
        # Strategy 1: Drop if too much missing
        if missing_pct > self.smart_config['missing_threshold_drop']:
            return 'drop_column'
        
        # Strategy 2: Check if missing pattern is informative
        if (target_column and missing_pct > 0.1 and 
            self._is_missing_pattern_informative(column, target_column)):
            return 'preserve_as_feature'
        
        # Strategy 3: High missing percentage - use advanced imputation
        if missing_pct > self.smart_config['missing_threshold_impute']:
            if data[column].dtype in ['int64', 'float64']:
                return 'iterative_imputation'
            else:
                return 'simple_imputation'
        
        # Strategy 4: Medium missing percentage - use KNN if numeric and sufficient features
        if missing_pct > 0.2 and data[column].dtype in ['int64', 'float64']:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 3:  # Need sufficient features for KNN
                return 'knn_imputation'
        
        # Strategy 5: Default simple imputation
        return 'simple_imputation'
    
    def _is_missing_pattern_informative(self, column: str, target_column: str) -> bool:
        """Check if the missing pattern in a column is informative for the target."""
        try:
            data = self.processed_data
            missing_indicator = data[column].isnull()
            
            if data[target_column].dtype in ['int64', 'float64']:
                # For numeric targets, check correlation
                correlation = missing_indicator.astype(int).corr(data[target_column])
                return abs(correlation) > 0.15  # Threshold for meaningful correlation
            else:
                # For categorical targets, check if missing rates vary significantly by class
                target_groups = data.groupby(target_column)[column].apply(lambda x: x.isnull().mean())
                return target_groups.std() > 0.15  # Significant variance in missing rates
                
        except Exception:
            return False
    
    def _smart_outlier_treatment(self):
        """Intelligent outlier detection and treatment."""
        self.log("[CLEAN] Applying smart outlier treatment")
        
        data = self.processed_data
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for column in numeric_columns:
            outliers_detected = self._detect_outliers_multiple_methods(data[column])
            outlier_percentage = outliers_detected.sum() / len(data)
            
            if outlier_percentage > 0:
                outlier_info[column] = {
                    'count': outliers_detected.sum(),
                    'percentage': outlier_percentage,
                    'strategy': self._determine_outlier_strategy(column, outlier_percentage)
                }
                
                strategy = outlier_info[column]['strategy']
                
                if strategy == 'cap':
                    # Cap at 1st and 99th percentiles
                    lower_cap = data[column].quantile(0.01)
                    upper_cap = data[column].quantile(0.99)
                    data[column] = data[column].clip(lower=lower_cap, upper=upper_cap)
                    self.log(f"[CAP] Column '{column}' - outliers capped at percentiles")
                    
                elif strategy == 'winsorize':
                    # Winsorize at 5th and 95th percentiles
                    lower_limit = data[column].quantile(0.05)
                    upper_limit = data[column].quantile(0.95)
                    data[column] = np.where(data[column] < lower_limit, lower_limit, data[column])
                    data[column] = np.where(data[column] > upper_limit, upper_limit, data[column])
                    self.log(f"[WINSORIZE] Column '{column}' - outliers winsorized")
                    
                elif strategy == 'transform':
                    # Apply log transformation for positive values
                    if data[column].min() > 0:
                        data[column] = np.log1p(data[column])
                        self.log(f"[TRANSFORM] Column '{column}' - log transformation applied")
                    else:
                        # Use Box-Cox or Yeo-Johnson transformation
                        try:
                            from sklearn.preprocessing import PowerTransformer
                            pt = PowerTransformer(method='yeo-johnson')
                            data[column] = pt.fit_transform(data[[column]]).flatten()
                            self.log(f"[TRANSFORM] Column '{column}' - power transformation applied")
                        except:
                            pass
                            
                elif strategy == 'flag':
                    # Create outlier flag and keep original values
                    data[f'{column}_is_outlier'] = outliers_detected.astype(int)
                    self.log(f"[FLAG] Column '{column}' - outliers flagged as feature")
        
        self.processed_data = data
        self.cleaning_strategies['outliers'] = outlier_info
    
    def _detect_outliers_multiple_methods(self, series: pd.Series) -> pd.Series:
        """Detect outliers using multiple methods and return consensus."""
        methods_results = []
        
        # Method 1: IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        methods_results.append(iqr_outliers)
        
        # Method 2: Z-score method
        z_scores = np.abs((series - series.mean()) / series.std())
        zscore_outliers = z_scores > 3
        methods_results.append(zscore_outliers)
        
        # Method 3: Modified Z-score method
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        modified_outliers = np.abs(modified_z_scores) > 3.5
        methods_results.append(modified_outliers)
        
        # Consensus: outlier if detected by at least 2 methods
        consensus = sum(methods_results) >= 2
        return consensus
    
    def _determine_outlier_strategy(self, column: str, outlier_percentage: float) -> str:
        """Determine the best strategy for handling outliers."""
        if outlier_percentage > 0.2:  # More than 20% outliers
            return 'transform'  # Likely distribution issue
        elif outlier_percentage > 0.1:  # 10-20% outliers
            return 'winsorize'  # Moderate outlier presence
        elif outlier_percentage > 0.05:  # 5-10% outliers
            return 'cap'  # Few but significant outliers
        else:
            return 'flag'  # Very few outliers, preserve information
    
    def _advanced_text_cleaning(self):
        """Advanced text cleaning including emoji removal and normalization."""
        self.log("[CLEAN] Applying advanced text cleaning")
        
        data = self.processed_data
        text_columns = data.select_dtypes(include=['object']).columns
        
        for column in text_columns:
            if data[column].dtype == 'object':
                # Remove emojis and special characters
                data[column] = data[column].astype(str).apply(self._clean_text_advanced)
                
                # Normalize whitespace
                data[column] = data[column].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
                
                # Handle common inconsistencies
                data[column] = data[column].apply(self._normalize_text_values)
                
        self.processed_data = data
    
    def _clean_text_advanced(self, text: str) -> str:
        """Advanced text cleaning function."""
        if pd.isna(text) or text == 'nan':
            return text
        
        text = str(text)
        
        # Remove emojis
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _normalize_text_values(self, text: str) -> str:
        """Normalize common text value inconsistencies."""
        if pd.isna(text) or text == 'nan':
            return text
        
        text = str(text).strip().lower()
        
        # Common boolean variations
        if text in ['yes', 'y', 'true', '1', 'on', 'enabled']:
            return 'yes'
        elif text in ['no', 'n', 'false', '0', 'off', 'disabled']:
            return 'no'
        
        # Common missing value representations
        if text in ['', 'null', 'none', 'na', 'n/a', '-', '--', 'unknown']:
            return np.nan
        
        return text
    
    def _optimize_feature_types(self):
        """Intelligently optimize feature data types for memory and performance."""
        self.log("[OPTIMIZE] Optimizing feature data types")
        
        data = self.processed_data
        original_memory = data.memory_usage(deep=True).sum()
        
        for column in data.columns:
            # Optimize numeric types
            if data[column].dtype in ['int64']:
                if data[column].min() >= 0:
                    if data[column].max() <= 255:
                        data[column] = data[column].astype('uint8')
                    elif data[column].max() <= 65535:
                        data[column] = data[column].astype('uint16')
                    elif data[column].max() <= 4294967295:
                        data[column] = data[column].astype('uint32')
                else:
                    if data[column].min() >= -128 and data[column].max() <= 127:
                        data[column] = data[column].astype('int8')
                    elif data[column].min() >= -32768 and data[column].max() <= 32767:
                        data[column] = data[column].astype('int16')
                    elif data[column].min() >= -2147483648 and data[column].max() <= 2147483647:
                        data[column] = data[column].astype('int32')
            
            elif data[column].dtype == 'float64':
                # Check if can be converted to float32 without loss
                if np.allclose(data[column].dropna(), data[column].dropna().astype('float32'), equal_nan=True):
                    data[column] = data[column].astype('float32')
            
            # Optimize categorical types
            elif data[column].dtype == 'object':
                unique_ratio = data[column].nunique() / len(data[column])
                if unique_ratio < 0.5:  # Less than 50% unique values
                    data[column] = data[column].astype('category')
        
        new_memory = data.memory_usage(deep=True).sum()
        memory_saved = (original_memory - new_memory) / original_memory * 100
        
        self.processed_data = data
        self.log(f"[OPTIMIZE] Memory usage reduced by {memory_saved:.1f}%")
    
    def _smart_duplicate_handling(self):
        """Intelligent duplicate handling with preservation of important variations."""
        self.log("[CLEAN] Applying smart duplicate handling")
        
        data = self.processed_data
        initial_rows = len(data)
        
        # Find exact duplicates
        exact_duplicates = data.duplicated()
        
        if exact_duplicates.sum() > 0:
            data = data.drop_duplicates()
            self.log(f"[REMOVE] Removed {exact_duplicates.sum()} exact duplicate rows")
        
        # Find near-duplicates for string columns
        text_columns = data.select_dtypes(include=['object', 'category']).columns
        
        if len(text_columns) > 0:
            # Use approximate string matching for near-duplicates
            near_duplicates = self._find_near_duplicates(data, text_columns)
            if near_duplicates > 0:
                self.log(f"[INFO] Found {near_duplicates} potential near-duplicate groups")
        
        final_rows = len(data)
        self.processed_data = data
        self.log(f"[INFO] Rows after duplicate removal: {initial_rows} -> {final_rows}")
    
    def _find_near_duplicates(self, data: pd.DataFrame, text_columns: List[str]) -> int:
        """Find near-duplicates using string similarity (simplified implementation)."""
        # This is a simplified version - in practice, you might use more sophisticated
        # string matching algorithms like Levenshtein distance or fuzzy matching
        near_duplicate_groups = 0
        
        for column in text_columns[:3]:  # Limit to first 3 text columns for performance
            if data[column].dtype in ['object', 'category']:
                # Group by similar strings (same length and first few characters)
                groups = data.groupby([
                    data[column].str.len(),
                    data[column].str[:3]
                ])[column].count()
                
                # Count groups with multiple similar entries
                near_duplicate_groups += (groups > 1).sum()
        
        return near_duplicate_groups
    
    def _analyze_feature_correlations(self):
        """Analyze and handle highly correlated features."""
        self.log("[ANALYZE] Analyzing feature correlations")
        
        data = self.processed_data
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > self.smart_config['correlation_threshold']:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })
        
        # Remove redundant features
        features_to_remove = set()
        for pair in high_corr_pairs:
            if pair['feature1'] not in features_to_remove:
                features_to_remove.add(pair['feature2'])
        
        if features_to_remove:
            data = data.drop(columns=list(features_to_remove))
            self.log(f"[REMOVE] Removed {len(features_to_remove)} highly correlated features")
        
        self.processed_data = data
        self.cleaning_strategies['correlations'] = {
            'high_corr_pairs': high_corr_pairs,
            'removed_features': list(features_to_remove)
        }
    
    def _improve_data_consistency(self):
        """Improve overall data consistency."""
        self.log("[IMPROVE] Improving data consistency")
        
        data = self.processed_data
        
        # Standardize categorical values
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            if data[column].dtype == 'object':
                # Convert to consistent case
                data[column] = data[column].astype(str).str.strip().str.lower()
                
                # Group similar values
                value_counts = data[column].value_counts()
                
                # If too many unique values, group less frequent ones
                if len(value_counts) > 100:
                    top_values = value_counts.head(50).index.tolist()
                    data[column] = data[column].apply(
                        lambda x: x if x in top_values else 'other'
                    )
                    self.log(f"[GROUP] Column '{column}' - grouped rare categories as 'other'")
        
        self.processed_data = data
    
    def _final_quality_assessment(self):
        """Final comprehensive quality assessment."""
        self.log("[ASSESS] Conducting final quality assessment")
        
        data = self.processed_data
        final_assessment = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_cells': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns),
        }
        
        final_assessment['missing_percentage'] = (final_assessment['missing_cells'] / 
                                                (final_assessment['total_rows'] * final_assessment['total_columns'])) * 100
        
        self.data_quality_metrics['final'] = final_assessment
        
        # Calculate improvement metrics
        initial = self.data_quality_metrics['initial']
        improvement = {
            'missing_reduction': initial['missing_percentage'] - final_assessment['missing_percentage'],
            'duplicate_reduction': initial['duplicate_rows'] - final_assessment['duplicate_rows'],
            'memory_reduction': initial['memory_usage_mb'] - final_assessment['memory_usage_mb'],
            'column_retention': final_assessment['total_columns'] / initial['total_columns']
        }
        
        self.data_quality_metrics['improvement'] = improvement
        
        self.log(f"[RESULTS] Final dataset: {final_assessment['total_rows']} rows, {final_assessment['total_columns']} columns")
        self.log(f"[RESULTS] Missing data: {final_assessment['missing_percentage']:.2f}% (reduced by {improvement['missing_reduction']:.2f}%)")
        self.log(f"[RESULTS] Memory usage: {final_assessment['memory_usage_mb']:.1f}MB (saved {improvement['memory_reduction']:.1f}MB)")
    
    def get_processing_summary(self) -> Dict:
        """Get comprehensive processing summary."""
        return {
            'processing_log': self.processing_log,
            'cleaning_strategies': self.cleaning_strategies,
            'quality_metrics': self.data_quality_metrics,
            'feature_engineering': self.feature_engineering_applied,
            'configuration': self.smart_config
        }
    
    def log(self, message: str):
        """Add message to processing log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.processing_log.append(formatted_message)
        print(formatted_message)
    
    def export_cleaned_data(self, file_path: str, format: str = 'csv'):
        """Export cleaned data to various formats."""
        if self.processed_data is None:
            raise ValueError("No processed data available to export")
        
        file_path = Path(file_path)
        
        if format.lower() == 'csv':
            self.processed_data.to_csv(file_path, index=False)
        elif format.lower() == 'parquet':
            self.processed_data.to_parquet(file_path, index=False)
        elif format.lower() == 'excel':
            self.processed_data.to_excel(file_path, index=False)
        elif format.lower() == 'json':
            self.processed_data.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.log(f"[EXPORT] Data exported to {file_path} in {format.upper()} format")
        
        # Also export processing summary
        summary_path = file_path.with_suffix('.summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.get_processing_summary(), f, indent=2, default=str)
        
        self.log(f"[EXPORT] Processing summary exported to {summary_path}")