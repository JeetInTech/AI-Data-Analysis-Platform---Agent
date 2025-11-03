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

# Core ML and preprocessing imports
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

# Advanced feature engineering
try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False

# Text processing imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Advanced NLP and embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# File format support
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import sqlalchemy as sa
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

warnings.filterwarnings('ignore')

class DataLineageTracker:
    """Track complete data transformation lineage for audit trails."""
    
    def __init__(self):
        self.transformations = []
        self.metadata = {}
        self.start_time = datetime.now()
        
    def log_transformation(self, operation: str, details: Dict, 
                          input_shape: Tuple = None, output_shape: Tuple = None):
        """Log a data transformation with full context."""
        transformation = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds()
        }
        self.transformations.append(transformation)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to lineage tracking."""
        self.metadata[key] = value
    
    def get_lineage_summary(self) -> Dict:
        """Get comprehensive lineage summary."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_transformations': len(self.transformations),
            'transformations': self.transformations,
            'metadata': self.metadata
        }
    
    def export_lineage(self, file_path: str):
        """Export lineage to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.get_lineage_summary(), f, indent=2, default=str)

class AdvancedFeatureEngineering:
    """Advanced automated feature engineering capabilities."""
    
    def __init__(self):
        self.generated_features = []
        self.feature_importance = {}
        
    def create_polynomial_features(self, df: pd.DataFrame, numeric_cols: List[str], 
                                  degree: int = 2, max_features: int = 50) -> pd.DataFrame:
        """Create polynomial and interaction features."""
        from sklearn.preprocessing import PolynomialFeatures
        
        if len(numeric_cols) > 10:  # Limit to prevent feature explosion
            numeric_cols = numeric_cols[:10]
        
        poly = PolynomialFeatures(degree=degree, interaction_only=False, 
                                include_bias=False)
        
        poly_features = poly.fit_transform(df[numeric_cols])
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Limit number of features to prevent explosion
        if len(feature_names) > max_features:
            feature_names = feature_names[:max_features]
            poly_features = poly_features[:, :max_features]
        
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Remove original columns to avoid duplication
        original_cols = [col for col in feature_names if col in numeric_cols]
        poly_df = poly_df.drop(columns=original_cols, errors='ignore')
        
        self.generated_features.extend(['poly_' + col for col in poly_df.columns])
        return pd.concat([df, poly_df.add_prefix('poly_')], axis=1)
    
    def create_datetime_features(self, df: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
        """Create comprehensive datetime features."""
        result_df = df.copy()
        
        for col in datetime_cols:
            if col not in df.columns:
                continue
                
            dt_series = pd.to_datetime(df[col], errors='coerce')
            
            # Basic datetime features
            result_df[f'{col}_year'] = dt_series.dt.year
            result_df[f'{col}_month'] = dt_series.dt.month
            result_df[f'{col}_day'] = dt_series.dt.day
            result_df[f'{col}_dayofweek'] = dt_series.dt.dayofweek
            result_df[f'{col}_dayofyear'] = dt_series.dt.dayofyear
            result_df[f'{col}_week'] = dt_series.dt.isocalendar().week
            result_df[f'{col}_quarter'] = dt_series.dt.quarter
            
            # Advanced datetime features
            result_df[f'{col}_hour'] = dt_series.dt.hour
            result_df[f'{col}_minute'] = dt_series.dt.minute
            result_df[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
            result_df[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
            result_df[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
            result_df[f'{col}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
            result_df[f'{col}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
            
            # Cyclical encoding for time features
            result_df[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
            result_df[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
            result_df[f'{col}_day_sin'] = np.sin(2 * np.pi * dt_series.dt.day / 31)
            result_df[f'{col}_day_cos'] = np.cos(2 * np.pi * dt_series.dt.day / 31)
            result_df[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
            result_df[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
            
            # Business calendar features
            result_df[f'{col}_season'] = ((dt_series.dt.month % 12 + 3) // 3).map({
                1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'
            })
            
            # Time since epoch (for trend analysis)
            result_df[f'{col}_timestamp'] = dt_series.astype('int64') // 10**9
            
            # Drop original datetime column
            result_df = result_df.drop(columns=[col])
            
            datetime_features = [col for col in result_df.columns if col.startswith(f'{col}_')]
            self.generated_features.extend(datetime_features)
        
        return result_df
    
    def create_frequency_encoding(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Create frequency encoding for categorical variables."""
        result_df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            freq_map = df[col].value_counts().to_dict()
            result_df[f'{col}_frequency'] = df[col].map(freq_map)
            result_df[f'{col}_frequency_norm'] = result_df[f'{col}_frequency'] / len(df)
            
            self.generated_features.extend([f'{col}_frequency', f'{col}_frequency_norm'])
        
        return result_df
    
    def create_target_encoding(self, df: pd.DataFrame, categorical_cols: List[str], 
                              target_col: str, cv_folds: int = 5) -> pd.DataFrame:
        """Create target encoding with cross-validation to prevent overfitting."""
        if target_col not in df.columns:
            return df
            
        result_df = df.copy()
        
        from sklearn.model_selection import KFold
        
        if df[target_col].dtype == 'object':
            # For classification, use label encoding first
            le = LabelEncoder()
            target_encoded = le.fit_transform(df[target_col].astype(str))
        else:
            target_encoded = df[target_col].values
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col not in df.columns or col == target_col:
                continue
            
            result_df[f'{col}_target_encoded'] = 0.0
            
            # Cross-validation target encoding
            for train_idx, val_idx in kf.split(df):
                train_df = df.iloc[train_idx]
                target_mean = train_df.groupby(col)[target_col].mean()
                global_mean = train_df[target_col].mean()
                
                # Apply to validation set
                val_encoding = df.iloc[val_idx][col].map(target_mean).fillna(global_mean)
                result_df.loc[val_idx, f'{col}_target_encoded'] = val_encoding
            
            self.generated_features.append(f'{col}_target_encoded')
        
        return result_df
    
    def create_aggregation_features(self, df: pd.DataFrame, numeric_cols: List[str], 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        """Create aggregation features grouped by categorical variables."""
        result_df = df.copy()
        
        for cat_col in categorical_cols[:3]:  # Limit to prevent explosion
            if cat_col not in df.columns:
                continue
            
            for num_col in numeric_cols[:5]:  # Limit numeric columns
                if num_col not in df.columns:
                    continue
                
                # Group statistics
                group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'min', 'max'])
                
                for stat in ['mean', 'std', 'min', 'max']:
                    feature_name = f'{num_col}_{stat}_by_{cat_col}'
                    result_df[feature_name] = df[cat_col].map(group_stats[stat])
                    self.generated_features.append(feature_name)
        
        return result_df
    
    def create_ratio_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create ratio features between numeric columns."""
        result_df = df.copy()
        
        # Limit combinations to prevent feature explosion
        max_combinations = 20
        combinations_created = 0
        
        for i, col1 in enumerate(numeric_cols[:10]):
            if combinations_created >= max_combinations:
                break
            
            for col2 in numeric_cols[i+1:10]:
                if combinations_created >= max_combinations:
                    break
                
                if col1 not in df.columns or col2 not in df.columns:
                    continue
                
                # Avoid division by zero
                mask = (df[col2] != 0) & (df[col2].notna())
                
                if mask.sum() > len(df) * 0.5:  # Only create if enough valid values
                    result_df[f'{col1}_div_{col2}'] = np.where(mask, df[col1] / df[col2], np.nan)
                    result_df[f'{col1}_mult_{col2}'] = df[col1] * df[col2]
                    
                    self.generated_features.extend([f'{col1}_div_{col2}', f'{col1}_mult_{col2}'])
                    combinations_created += 2
        
        return result_df
    
    def create_statistical_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create statistical features across numeric columns."""
        result_df = df.copy()
        
        if len(numeric_cols) < 2:
            return result_df
        
        # Row-wise statistics
        numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            result_df['row_mean'] = numeric_df.mean(axis=1)
            result_df['row_std'] = numeric_df.std(axis=1)
            result_df['row_min'] = numeric_df.min(axis=1)
            result_df['row_max'] = numeric_df.max(axis=1)
            result_df['row_median'] = numeric_df.median(axis=1)
            result_df['row_sum'] = numeric_df.sum(axis=1)
            result_df['row_range'] = result_df['row_max'] - result_df['row_min']
            result_df['row_skew'] = numeric_df.skew(axis=1)
            result_df['row_kurtosis'] = numeric_df.kurtosis(axis=1)
            
            self.generated_features.extend([
                'row_mean', 'row_std', 'row_min', 'row_max', 'row_median', 
                'row_sum', 'row_range', 'row_skew', 'row_kurtosis'
            ])
        
        return result_df

class AdvancedTextProcessor:
    """Advanced text processing with NLP capabilities."""
    
    def __init__(self):
        self.text_features = []
        self.embedding_model = None
        
        # Initialize NLTK data if available
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                pass
        
        # Initialize sentence transformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                pass
    
    def basic_text_cleaning(self, text_series: pd.Series) -> pd.Series:
        """Perform basic text cleaning."""
        cleaned = text_series.astype(str)
        cleaned = cleaned.str.lower()
        cleaned = cleaned.str.replace(r'[^\w\s]', ' ', regex=True)
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
        cleaned = cleaned.str.strip()
        cleaned = cleaned.replace(['nan', '', ' '], np.nan)
        return cleaned
    
    def extract_text_features(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        """Extract comprehensive text features."""
        result_df = df.copy()
        
        for col in text_cols:
            if col not in df.columns:
                continue
            
            text_series = df[col].astype(str)
            
            # Basic text statistics
            result_df[f'{col}_length'] = text_series.str.len()
            result_df[f'{col}_word_count'] = text_series.str.split().str.len()
            result_df[f'{col}_char_count'] = text_series.str.replace(' ', '').str.len()
            result_df[f'{col}_sentence_count'] = text_series.str.count(r'[.!?]+')
            result_df[f'{col}_avg_word_length'] = (
                result_df[f'{col}_char_count'] / result_df[f'{col}_word_count']
            ).fillna(0)
            
            # Advanced text features
            result_df[f'{col}_uppercase_ratio'] = (
                text_series.str.count(r'[A-Z]') / text_series.str.len()
            ).fillna(0)
            result_df[f'{col}_digit_ratio'] = (
                text_series.str.count(r'\d') / text_series.str.len()
            ).fillna(0)
            result_df[f'{col}_special_char_ratio'] = (
                text_series.str.count(r'[^a-zA-Z0-9\s]') / text_series.str.len()
            ).fillna(0)
            
            # Linguistic features if NLTK available
            if NLTK_AVAILABLE:
                result_df = self._extract_linguistic_features(result_df, col, text_series)
            
            self.text_features.extend([
                f'{col}_length', f'{col}_word_count', f'{col}_char_count',
                f'{col}_sentence_count', f'{col}_avg_word_length',
                f'{col}_uppercase_ratio', f'{col}_digit_ratio', f'{col}_special_char_ratio'
            ])
        
        return result_df
    
    def _extract_linguistic_features(self, df: pd.DataFrame, col: str, text_series: pd.Series) -> pd.DataFrame:
        """Extract linguistic features using NLTK."""
        try:
            # Tokenization and POS tagging for sample (to avoid performance issues)
            sample_size = min(1000, len(text_series))
            sample_indices = text_series.sample(n=sample_size, random_state=42).index
            
            # Initialize feature columns
            df[f'{col}_noun_ratio'] = 0.0
            df[f'{col}_verb_ratio'] = 0.0
            df[f'{col}_adj_ratio'] = 0.0
            df[f'{col}_unique_word_ratio'] = 0.0
            
            for idx in sample_indices:
                text = str(text_series.loc[idx])
                if pd.isna(text) or text == 'nan':
                    continue
                
                # Tokenize and POS tag
                tokens = word_tokenize(text.lower())
                tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
                
                if len(tokens) > 0:
                    pos_tags = pos_tag(tokens)
                    
                    # Count POS types
                    noun_count = sum(1 for _, pos in pos_tags if pos.startswith('NN'))
                    verb_count = sum(1 for _, pos in pos_tags if pos.startswith('VB'))
                    adj_count = sum(1 for _, pos in pos_tags if pos.startswith('JJ'))
                    
                    # Calculate ratios
                    total_tokens = len(tokens)
                    df.loc[idx, f'{col}_noun_ratio'] = noun_count / total_tokens
                    df.loc[idx, f'{col}_verb_ratio'] = verb_count / total_tokens
                    df.loc[idx, f'{col}_adj_ratio'] = adj_count / total_tokens
                    df.loc[idx, f'{col}_unique_word_ratio'] = len(set(tokens)) / total_tokens
            
            # Fill remaining values with mean
            for feature in [f'{col}_noun_ratio', f'{col}_verb_ratio', f'{col}_adj_ratio', f'{col}_unique_word_ratio']:
                mean_val = df[feature].replace(0, np.nan).mean()
                df[feature] = df[feature].replace(0, mean_val)
            
            self.text_features.extend([
                f'{col}_noun_ratio', f'{col}_verb_ratio', f'{col}_adj_ratio', f'{col}_unique_word_ratio'
            ])
            
        except Exception as e:
            print(f"Warning: Could not extract linguistic features for {col}: {e}")
        
        return df
    
    def create_text_embeddings(self, df: pd.DataFrame, text_cols: List[str], 
                              max_features: int = 10) -> pd.DataFrame:
        """Create text embeddings using sentence transformers."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.embedding_model is None:
            return df
        
        result_df = df.copy()
        
        for col in text_cols[:2]:  # Limit to prevent memory issues
            if col not in df.columns:
                continue
            
            try:
                # Clean text first
                text_series = self.basic_text_cleaning(df[col])
                valid_texts = text_series.dropna().astype(str).tolist()
                
                if len(valid_texts) == 0:
                    continue
                
                # Create embeddings for valid texts
                embeddings = self.embedding_model.encode(valid_texts, show_progress_bar=False)
                
                # Use PCA to reduce dimensionality
                from sklearn.decomposition import PCA
                n_components = min(max_features, embeddings.shape[1], len(valid_texts))
                pca = PCA(n_components=n_components)
                reduced_embeddings = pca.fit_transform(embeddings)
                
                # Create DataFrame with embeddings
                embedding_df = pd.DataFrame(
                    reduced_embeddings,
                    columns=[f'{col}_embed_{i}' for i in range(n_components)],
                    index=text_series.dropna().index
                )
                
                # Fill missing values with mean
                for embed_col in embedding_df.columns:
                    mean_val = embedding_df[embed_col].mean()
                    full_series = pd.Series(mean_val, index=df.index, name=embed_col)
                    full_series.loc[embedding_df.index] = embedding_df[embed_col]
                    result_df[embed_col] = full_series
                
                self.text_features.extend(embedding_df.columns.tolist())
                
            except Exception as e:
                print(f"Warning: Could not create embeddings for {col}: {e}")
        
        return result_df
    
    def create_tfidf_features(self, df: pd.DataFrame, text_cols: List[str], 
                             max_features: int = 50) -> pd.DataFrame:
        """Create TF-IDF features for text columns."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        result_df = df.copy()
        
        for col in text_cols[:2]:  # Limit to prevent memory issues
            if col not in df.columns:
                continue
            
            try:
                # Clean and prepare text
                text_series = self.basic_text_cleaning(df[col])
                valid_texts = text_series.fillna('').astype(str)
                
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english' if NLTK_AVAILABLE else None,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8
                )
                
                tfidf_matrix = vectorizer.fit_transform(valid_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Create DataFrame with TF-IDF features
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f'{col}_tfidf_{name}' for name in feature_names],
                    index=df.index
                )
                
                result_df = pd.concat([result_df, tfidf_df], axis=1)
                self.text_features.extend(tfidf_df.columns.tolist())
                
            except Exception as e:
                print(f"Warning: Could not create TF-IDF features for {col}: {e}")
        
        return result_df

class EnhancedAIDecisionEngine:
    """Enhanced AI reasoning engine with advanced analytics."""
    
    def __init__(self):
        self.decisions = []
        self.reasoning_log = []
        self.meta_features = {}
        
    def analyze_column_characteristics(self, series: pd.Series, column_name: str) -> Dict:
        """Enhanced column analysis with advanced statistics."""
        analysis = {
            'column': column_name,
            'dtype': str(series.dtype),
            'total_rows': len(series),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100,
            'memory_usage': series.memory_usage(deep=True) / 1024,  # KB
        }
        
        # Advanced statistical analysis
        if pd.api.types.is_numeric_dtype(series):
            analysis.update(self._analyze_numeric_column(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            analysis.update(self._analyze_datetime_column(series))
        else:
            analysis.update(self._analyze_text_column(series))
        
        # AI reasoning for data type
        analysis['ai_type'] = self._determine_column_type(series, analysis)
        analysis['reasoning'] = self._generate_column_reasoning(analysis)
        
        # AI reasoning for missing value strategy
        analysis['missing_strategy'] = self._determine_missing_strategy(analysis, series)
        
        # AI reasoning for cleaning priority
        analysis['priority'] = self._determine_cleaning_priority(analysis)
        
        return analysis
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict:
        """Analyze numeric column characteristics."""
        numeric_series = pd.to_numeric(series, errors='coerce')
        valid_series = numeric_series.dropna()
        
        if len(valid_series) == 0:
            return {'numeric_stats': 'no_valid_data'}
        
        return {
            'mean': float(valid_series.mean()),
            'median': float(valid_series.median()),
            'std': float(valid_series.std()),
            'min': float(valid_series.min()),
            'max': float(valid_series.max()),
            'skewness': float(valid_series.skew()),
            'kurtosis': float(valid_series.kurtosis()),
            'q25': float(valid_series.quantile(0.25)),
            'q75': float(valid_series.quantile(0.75)),
            'outlier_count': self._count_outliers(valid_series),
            'is_integer_like': all(valid_series == valid_series.astype(int))
        }
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict:
        """Analyze datetime column characteristics."""
        try:
            dt_series = pd.to_datetime(series, errors='coerce').dropna()
            if len(dt_series) == 0:
                return {'datetime_stats': 'no_valid_data'}
            
            return {
                'min_date': dt_series.min().isoformat(),
                'max_date': dt_series.max().isoformat(),
                'date_range_days': (dt_series.max() - dt_series.min()).days,
                'unique_dates': dt_series.nunique(),
                'has_time': any(dt_series.dt.hour != 0) or any(dt_series.dt.minute != 0)
            }
        except:
            return {'datetime_stats': 'invalid_datetime'}
    
    def _analyze_text_column(self, series: pd.Series) -> Dict:
        """Analyze text column characteristics."""
        text_series = series.astype(str).dropna()
        if len(text_series) == 0:
            return {'text_stats': 'no_valid_data'}
        
        lengths = text_series.str.len()
        
        return {
            'avg_length': float(lengths.mean()),
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'std_length': float(lengths.std()),
            'contains_numbers': text_series.str.contains(r'\d').sum(),
            'contains_special_chars': text_series.str.contains(r'[^a-zA-Z0-9\s]').sum(),
            'avg_words': text_series.str.split().str.len().mean(),
            'likely_categorical': self._is_likely_categorical(text_series)
        }
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def _is_likely_categorical(self, text_series: pd.Series) -> bool:
        """Determine if text column is likely categorical."""
        unique_ratio = text_series.nunique() / len(text_series)
        avg_length = text_series.str.len().mean()
        return unique_ratio < 0.5 and avg_length < 100
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Enhanced datetime detection."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        
        datetime_matches = 0
        for value in sample:
            try:
                pd.to_datetime(str(value))
                datetime_matches += 1
            except:
                continue
        
        return datetime_matches / len(sample) > 0.6
    
    def _determine_column_type(self, series: pd.Series, analysis: Dict) -> str:
        """Enhanced column type determination."""
        if pd.api.types.is_datetime64_any_dtype(series) or self._is_datetime_column(series):
            return 'datetime'
        
        if pd.api.types.is_numeric_dtype(series):
            if analysis.get('is_integer_like', False) and analysis['unique_count'] < 20:
                return 'categorical_numeric'
            return 'numerical'
        
        if series.dtype == 'object':
            text_stats = analysis.get('avg_length', 0)
            if text_stats > 100 or not analysis.get('likely_categorical', False):
                return 'text'
            return 'categorical'
        
        return 'other'
    
    def _generate_column_reasoning(self, analysis: Dict) -> str:
        """Generate detailed AI reasoning for column classification."""
        col_type = analysis['ai_type']
        unique_pct = analysis['unique_percentage']
        null_pct = analysis['null_percentage']
        
        if col_type == 'numerical':
            if 'outlier_count' in analysis:
                outlier_info = f", {analysis['outlier_count']} outliers detected"
            else:
                outlier_info = ""
            return f"🤖[AI ANALYSIS] Numerical data with {analysis['unique_count']} unique values{outlier_info}"
        
        elif col_type == 'categorical':
            return f"🤖[AI ANALYSIS] Categorical data - {analysis['unique_count']} categories ({unique_pct:.1f}% unique)"
        
        elif col_type == 'categorical_numeric':
            return f"🤖[AI ANALYSIS] Integer values but only {analysis['unique_count']} unique - likely categorical encoding"
        
        elif col_type == 'datetime':
            if 'date_range_days' in analysis:
                return f"🤖[AI ANALYSIS] Datetime data spanning {analysis['date_range_days']} days"
            return "🤖[AI ANALYSIS] Datetime patterns detected"
        
        elif col_type == 'text':
            if 'avg_length' in analysis:
                return f"🤖[AI ANALYSIS] Text data (avg length: {analysis['avg_length']:.1f} chars) - high cardinality"
            return "🤖[AI ANALYSIS] Free-form text data detected"
        
        else:
            return f"🤖[AI ANALYSIS] Unknown data type requiring investigation"
    
    def _determine_missing_strategy(self, analysis: Dict, series: pd.Series) -> Dict:
        """Enhanced missing value strategy determination."""
        null_pct = analysis['null_percentage']
        col_type = analysis['ai_type']
        
        if null_pct == 0:
            return {
                'strategy': 'none',
                'reasoning': "✅[OK] No missing values - no action needed"
            }
        
        # Strategy based on missing percentage and data type
        if null_pct < 2:
            if col_type == 'numerical':
                return {
                    'strategy': 'median',
                    'reasoning': f"🧠[AI DECISION] {null_pct:.1f}% missing - median imputation preserves distribution"
                }
            elif col_type in ['categorical', 'categorical_numeric']:
                return {
                    'strategy': 'mode',
                    'reasoning': f"🧠[AI DECISION] {null_pct:.1f}% missing - mode imputation for categorical"
                }
            elif col_type == 'text':
                return {
                    'strategy': 'unknown_category',
                    'reasoning': f"🧠[AI DECISION] {null_pct:.1f}% missing text - create 'Unknown' category"
                }
                
        elif null_pct < 10:
            if col_type == 'numerical':
                return {
                    'strategy': 'knn',
                    'reasoning': f"🧠[AI DECISION] {null_pct:.1f}% missing - KNN preserves feature relationships"
                }
            else:
                return {
                    'strategy': 'mode_with_validation',
                    'reasoning': f"🧠[AI DECISION] {null_pct:.1f}% missing - mode with cross-validation"
                }
                
        elif null_pct < 25:
            if col_type == 'numerical' and analysis.get('outlier_count', 0) > 0:
                return {
                    'strategy': 'iterative',
                    'reasoning': f"🧠[AI DECISION] {null_pct:.1f}% missing with outliers - iterative imputation"
                }
            else:
                return {
                    'strategy': 'create_indicator',
                    'reasoning': f"🧠[AI DECISION] {null_pct:.1f}% missing - pattern may be meaningful, create indicator"
                }
                
        elif null_pct < 50:
            return {
                'strategy': 'advanced_imputation',
                'reasoning': f"🧠[AI DECISION] {null_pct:.1f}% missing - requires advanced imputation strategy"
            }
        else:
            return {
                'strategy': 'consider_removal',
                'reasoning': f"⚠️[WARNING] {null_pct:.1f}% missing - consider column removal or separate analysis"
            }
    
    def _determine_cleaning_priority(self, analysis: Dict) -> Dict:
        """Enhanced cleaning priority determination."""
        null_pct = analysis['null_percentage']
        unique_pct = analysis['unique_percentage']
        col_type = analysis['ai_type']
        
        # High priority conditions
        if null_pct > 40:
            return {'level': 'high', 'reason': 'Very high missing values need immediate attention'}
        
        if col_type == 'categorical' and unique_pct > 90:
            return {'level': 'high', 'reason': 'Extremely high cardinality categorical needs strategy'}
        
        if col_type == 'text' and analysis.get('avg_length', 0) > 500:
            return {'level': 'high', 'reason': 'Long text data needs specialized NLP processing'}
        
        # Medium priority conditions
        if null_pct > 15:
            return {'level': 'medium', 'reason': 'Moderate missing values need attention'}
        
        if col_type == 'datetime':
            return {'level': 'medium', 'reason': 'Datetime feature extraction recommended'}
        
        if col_type == 'numerical' and analysis.get('outlier_count', 0) > len(analysis) * 0.05:
            return {'level': 'medium', 'reason': 'Significant outliers detected'}
        
        if col_type == 'categorical' and unique_pct > 70:
            return {'level': 'medium', 'reason': 'High cardinality categorical needs encoding strategy'}
        
        # Low priority (standard cleaning)
        return {'level': 'low', 'reason': 'Standard cleaning procedures sufficient'}
    
    def extract_dataset_meta_features(self, df: pd.DataFrame) -> Dict:
        """Extract comprehensive meta-features for intelligent processing."""
        meta_features = {
            # Basic characteristics
            'n_rows': df.shape[0],
            'n_cols': df.shape[1],
            'total_cells': df.shape[0] * df.shape[1],
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            
            # Data type distribution
            'n_numeric': len(df.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'n_datetime': len(df.select_dtypes(include=['datetime', 'datetimetz']).columns),
            
            # Missing value patterns
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'columns_with_missing': (df.isnull().sum() > 0).sum(),
            'complete_rows_percentage': (df.dropna().shape[0] / df.shape[0]) * 100,
        }
        
        # Calculate percentages
        meta_features['pct_numeric'] = (meta_features['n_numeric'] / meta_features['n_cols']) * 100
        meta_features['pct_categorical'] = (meta_features['n_categorical'] / meta_features['n_cols']) * 100
        meta_features['pct_datetime'] = (meta_features['n_datetime'] / meta_features['n_cols']) * 100
        
        # Advanced characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Correlation analysis
            corr_matrix = df[numeric_cols].corr().abs()
            upper_triangle = np.triu(corr_matrix, k=1)
            meta_features['max_correlation'] = np.nanmax(upper_triangle) if upper_triangle.size > 0 else 0
            meta_features['high_corr_pairs'] = (upper_triangle > 0.8).sum()
            
            # Outlier analysis
            outlier_counts = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts.append(outliers)
            
            meta_features['total_outliers'] = sum(outlier_counts)
            meta_features['outlier_percentage'] = (sum(outlier_counts) / len(df)) * 100
        else:
            meta_features['max_correlation'] = 0
            meta_features['high_corr_pairs'] = 0
            meta_features['total_outliers'] = 0
            meta_features['outlier_percentage'] = 0
        
        # Text analysis
        text_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].astype(str).str.len().mean() > 50:
                text_cols.append(col)
        
        meta_features['n_text'] = len(text_cols)
        meta_features['pct_text'] = (len(text_cols) / meta_features['n_cols']) * 100 if meta_features['n_cols'] > 0 else 0
        
        # Cardinality analysis
        high_cardinality_cols = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8:
                high_cardinality_cols.append(col)
        
        meta_features['n_high_cardinality'] = len(high_cardinality_cols)
        meta_features['pct_high_cardinality'] = (len(high_cardinality_cols) / meta_features['n_cols']) * 100
        
        # Data quality score
        completeness_score = (100 - meta_features['missing_percentage']) / 100
        correlation_penalty = min(meta_features['high_corr_pairs'] * 0.05, 0.3)
        outlier_penalty = min(meta_features['outlier_percentage'] * 0.01, 0.2)
        
        meta_features['data_quality_score'] = max(0, completeness_score - correlation_penalty - outlier_penalty)
        
        # Complexity indicators
        meta_features['dataset_complexity'] = self._assess_dataset_complexity(meta_features)
        meta_features['recommended_processing_time'] = self._estimate_processing_time(meta_features)
        
        self.meta_features = meta_features
        return meta_features
    
    def _assess_dataset_complexity(self, meta_features: Dict) -> str:
        """Assess dataset complexity based on meta-features."""
        complexity_score = 0
        
        # Size complexity
        if meta_features['total_cells'] > 1000000:
            complexity_score += 3
        elif meta_features['total_cells'] > 100000:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Missing data complexity
        if meta_features['missing_percentage'] > 30:
            complexity_score += 3
        elif meta_features['missing_percentage'] > 10:
            complexity_score += 2
        
        # Data type complexity
        if meta_features['pct_text'] > 30:
            complexity_score += 2
        if meta_features['n_high_cardinality'] > 5:
            complexity_score += 2
        
        # Correlation complexity
        if meta_features['high_corr_pairs'] > 10:
            complexity_score += 2
        
        if complexity_score <= 3:
            return 'simple'
        elif complexity_score <= 6:
            return 'moderate'
        elif complexity_score <= 9:
            return 'complex'
        else:
            return 'highly_complex'
    
    def _estimate_processing_time(self, meta_features: Dict) -> str:
        """Estimate processing time based on dataset characteristics."""
        if meta_features['dataset_complexity'] == 'simple':
            return 'under_1_minute'
        elif meta_features['dataset_complexity'] == 'moderate':
            return '1_5_minutes'
        elif meta_features['dataset_complexity'] == 'complex':
            return '5_15_minutes'
        else:
            return 'over_15_minutes'
    
    def log_decision(self, decision: str):
        """Log AI decision with enhanced context."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'decision': decision,
            'context': getattr(self, 'current_context', 'general')
        }
        self.reasoning_log.append(log_entry)
        print(f"[{timestamp}] {decision}")

class SmartDataProcessor:
    """Enhanced intelligent multi-format data processor with advanced AI reasoning."""
    
    def __init__(self):
        # Core data storage
        self.original_data = None
        self.cleaned_data = None
        self.processed_data = None
        
        # Processing configuration
        self.target_column = None
        self.problem_type = None
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
        
        # Enhanced components
        self.ai_engine = EnhancedAIDecisionEngine()
        self.lineage_tracker = DataLineageTracker()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.text_processor = AdvancedTextProcessor()
        
        # Analysis results
        self.data_info = {}
        self.column_analyses = {}
        self.cleaning_recommendations = {}
        self.meta_features = {}
        
        # Processing logs
        self.processing_log = []
        
        # Data source tracking
        self.data_source = {'type': None, 'path': None, 'metadata': {}}
    
    def load_and_analyze(self, file_path: str) -> bool:
        """Load CSV and perform intelligent analysis with AI reasoning (original method)."""
        return self._load_csv_file(file_path)
    
    def load_and_analyze_dataframe(self, df: pd.DataFrame, source_name: str = "DataFrame") -> bool:
        """Load DataFrame and perform analysis (new method for main.py)."""
        try:
            self.ai_engine.log_decision(f"🚀[LOAD] {source_name}")
            
            self.original_data = df.copy()
            self.data_source = {
                'type': 'dataframe',
                'path': source_name,
                'metadata': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict()
                }
            }
            
            self.lineage_tracker.log_transformation(
                'data_loading',
                {'source': source_name, 'format': 'dataframe'},
                None,
                df.shape
            )
            
            self.ai_engine.log_decision(f"📊[INFO] DataFrame loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Perform AI analysis
            return self._perform_ai_analysis()
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌[ERROR] Error loading DataFrame: {str(e)}")
            return False
    
    def _load_csv_file(self, file_path: str) -> bool:
        """Load CSV file with intelligent encoding detection."""
        try:
            self.ai_engine.log_decision("🚀[START] Starting intelligent CSV loading...")
            
            # Try different encodings with AI reasoning
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.original_data = pd.read_csv(file_path, encoding=encoding)
                    self.ai_engine.log_decision(f"✅[SUCCESS] Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    self.ai_engine.log_decision(f"❌ Failed with {encoding} encoding, trying next...")
                    continue
            
            if self.original_data is None:
                raise Exception("Could not load file with any encoding")
            
            self.data_source = {
                'type': 'csv',
                'path': file_path,
                'metadata': {
                    'size_bytes': Path(file_path).stat().st_size,
                    'encoding': encoding
                }
            }
            
            self.lineage_tracker.log_transformation(
                'data_loading',
                {'source': file_path, 'format': 'csv', 'encoding': encoding},
                None,
                self.original_data.shape
            )
                
            self.ai_engine.log_decision(f"📊[INFO] Dataset loaded: {self.original_data.shape[0]:,} rows × {self.original_data.shape[1]} columns")
            
            # Perform AI analysis
            return self._perform_ai_analysis()
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error loading CSV: {str(e)}")
            return False
    
    def load_excel_file(self, file_path: str, sheet_name: str = None) -> bool:
        """Load Excel file with sheet selection."""
        if not EXCEL_AVAILABLE:
            self.ai_engine.log_decision("❌ Excel support not available - install openpyxl")
            return False
        
        try:
            self.ai_engine.log_decision(f"📊 Loading Excel file: {file_path}")
            
            # Read Excel file and get sheet names
            excel_file = pd.ExcelFile(file_path)
            available_sheets = excel_file.sheet_names
            
            self.ai_engine.log_decision(f"📋 Found {len(available_sheets)} sheets: {available_sheets}")
            
            # Use specified sheet or first sheet
            sheet_to_load = sheet_name if sheet_name and sheet_name in available_sheets else available_sheets[0]
            self.ai_engine.log_decision(f"📖 Loading sheet: {sheet_to_load}")
            
            # Load the sheet
            self.original_data = pd.read_excel(file_path, sheet_name=sheet_to_load)
            
            self.data_source = {
                'type': 'excel',
                'path': file_path,
                'metadata': {
                    'sheet_name': sheet_to_load,
                    'available_sheets': available_sheets,
                    'size_bytes': Path(file_path).stat().st_size
                }
            }
            
            self.lineage_tracker.log_transformation(
                'data_loading',
                {'source': file_path, 'format': 'excel', 'sheet': sheet_to_load},
                None,
                self.original_data.shape
            )
            
            self.ai_engine.log_decision(f"✅ Excel loaded: {self.original_data.shape[0]:,} rows × {self.original_data.shape[1]} columns")
            
            return self._perform_ai_analysis()
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error loading Excel: {str(e)}")
            return False
    
    def load_json_file(self, file_path: str) -> bool:
        """Load JSON file with structure detection."""
        try:
            self.ai_engine.log_decision(f"🔧 Loading JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Convert JSON to DataFrame based on structure
            if isinstance(json_data, list):
                self.original_data = pd.json_normalize(json_data)
                self.ai_engine.log_decision("📋 JSON structure: List of objects - normalized to DataFrame")
            elif isinstance(json_data, dict):
                if all(isinstance(v, list) for v in json_data.values()):
                    # Dictionary of lists
                    self.original_data = pd.DataFrame(json_data)
                    self.ai_engine.log_decision("📋 JSON structure: Dictionary of lists - converted to DataFrame")
                else:
                    # Single object or nested structure
                    self.original_data = pd.json_normalize([json_data])
                    self.ai_engine.log_decision("📋 JSON structure: Single object - normalized to DataFrame")
            else:
                raise ValueError("JSON structure not suitable for tabular analysis")
            
            self.data_source = {
                'type': 'json',
                'path': file_path,
                'metadata': {
                    'structure_type': type(json_data).__name__,
                    'size_bytes': Path(file_path).stat().st_size
                }
            }
            
            self.lineage_tracker.log_transformation(
                'data_loading',
                {'source': file_path, 'format': 'json', 'structure': type(json_data).__name__},
                None,
                self.original_data.shape
            )
            
            self.ai_engine.log_decision(f"✅ JSON loaded: {self.original_data.shape[0]:,} rows × {self.original_data.shape[1]} columns")
            
            return self._perform_ai_analysis()
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error loading JSON: {str(e)}")
            return False
    
    def load_parquet_file(self, file_path: str) -> bool:
        """Load Parquet file."""
        if not PARQUET_AVAILABLE:
            self.ai_engine.log_decision("❌ Parquet support not available - install pyarrow")
            return False
        
        try:
            self.ai_engine.log_decision(f"⚡ Loading Parquet file: {file_path}")
            
            self.original_data = pd.read_parquet(file_path)
            
            self.data_source = {
                'type': 'parquet',
                'path': file_path,
                'metadata': {
                    'size_bytes': Path(file_path).stat().st_size
                }
            }
            
            self.lineage_tracker.log_transformation(
                'data_loading',
                {'source': file_path, 'format': 'parquet'},
                None,
                self.original_data.shape
            )
            
            self.ai_engine.log_decision(f"✅ Parquet loaded: {self.original_data.shape[0]:,} rows × {self.original_data.shape[1]} columns")
            
            return self._perform_ai_analysis()
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error loading Parquet: {str(e)}")
            return False
    
    def load_from_sql(self, connection_string: str, query: str) -> bool:
        """Load data from SQL database."""
        if not SQL_AVAILABLE:
            self.ai_engine.log_decision("❌ SQL support not available - install sqlalchemy")
            return False
        
        try:
            self.ai_engine.log_decision("🗄️ Connecting to database...")
            
            engine = sa.create_engine(connection_string)
            
            self.ai_engine.log_decision("📊 Executing SQL query...")
            self.original_data = pd.read_sql(query, engine)
            
            self.data_source = {
                'type': 'sql',
                'path': connection_string,
                'metadata': {
                    'query': query,
                    'connection_type': connection_string.split(':')[0]
                }
            }
            
            self.lineage_tracker.log_transformation(
                'data_loading',
                {'source': 'database', 'format': 'sql', 'query': query},
                None,
                self.original_data.shape
            )
            
            self.ai_engine.log_decision(f"✅ SQL data loaded: {self.original_data.shape[0]:,} rows × {self.original_data.shape[1]} columns")
            
            return self._perform_ai_analysis()
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error loading from SQL: {str(e)}")
            return False
    
    def _perform_ai_analysis(self) -> bool:
        """Perform comprehensive AI analysis of the dataset."""
        try:
            df = self.original_data
            
            self.ai_engine.log_decision("🔍[ANALYZE] AI performing comprehensive dataset analysis...")
            
            # Extract meta-features for intelligent processing
            self.meta_features = self.ai_engine.extract_dataset_meta_features(df)
            self.ai_engine.log_decision(f"🧠[INFO] Dataset complexity: {self.meta_features['dataset_complexity']}")
            self.ai_engine.log_decision(f"📊[INFO] Data quality score: {self.meta_features['data_quality_score']:.2f}")
            
            # Basic dataset characteristics
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells) * 100
            
            self.data_info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'unique_counts': {col: df[col].nunique() for col in df.columns},
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'completeness_percentage': completeness,
            }
            
            self.ai_engine.log_decision(f"🧠[ASSESS] AI Assessment: Dataset is {completeness:.1f}% complete")
            
            # Analyze each column with enhanced AI reasoning
            self.ai_engine.log_decision("🎯[ANALYZE] AI analyzing column characteristics with advanced statistics...")
            
            for col in df.columns:
                analysis = self.ai_engine.analyze_column_characteristics(df[col], col)
                self.column_analyses[col] = analysis
                self.ai_engine.log_decision(f"  📋 {col}: {analysis['reasoning']}")
            
            # Categorize columns by AI analysis
            self._categorize_columns()
            
            # Generate enhanced cleaning recommendations
            self._generate_cleaning_recommendations()
            
            # Log lineage
            self.lineage_tracker.log_transformation(
                'ai_analysis',
                {
                    'columns_analyzed': len(self.column_analyses),
                    'meta_features_extracted': len(self.meta_features),
                    'data_quality_score': self.meta_features['data_quality_score']
                },
                df.shape,
                df.shape
            )
            
            self.ai_engine.log_decision("✅ Enhanced AI analysis complete - ready for intelligent processing")
            return True
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error in AI analysis: {str(e)}")
            return False
    
    def _categorize_columns(self):
        """Enhanced column categorization based on AI analysis."""
        self.data_info.update({
            'numerical_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'high_cardinality_columns': [],
            'high_missing_columns': [],
        })
        
        for col, analysis in self.column_analyses.items():
            # Categorize by AI-determined type
            if analysis['ai_type'] == 'numerical':
                self.data_info['numerical_columns'].append(col)
            elif analysis['ai_type'] in ['categorical', 'categorical_numeric']:
                self.data_info['categorical_columns'].append(col)
            elif analysis['ai_type'] == 'datetime':
                self.data_info['datetime_columns'].append(col)
            elif analysis['ai_type'] == 'text':
                self.data_info['text_columns'].append(col)
            
            # Flag problematic columns
            if analysis['unique_percentage'] > 80:
                self.data_info['high_cardinality_columns'].append(col)
            
            if analysis['null_percentage'] > 20:
                self.data_info['high_missing_columns'].append(col)
        
        # Log categorization results
        self.ai_engine.log_decision(
            f"🏷️ AI Categorization: "
            f"{len(self.data_info['numerical_columns'])} numerical, "
            f"{len(self.data_info['categorical_columns'])} categorical, "
            f"{len(self.data_info['datetime_columns'])} datetime, "
            f"{len(self.data_info['text_columns'])} text columns"
        )
        
        if self.data_info['high_cardinality_columns']:
            self.ai_engine.log_decision(f"⚠️ High cardinality columns: {len(self.data_info['high_cardinality_columns'])}")
        
        if self.data_info['high_missing_columns']:
            self.ai_engine.log_decision(f"🕳️ High missing value columns: {len(self.data_info['high_missing_columns'])}")
    
    def _generate_cleaning_recommendations(self):
        """Generate enhanced AI-powered cleaning recommendations."""
        self.ai_engine.log_decision("🎯 AI generating comprehensive cleaning recommendations...")
        
        for col, analysis in self.column_analyses.items():
            recommendations = []
            
            # Missing value recommendations
            if analysis['null_percentage'] > 0:
                strategy = analysis['missing_strategy']
                recommendations.append(strategy['reasoning'])
                
                # Add specific implementation details
                if strategy['strategy'] == 'knn':
                    recommendations.append("🔧 Implementation: Use 5-neighbor KNN with related numeric features")
                elif strategy['strategy'] == 'iterative':
                    recommendations.append("🔧 Implementation: Use IterativeImputer with Random Forest estimator")
                elif strategy['strategy'] == 'create_indicator':
                    recommendations.append("🔧 Implementation: Create binary missing flag + median/mode imputation")
            
            # Data type specific recommendations
            if analysis['ai_type'] == 'datetime':
                recommendations.append("📅 Extract comprehensive datetime features (cyclical encoding recommended)")
                recommendations.append("📊 Consider time-series analysis if temporal patterns exist")
            
            elif analysis['ai_type'] == 'categorical':
                if analysis['unique_count'] > 50:
                    recommendations.append("🎯 High cardinality detected - use target encoding or embedding")
                    recommendations.append("📊 Consider frequency-based encoding for rare categories")
                else:
                    recommendations.append("🏷️ Use one-hot encoding for low cardinality categories")
            
            elif analysis['ai_type'] == 'text':
                if analysis.get('avg_length', 0) > 100:
                    recommendations.append("📝 Long text detected - consider NLP processing (TF-IDF, embeddings)")
                    recommendations.append("🧠 Extract linguistic features (POS tags, sentiment, readability)")
                else:
                    recommendations.append("📝 Short text - treat as high-cardinality categorical")
            
            elif analysis['ai_type'] == 'numerical':
                if analysis.get('outlier_count', 0) > 0:
                    outlier_pct = (analysis['outlier_count'] / analysis['total_rows']) * 100
                    if outlier_pct > 5:
                        recommendations.append(f"⚠️ {outlier_pct:.1f}% outliers - consider robust scaling or capping")
                    recommendations.append("📊 Apply outlier detection (Isolation Forest for complex patterns)")
                
                if analysis.get('skewness', 0) > 2:
                    recommendations.append("📈 High skewness - consider log transformation or Box-Cox")
            
            # Feature engineering recommendations
            if analysis['ai_type'] in ['numerical', 'categorical']:
                recommendations.append("🔧 Consider interaction features with other relevant variables")
            
            # Priority-based recommendations
            priority_level = analysis['priority']['level']
            if priority_level == 'high':
                recommendations.append("🚨 HIGH PRIORITY: Manual review recommended before automated processing")
            elif priority_level == 'medium':
                recommendations.append("🟡 MEDIUM PRIORITY: Apply advanced techniques for optimal results")
            
            self.cleaning_recommendations[col] = {
                'priority': priority_level,
                'recommendations': recommendations,
                'auto_apply': priority_level == 'low',  # Only auto-apply low priority items
                'estimated_improvement': self._estimate_improvement(analysis),
                'processing_time': self._estimate_column_processing_time(analysis)
            }
        
        # Summarize recommendations
        high_priority = sum(1 for rec in self.cleaning_recommendations.values() if rec['priority'] == 'high')
        medium_priority = sum(1 for rec in self.cleaning_recommendations.values() if rec['priority'] == 'medium')
        
        self.ai_engine.log_decision(f"📋 Recommendations generated: {high_priority} high, {medium_priority} medium priority")
    
    def _estimate_improvement(self, analysis: Dict) -> str:
        """Estimate potential improvement from cleaning this column."""
        null_pct = analysis['null_percentage']
        col_type = analysis['ai_type']
        
        if null_pct > 30:
            return 'high'
        elif null_pct > 10 or col_type in ['datetime', 'text']:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_column_processing_time(self, analysis: Dict) -> str:
        """Estimate processing time for column cleaning."""
        col_type = analysis['ai_type']
        rows = analysis['total_rows']
        
        if col_type == 'text' and rows > 10000:
            return 'long'
        elif col_type in ['datetime', 'categorical'] and analysis['unique_count'] > 1000:
            return 'medium'
        else:
            return 'short'
    
    # Enhanced cleaning methods
    def smart_clean_data(self, user_approved_strategies: Optional[Dict] = None) -> bool:
        """Perform enhanced intelligent data cleaning with advanced techniques."""
        if self.original_data is None:
            self.ai_engine.log_decision("❌ No data loaded for cleaning")
            return False
        
        self.ai_engine.log_decision("🧹 Starting enhanced AI-powered data cleaning...")
        self.cleaned_data = self.original_data.copy()
        original_shape = self.original_data.shape
        
        # Enhanced cleaning pipeline with progress tracking
        cleaning_steps = [
            ("datetime_features", self._clean_datetime_columns),
            ("missing_values", self._enhanced_missing_value_handling),
            ("text_processing", self._advanced_text_processing),
            ("categorical_encoding", self._intelligent_categorical_processing),
            ("outlier_handling", self._advanced_outlier_handling),
            ("data_validation", self._validate_cleaned_data)
        ]
        
        for step_name, step_function in cleaning_steps:
            try:
                self.ai_engine.log_decision(f"🔄 Executing: {step_name}")
                step_function(user_approved_strategies)
                
                self.lineage_tracker.log_transformation(
                    step_name,
                    {'method': step_function.__name__},
                    original_shape,
                    self.cleaned_data.shape
                )
                
            except Exception as e:
                self.ai_engine.log_decision(f"⚠️ Error in {step_name}: {str(e)}")
                continue
        
        # Final summary
        cleaned_shape = self.cleaned_data.shape
        
        self.ai_engine.log_decision("✅ Enhanced data cleaning complete!")
        self.ai_engine.log_decision(f"📊 Original: {original_shape[0]:,} rows × {original_shape[1]} columns")
        self.ai_engine.log_decision(f"📊 Cleaned: {cleaned_shape[0]:,} rows × {cleaned_shape[1]} columns")
        
        if cleaned_shape[0] < original_shape[0]:
            rows_removed = original_shape[0] - cleaned_shape[0]
            self.ai_engine.log_decision(f"🗑️ Removed {rows_removed:,} rows ({rows_removed/original_shape[0]*100:.1f}%) to maintain data quality")
        
        if cleaned_shape[1] != original_shape[1]:
            cols_diff = cleaned_shape[1] - original_shape[1]
            if cols_diff > 0:
                self.ai_engine.log_decision(f"➕ Added {cols_diff} engineered features")
            else:
                self.ai_engine.log_decision(f"➖ Removed {-cols_diff} problematic columns")
        
        return True
    
    def _clean_datetime_columns(self, user_strategies: Optional[Dict] = None):
        """Enhanced datetime column processing with comprehensive feature extraction."""
        datetime_cols = self.data_info.get('datetime_columns', [])
        
        if not datetime_cols:
            return
        
        self.ai_engine.log_decision(f"📅 Processing {len(datetime_cols)} datetime columns with advanced feature extraction...")
        
        for col in datetime_cols:
            if col not in self.cleaned_data.columns:
                continue
                
            try:
                original_shape = self.cleaned_data.shape
                
                # Use advanced feature engineering for datetime
                self.cleaned_data = self.feature_engineer.create_datetime_features(
                    self.cleaned_data, [col]
                )
                
                new_features = self.cleaned_data.shape[1] - original_shape[1]
                self.ai_engine.log_decision(f"  ✅ {col} → extracted {new_features} datetime features")
                
            except Exception as e:
                self.ai_engine.log_decision(f"  ❌ Failed to process {col}: {str(e)}")
    
    def _enhanced_missing_value_handling(self, user_strategies: Optional[Dict] = None):
        """Enhanced missing value handling with advanced imputation techniques."""
        self.ai_engine.log_decision("🩹 AI applying advanced missing value strategies...")
        
        for col in self.cleaned_data.columns:
            if col not in self.column_analyses:
                continue
                
            analysis = self.column_analyses[col]
            strategy_info = analysis['missing_strategy']
            
            if strategy_info['strategy'] == 'none':
                continue
            
            missing_count = self.cleaned_data[col].isnull().sum()
            if missing_count == 0:
                continue
            
            original_shape = self.cleaned_data.shape
            
            # Apply enhanced strategies
            strategy = strategy_info['strategy']
            
            try:
                if strategy == 'iterative':
                    self._apply_iterative_imputation(col, analysis)
                elif strategy == 'knn':
                    self._apply_knn_imputation(col, analysis)
                elif strategy == 'advanced_imputation':
                    self._apply_advanced_imputation(col, analysis)
                elif strategy == 'create_indicator':
                    self._create_missing_indicator(col, analysis)
                elif strategy in ['median', 'mode', 'mode_with_validation']:
                    self._apply_basic_imputation(col, analysis, strategy)
                elif strategy == 'unknown_category':
                    self._apply_unknown_category(col)
                    
                # Log successful imputation
                remaining_missing = self.cleaned_data[col].isnull().sum()
                if remaining_missing < missing_count:
                    filled_count = missing_count - remaining_missing
                    self.ai_engine.log_decision(f"  ✅ {col}: filled {filled_count} values using {strategy}")
                
            except Exception as e:
                self.ai_engine.log_decision(f"  ❌ {col}: {strategy} failed - {str(e)}")
                # Fallback to simple strategy
                self._apply_fallback_imputation(col, analysis)
    
    def _apply_iterative_imputation(self, col: str, analysis: Dict):
        """Apply iterative imputation using advanced estimators."""
        numeric_cols = [c for c in self.cleaned_data.select_dtypes(include=[np.number]).columns 
                       if c != col and self.cleaned_data[c].isnull().sum() < len(self.cleaned_data) * 0.3]
        
        if len(numeric_cols) < 2:
            raise ValueError("Insufficient features for iterative imputation")
        
        # Use up to 10 most correlated features
        if len(numeric_cols) > 10:
            corr_with_target = self.cleaned_data[numeric_cols + [col]].corr()[col].abs().sort_values(ascending=False)
            numeric_cols = corr_with_target.iloc[1:11].index.tolist()
        
        cols_to_impute = [col] + numeric_cols
        
        # Use Random Forest as estimator for better results
        from sklearn.ensemble import RandomForestRegressor
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            max_iter=10,
            random_state=42
        )
        
        imputed_data = imputer.fit_transform(self.cleaned_data[cols_to_impute])
        self.cleaned_data[col] = imputed_data[:, 0]
    
    def _apply_knn_imputation(self, col: str, analysis: Dict):
        """Apply KNN imputation with intelligent neighbor selection."""
        numeric_cols = [c for c in self.cleaned_data.select_dtypes(include=[np.number]).columns 
                       if c != col and self.cleaned_data[c].isnull().sum() < len(self.cleaned_data) * 0.5]
        
        if len(numeric_cols) < 1:
            raise ValueError("No suitable features for KNN imputation")
        
        # Use up to 8 most correlated features
        if len(numeric_cols) > 8:
            corr_with_target = self.cleaned_data[numeric_cols + [col]].corr()[col].abs().sort_values(ascending=False)
            numeric_cols = corr_with_target.iloc[1:9].index.tolist()
        
        cols_to_impute = [col] + numeric_cols
        
        # Adaptive neighbor count based on dataset size
        n_neighbors = min(5, max(3, len(self.cleaned_data) // 100))
        
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_data = imputer.fit_transform(self.cleaned_data[cols_to_impute])
        self.cleaned_data[col] = imputed_data[:, 0]
    
    def _apply_advanced_imputation(self, col: str, analysis: Dict):
        """Apply advanced imputation based on data characteristics."""
        col_type = analysis['ai_type']
        null_pct = analysis['null_percentage']
        
        if col_type == 'numerical' and null_pct > 30:
            # Use multiple imputation with uncertainty estimation
            try:
                self._apply_iterative_imputation(col, analysis)
            except:
                # Fallback to median with noise
                median_val = self.cleaned_data[col].median()
                std_val = self.cleaned_data[col].std()
                noise = np.random.normal(0, std_val * 0.1, self.cleaned_data[col].isnull().sum())
                fill_values = median_val + noise
                self.cleaned_data.loc[self.cleaned_data[col].isnull(), col] = fill_values
        else:
            # Use sophisticated categorical imputation
            self._apply_sophisticated_categorical_imputation(col, analysis)
    
    def _apply_sophisticated_categorical_imputation(self, col: str, analysis: Dict):
        """Apply sophisticated categorical imputation."""
        # Create missing indicator first
        self.cleaned_data[f'{col}_was_missing'] = self.cleaned_data[col].isnull().astype(int)
        
        # Use predictive imputation if possible
        try:
            # Use other categorical columns to predict missing values
            cat_cols = [c for c in self.data_info.get('categorical_columns', []) 
                       if c != col and c in self.cleaned_data.columns 
                       and self.cleaned_data[c].isnull().sum() < len(self.cleaned_data) * 0.3]
            
            if len(cat_cols) >= 2:
                # Use mode within similar groups
                for cat_col in cat_cols[:2]:  # Use top 2 predictive columns
                    group_modes = self.cleaned_data.groupby(cat_col)[col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown')
                    missing_mask = self.cleaned_data[col].isnull()
                    for group, mode_val in group_modes.items():
                        mask = missing_mask & (self.cleaned_data[cat_col] == group)
                        self.cleaned_data.loc[mask, col] = mode_val
            
            # Fill remaining with overall mode
            if self.cleaned_data[col].isnull().any():
                overall_mode = self.cleaned_data[col].mode().iloc[0] if len(self.cleaned_data[col].mode()) > 0 else 'Unknown'
                self.cleaned_data[col].fillna(overall_mode, inplace=True)
                
        except:
            # Simple fallback
            mode_val = self.cleaned_data[col].mode().iloc[0] if len(self.cleaned_data[col].mode()) > 0 else 'Unknown'
            self.cleaned_data[col].fillna(mode_val, inplace=True)
    
    def _create_missing_indicator(self, col: str, analysis: Dict):
        """Create missing indicator and impute."""
        # Create binary missing indicator
        self.cleaned_data[f'{col}_missing'] = self.cleaned_data[col].isnull().astype(int)
        
        # Then impute based on type
        if analysis['ai_type'] == 'numerical':
            fill_value = self.cleaned_data[col].median()
            self.cleaned_data[col].fillna(fill_value, inplace=True)
        else:
            fill_value = 'Missing_Value'
            self.cleaned_data[col].fillna(fill_value, inplace=True)
    
    def _apply_basic_imputation(self, col: str, analysis: Dict, strategy: str):
        """Apply basic imputation strategies."""
        if strategy in ['median', 'mode_with_validation'] and analysis['ai_type'] == 'numerical':
            fill_value = self.cleaned_data[col].median()
            self.cleaned_data[col].fillna(fill_value, inplace=True)
        else:
            # Mode imputation
            mode_series = self.cleaned_data[col].mode()
            fill_value = mode_series.iloc[0] if len(mode_series) > 0 else 'Unknown'
            self.cleaned_data[col].fillna(fill_value, inplace=True)
    
    def _apply_unknown_category(self, col: str):
        """Apply unknown category for text/categorical data."""
        self.cleaned_data[col].fillna('Unknown', inplace=True)
    
    def _apply_fallback_imputation(self, col: str, analysis: Dict):
        """Apply fallback imputation when advanced methods fail."""
        if analysis['ai_type'] == 'numerical':
            fill_value = self.cleaned_data[col].median()
        else:
            mode_series = self.cleaned_data[col].mode()
            fill_value = mode_series.iloc[0] if len(mode_series) > 0 else 'Unknown'
        
        self.cleaned_data[col].fillna(fill_value, inplace=True)
        self.ai_engine.log_decision(f"  🔄 {col}: applied fallback imputation")
    
    def _advanced_text_processing(self, user_strategies: Optional[Dict] = None):
        """Advanced text processing with NLP capabilities."""
        text_cols = [col for col in self.data_info.get('text_columns', []) 
                    if col in self.cleaned_data.columns]
        
        if not text_cols:
            return
        
        self.ai_engine.log_decision(f"📝 Processing {len(text_cols)} text columns with advanced NLP...")
        
        # Basic text cleaning
        for col in text_cols:
            if col not in self.cleaned_data.columns:
                continue
                
            original_unique = self.cleaned_data[col].nunique()
            
            # Enhanced text cleaning
            self.cleaned_data[col] = self.text_processor.basic_text_cleaning(self.cleaned_data[col])
            
            new_unique = self.cleaned_data[col].nunique()
            self.ai_engine.log_decision(f"  🧹 {col}: cleaned text, {original_unique} → {new_unique} unique values")
        
        # Extract comprehensive text features
        if len(text_cols) > 0:
            original_shape = self.cleaned_data.shape
            
            # Extract text features
            self.cleaned_data = self.text_processor.extract_text_features(self.cleaned_data, text_cols)
            
            # Create embeddings for most important text columns (limit to prevent memory issues)
            if len(text_cols) <= 2:
                self.cleaned_data = self.text_processor.create_text_embeddings(
                    self.cleaned_data, text_cols[:1], max_features=5
                )
            
            new_features = self.cleaned_data.shape[1] - original_shape[1]
            if new_features > 0:
                self.ai_engine.log_decision(f"  ✅ Extracted {new_features} text features from {len(text_cols)} columns")
    
    def _intelligent_categorical_processing(self, user_strategies: Optional[Dict] = None):
        """Intelligent categorical data processing."""
        cat_cols = [col for col in self.data_info.get('categorical_columns', []) 
                   if col in self.cleaned_data.columns]
        
        if not cat_cols:
            return
        
        self.ai_engine.log_decision(f"🏷️ Processing {len(cat_cols)} categorical columns with intelligent encoding...")
        
        for col in cat_cols:
            if col not in self.cleaned_data.columns:
                continue
                
            unique_count = self.cleaned_data[col].nunique()
            unique_ratio = unique_count / len(self.cleaned_data)
            
            # Intelligent encoding strategy
            if unique_count <= 10:
                # Low cardinality - one-hot encoding
                dummies = pd.get_dummies(self.cleaned_data[col], prefix=col, dummy_na=False)
                self.cleaned_data = pd.concat([self.cleaned_data, dummies], axis=1)
                self.cleaned_data.drop(col, axis=1, inplace=True)
                self.ai_engine.log_decision(f"  🎯 {col}: one-hot encoded ({unique_count} categories)")
                
            elif unique_count <= 50:
                # Medium cardinality - frequency encoding
                self.cleaned_data = self.feature_engineer.create_frequency_encoding(
                    self.cleaned_data, [col]
                )
                self.ai_engine.log_decision(f"  📊 {col}: frequency encoded ({unique_count} categories)")
                
            else:
                # High cardinality - group rare categories and frequency encode
                value_counts = self.cleaned_data[col].value_counts()
                top_categories = value_counts.head(30).index  # Keep top 30
                self.cleaned_data[col] = self.cleaned_data[col].apply(
                    lambda x: x if x in top_categories else 'Other_Rare'
                )
                
                # Then frequency encode
                self.cleaned_data = self.feature_engineer.create_frequency_encoding(
                    self.cleaned_data, [col]
                )
                
                new_unique = self.cleaned_data[col].nunique()
                self.ai_engine.log_decision(f"  🎯 {col}: grouped rare + frequency encoded ({unique_count} → {new_unique})")
    
    def _advanced_outlier_handling(self, user_strategies: Optional[Dict] = None):
        """Advanced outlier detection and handling."""
        numeric_cols = [col for col in self.data_info.get('numerical_columns', []) 
                       if col in self.cleaned_data.columns]
        
        if not numeric_cols:
            return
        
        self.ai_engine.log_decision(f"📊 Advanced outlier processing for {len(numeric_cols)} numerical columns...")
        
        # Use Isolation Forest for multivariate outlier detection
        if len(numeric_cols) >= 3:
            try:
                numeric_data = self.cleaned_data[numeric_cols].select_dtypes(include=[np.number])
                
                # Handle any remaining NaN values
                numeric_data = numeric_data.fillna(numeric_data.median())
                
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = isolation_forest.fit_predict(numeric_data)
                
                outlier_count = (outlier_labels == -1).sum()
                outlier_pct = (outlier_count / len(self.cleaned_data)) * 100
                
                if outlier_pct < 5:  # Remove if less than 5% of data
                    self.cleaned_data = self.cleaned_data[outlier_labels != -1]
                    self.ai_engine.log_decision(f"  🎯 Isolation Forest: removed {outlier_count} multivariate outliers ({outlier_pct:.1f}%)")
                else:
                    self.ai_engine.log_decision(f"  ⚠️ Isolation Forest: {outlier_pct:.1f}% outliers detected - kept data due to high percentage")
                    
            except Exception as e:
                self.ai_engine.log_decision(f"  ❌ Isolation Forest failed: {str(e)}")
        
        # Individual column outlier handling
        total_outliers_handled = 0
        
        for col in numeric_cols:
            if col not in self.cleaned_data.columns:
                continue
                
            # Enhanced IQR method with adaptive bounds
            Q1 = self.cleaned_data[col].quantile(0.25)
            Q3 = self.cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use adaptive multiplier based on data distribution
            skewness = abs(self.cleaned_data[col].skew())
            multiplier = 2.5 if skewness < 2 else 3.0  # More conservative for skewed data
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers_mask = (self.cleaned_data[col] < lower_bound) | (self.cleaned_data[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                outlier_percentage = outliers_count / len(self.cleaned_data) * 100
                
                if outlier_percentage < 1:  # Remove if < 1% of data
                    self.cleaned_data = self.cleaned_data[~outliers_mask]
                    total_outliers_handled += outliers_count
                    self.ai_engine.log_decision(f"  🗑️ {col}: removed {outliers_count} outliers ({outlier_percentage:.1f}%)")
                elif outlier_percentage < 3:  # Cap if < 3% of data
                    self.cleaned_data[col] = np.clip(self.cleaned_data[col], lower_bound, upper_bound)
                    total_outliers_handled += outliers_count
                    self.ai_engine.log_decision(f"  📏 {col}: capped {outliers_count} outliers ({outlier_percentage:.1f}%)")
                else:
                    # Use robust scaling for high outlier percentage
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    self.cleaned_data[f'{col}_robust_scaled'] = scaler.fit_transform(self.cleaned_data[[col]])
                    self.ai_engine.log_decision(f"  🛡️ {col}: applied robust scaling ({outlier_percentage:.1f}% outliers)")
        
        if total_outliers_handled > 0:
            self.ai_engine.log_decision(f"🧹 Total outliers handled: {total_outliers_handled}")
    
    def _validate_cleaned_data(self, user_strategies: Optional[Dict] = None):
        """Validate cleaned data quality."""
        self.ai_engine.log_decision("🔍 Validating cleaned data quality...")
        
        # Check for remaining missing values
        total_missing = self.cleaned_data.isnull().sum().sum()
        if total_missing > 0:
            missing_cols = self.cleaned_data.columns[self.cleaned_data.isnull().any()].tolist()
            self.ai_engine.log_decision(f"  ⚠️ {total_missing} missing values remain in: {missing_cols}")
        else:
            self.ai_engine.log_decision("  ✅[OK] No missing values remaining")
        
        # Check data types
        non_numeric_cols = self.cleaned_data.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            self.ai_engine.log_decision(f"  📝 Non-numeric columns remaining: {len(non_numeric_cols)}")
            
            # Final encoding for any remaining categorical columns
            for col in non_numeric_cols:
                if col in self.cleaned_data.columns:
                    unique_count = self.cleaned_data[col].nunique()
                    if unique_count <= 20:  # Small enough for label encoding
                        le = LabelEncoder()
                        self.cleaned_data[col] = le.fit_transform(self.cleaned_data[col].astype(str))
                        self.encoders[col] = le
                        self.ai_engine.log_decision(f"    🏷️ {col}: applied label encoding ({unique_count} categories)")
                    else:
                        # Drop if too many categories
                        self.cleaned_data.drop(col, axis=1, inplace=True)
                        self.ai_engine.log_decision(f"    🗑️ {col}: removed due to high cardinality ({unique_count})")
        else:
            self.ai_engine.log_decision("  ✅ All columns are numeric")
        
        # Check for infinite values
        if np.isinf(self.cleaned_data.select_dtypes(include=[np.number])).any().any():
            # Replace infinite values
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                inf_count = np.isinf(self.cleaned_data[col]).sum()
                if inf_count > 0:
                    finite_values = self.cleaned_data[col][np.isfinite(self.cleaned_data[col])]
                    if len(finite_values) > 0:
                        max_val = finite_values.max()
                        min_val = finite_values.min()
                        self.cleaned_data[col] = np.where(np.isposinf(self.cleaned_data[col]), max_val, self.cleaned_data[col])
                        self.cleaned_data[col] = np.where(np.isneginf(self.cleaned_data[col]), min_val, self.cleaned_data[col])
                        self.ai_engine.log_decision(f"    ♾️ {col}: replaced {inf_count} infinite values")
            self.ai_engine.log_decision("  ✅ Infinite values handled")
        
        # Final data quality assessment
        final_quality = self._calculate_quality_improvement()
        if final_quality:
            improvement = final_quality['improvement']
            if improvement > 10:
                quality_rating = "🟢 Excellent"
            elif improvement > 5:
                quality_rating = "🟡 Good" 
            elif improvement > 0:
                quality_rating = "🟠 Moderate"
            else:
                quality_rating = "🔴 Minimal"
            
            self.ai_engine.log_decision(f"  📊 Data quality improvement: {quality_rating} (+{improvement:.1f}%)")
        
        self.ai_engine.log_decision("✅ Data validation complete - ready for machine learning")
    
    # Advanced feature engineering methods
    def advanced_feature_engineering(self, target_column: str = None) -> bool:
        """Perform comprehensive automated feature engineering."""
        if self.cleaned_data is None:
            self.ai_engine.log_decision("❌ No cleaned data available for feature engineering")
            return False
        
        self.ai_engine.log_decision("🔧 Starting advanced automated feature engineering...")
        
        original_shape = self.cleaned_data.shape
        
        # Get column types for feature engineering
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in self.cleaned_data.select_dtypes(include=['object', 'category']).columns 
                           if self.cleaned_data[col].nunique() < 50]
        datetime_cols = self.data_info.get('datetime_columns', [])
        
        try:
            # 1. Polynomial and interaction features
            if len(numeric_cols) >= 2:
                self.ai_engine.log_decision("  🔢 Creating polynomial and interaction features...")
                self.cleaned_data = self.feature_engineer.create_polynomial_features(
                    self.cleaned_data, numeric_cols[:8], degree=2, max_features=30
                )
            
            # 2. Statistical features across numeric columns
            if len(numeric_cols) >= 3:
                self.ai_engine.log_decision("  📊 Creating statistical aggregation features...")
                self.cleaned_data = self.feature_engineer.create_statistical_features(
                    self.cleaned_data, numeric_cols
                )
            
            # 3. Ratio features
            if len(numeric_cols) >= 4:
                self.ai_engine.log_decision("  ➗ Creating ratio features...")
                self.cleaned_data = self.feature_engineer.create_ratio_features(
                    self.cleaned_data, numeric_cols[:10]
                )
            
            # 4. Frequency encoding for categorical variables
            if len(categorical_cols) > 0:
                self.ai_engine.log_decision("  📈 Creating frequency encoding features...")
                self.cleaned_data = self.feature_engineer.create_frequency_encoding(
                    self.cleaned_data, categorical_cols
                )
            
            # 5. Target encoding if target column provided
            if target_column and target_column in self.cleaned_data.columns and len(categorical_cols) > 0:
                self.ai_engine.log_decision(f"  🎯 Creating target encoding features for {target_column}...")
                self.cleaned_data = self.feature_engineer.create_target_encoding(
                    self.cleaned_data, categorical_cols, target_column
                )
            
            # 6. Aggregation features
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                self.ai_engine.log_decision("  📊 Creating aggregation features...")
                self.cleaned_data = self.feature_engineer.create_aggregation_features(
                    self.cleaned_data, numeric_cols[:5], categorical_cols[:3]
                )
            
            new_features = self.cleaned_data.shape[1] - original_shape[1]
            
            self.lineage_tracker.log_transformation(
                'advanced_feature_engineering',
                {
                    'features_created': new_features,
                    'methods': ['polynomial', 'statistical', 'ratio', 'frequency', 'target', 'aggregation']
                },
                original_shape,
                self.cleaned_data.shape
            )
            
            self.ai_engine.log_decision(f"✅ Feature engineering complete: added {new_features} features")
            return True
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error in feature engineering: {str(e)}")
            return False
    
    def knn_imputation(self, columns: List[str] = None, n_neighbors: int = 5) -> bool:
        """Perform KNN imputation on specified columns."""
        if self.cleaned_data is None:
            self.ai_engine.log_decision("❌ No data available for KNN imputation")
            return False
        
        try:
            self.ai_engine.log_decision(f"🔍 Starting KNN imputation with {n_neighbors} neighbors...")
            
            if columns is None:
                # Find columns with missing values
                columns = self.cleaned_data.columns[self.cleaned_data.isnull().any()].tolist()
            
            if not columns:
                self.ai_engine.log_decision("✅ No columns need KNN imputation")
                return True
            
            # Select numeric columns for imputation
            numeric_cols = [col for col in columns if col in self.cleaned_data.select_dtypes(include=[np.number]).columns]
            
            if not numeric_cols:
                self.ai_engine.log_decision("❌ No numeric columns available for KNN imputation")
                return False
            
            # Get additional features for context
            feature_cols = [col for col in self.cleaned_data.select_dtypes(include=[np.number]).columns
                           if col not in numeric_cols and self.cleaned_data[col].isnull().sum() < len(self.cleaned_data) * 0.3]
            
            # Limit features to prevent overfitting
            all_cols = numeric_cols + feature_cols[:10]
            
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            imputed_data = imputer.fit_transform(self.cleaned_data[all_cols])
            
            # Update only the originally missing columns
            for i, col in enumerate(all_cols):
                if col in numeric_cols:
                    original_missing = self.cleaned_data[col].isnull().sum()
                    self.cleaned_data[col] = imputed_data[:, i]
                    self.ai_engine.log_decision(f"  ✅ {col}: imputed {original_missing} values using KNN")
            
            self.lineage_tracker.log_transformation(
                'knn_imputation',
                {'columns': numeric_cols, 'n_neighbors': n_neighbors, 'feature_columns': len(feature_cols)},
                self.cleaned_data.shape,
                self.cleaned_data.shape
            )
            
            self.ai_engine.log_decision(f"✅ KNN imputation complete for {len(numeric_cols)} columns")
            return True
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error in KNN imputation: {str(e)}")
            return False
    
    def iterative_imputation(self, columns: List[str] = None, max_iter: int = 10) -> bool:
        """Perform iterative imputation using advanced estimators."""
        if self.cleaned_data is None:
            self.ai_engine.log_decision("❌ No data available for iterative imputation")
            return False
        
        try:
            self.ai_engine.log_decision(f"🔄 Starting iterative imputation with max {max_iter} iterations...")
            
            if columns is None:
                columns = self.cleaned_data.columns[self.cleaned_data.isnull().any()].tolist()
            
            if not columns:
                self.ai_engine.log_decision("✅ No columns need iterative imputation")
                return True
            
            # Select numeric columns
            numeric_cols = [col for col in columns if col in self.cleaned_data.select_dtypes(include=[np.number]).columns]
            
            if not numeric_cols:
                self.ai_engine.log_decision("❌ No numeric columns available for iterative imputation")
                return False
            
            # Get feature columns
            feature_cols = [col for col in self.cleaned_data.select_dtypes(include=[np.number]).columns
                           if col not in numeric_cols and self.cleaned_data[col].isnull().sum() < len(self.cleaned_data) * 0.2]
            
            all_cols = numeric_cols + feature_cols[:15]
            
            # Use Random Forest for better performance
            from sklearn.ensemble import RandomForestRegressor
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=20, random_state=42),
                max_iter=max_iter,
                random_state=42,
                verbose=False
            )
            
            imputed_data = imputer.fit_transform(self.cleaned_data[all_cols])
            
            # Update columns
            for i, col in enumerate(all_cols):
                if col in numeric_cols:
                    original_missing = self.cleaned_data[col].isnull().sum()
                    self.cleaned_data[col] = imputed_data[:, i]
                    self.ai_engine.log_decision(f"  ✅ {col}: imputed {original_missing} values using iterative method")
            
            self.lineage_tracker.log_transformation(
                'iterative_imputation',
                {'columns': numeric_cols, 'max_iter': max_iter, 'feature_columns': len(feature_cols)},
                self.cleaned_data.shape,
                self.cleaned_data.shape
            )
            
            self.ai_engine.log_decision(f"✅ Iterative imputation complete for {len(numeric_cols)} columns")
            return True
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error in iterative imputation: {str(e)}")
            return False
    
    def text_preprocessing(self, columns: List[str] = None, include_embeddings: bool = False) -> bool:
        """Perform advanced text preprocessing with NLP features."""
        if self.cleaned_data is None:
            self.ai_engine.log_decision("❌ No data available for text preprocessing")
            return False
        
        try:
            if columns is None:
                columns = self.data_info.get('text_columns', [])
            
            # Also check for object columns that might be text
            text_candidates = []
            for col in self.cleaned_data.select_dtypes(include=['object']).columns:
                if col not in columns and self.cleaned_data[col].astype(str).str.len().mean() > 20:
                    text_candidates.append(col)
            
            all_text_cols = list(set(columns + text_candidates))
            
            if not all_text_cols:
                self.ai_engine.log_decision("✅ No text columns found for preprocessing")
                return True
            
            self.ai_engine.log_decision(f"📝 Starting advanced text preprocessing for {len(all_text_cols)} columns...")
            
            original_shape = self.cleaned_data.shape
            
            # Extract comprehensive text features
            self.cleaned_data = self.text_processor.extract_text_features(self.cleaned_data, all_text_cols)
            
            # Create TF-IDF features for short text columns
            short_text_cols = [col for col in all_text_cols 
                             if col in self.cleaned_data.columns and 
                             self.cleaned_data[col].astype(str).str.len().mean() < 200]
            
            if short_text_cols:
                self.ai_engine.log_decision(f"  📊 Creating TF-IDF features for {len(short_text_cols)} short text columns...")
                self.cleaned_data = self.text_processor.create_tfidf_features(
                    self.cleaned_data, short_text_cols[:2], max_features=25
                )
            
            # Create embeddings if requested and available
            if include_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
                self.ai_engine.log_decision("  🧠 Creating text embeddings...")
                self.cleaned_data = self.text_processor.create_text_embeddings(
                    self.cleaned_data, all_text_cols[:1], max_features=8
                )
            
            # Remove original text columns to avoid issues in ML
            for col in all_text_cols:
                if col in self.cleaned_data.columns:
                    self.cleaned_data.drop(col, axis=1, inplace=True)
            
            new_features = self.cleaned_data.shape[1] - original_shape[1] + len(all_text_cols)
            
            self.lineage_tracker.log_transformation(
                'text_preprocessing',
                {
                    'text_columns': all_text_cols,
                    'features_created': new_features,
                    'tfidf_columns': len(short_text_cols),
                    'embeddings_created': include_embeddings
                },
                original_shape,
                self.cleaned_data.shape
            )
            
            self.ai_engine.log_decision(f"✅ Text preprocessing complete: created {new_features} text features")
            return True
            
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error in text preprocessing: {str(e)}")
            return False
    
    def handle_outliers(self, method: str = 'isolation_forest', contamination: float = 0.1) -> bool:
        """Advanced outlier handling with multiple methods."""
        if self.cleaned_data is None:
            self.ai_engine.log_decision("❌ No data available for outlier handling")
            return False
        
        try:
            self.ai_engine.log_decision(f"🎯 Starting outlier handling using {method}...")
            
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                self.ai_engine.log_decision("❌ Insufficient numeric columns for outlier detection")
                return False
            
            original_shape = self.cleaned_data.shape
            
            if method == 'isolation_forest':
                return self._handle_outliers_isolation_forest(numeric_cols, contamination)
            elif method == 'statistical':
                return self._handle_outliers_statistical(numeric_cols)
            elif method == 'robust_scaling':
                return self._handle_outliers_robust_scaling(numeric_cols)
            elif method == 'winsorization':
                return self._handle_outliers_winsorization(numeric_cols)
            else:
                self.ai_engine.log_decision(f"❌ Unknown outlier method: {method}")
                return False
                
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error in outlier handling: {str(e)}")
            return False
    
    def _handle_outliers_isolation_forest(self, numeric_cols: List[str], contamination: float) -> bool:
        """Handle outliers using Isolation Forest."""
        # Handle missing values first
        numeric_data = self.cleaned_data[numeric_cols].fillna(self.cleaned_data[numeric_cols].median())
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        outlier_labels = iso_forest.fit_predict(numeric_data)
        
        outlier_count = (outlier_labels == -1).sum()
        outlier_pct = (outlier_count / len(self.cleaned_data)) * 100
        
        if outlier_pct < 8:  # Remove if less than 8%
            self.cleaned_data = self.cleaned_data[outlier_labels != -1]
            self.ai_engine.log_decision(f"  🎯 Removed {outlier_count} outliers ({outlier_pct:.1f}%) using Isolation Forest")
        else:
            # Create outlier indicator instead
            self.cleaned_data['is_outlier'] = (outlier_labels == -1).astype(int)
            self.ai_engine.log_decision(f"  🚩 Created outlier indicator ({outlier_pct:.1f}% outliers) - too many to remove")
        
        self.lineage_tracker.log_transformation(
            'outlier_handling_isolation_forest',
            {'method': 'isolation_forest', 'contamination': contamination, 'outliers_found': outlier_count},
            (len(self.cleaned_data) + outlier_count, self.cleaned_data.shape[1]) if outlier_pct < 8 else self.cleaned_data.shape,
            self.cleaned_data.shape
        )
        
        return True
    
    def _handle_outliers_statistical(self, numeric_cols: List[str]) -> bool:
        """Handle outliers using statistical methods (IQR)."""
        total_outliers = 0
        
        for col in numeric_cols:
            Q1 = self.cleaned_data[col].quantile(0.25)
            Q3 = self.cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use adaptive bounds based on distribution
            skewness = abs(self.cleaned_data[col].skew())
            multiplier = 2.0 if skewness > 2 else 1.5
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers_mask = (self.cleaned_data[col] < lower_bound) | (self.cleaned_data[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(self.cleaned_data)) * 100
                
                if outlier_pct < 2:
                    # Remove outliers
                    self.cleaned_data = self.cleaned_data[~outliers_mask]
                    total_outliers += outlier_count
                    self.ai_engine.log_decision(f"  🗑️ {col}: removed {outlier_count} outliers ({outlier_pct:.1f}%)")
                else:
                    # Cap outliers
                    self.cleaned_data[col] = np.clip(self.cleaned_data[col], lower_bound, upper_bound)
                    total_outliers += outlier_count
                    self.ai_engine.log_decision(f"  📏 {col}: capped {outlier_count} outliers ({outlier_pct:.1f}%)")
        
        self.ai_engine.log_decision(f"✅ Statistical outlier handling complete: {total_outliers} outliers processed")
        return True
    
    def _handle_outliers_robust_scaling(self, numeric_cols: List[str]) -> bool:
        """Handle outliers using robust scaling."""
        from sklearn.preprocessing import RobustScaler
        
        for col in numeric_cols:
            scaler = RobustScaler()
            self.cleaned_data[f'{col}_robust'] = scaler.fit_transform(self.cleaned_data[[col]])
            self.scalers[f'{col}_robust'] = scaler
        
        self.ai_engine.log_decision(f"✅ Applied robust scaling to {len(numeric_cols)} columns")
        return True
    
    def _handle_outliers_winsorization(self, numeric_cols: List[str], limits: Tuple[float, float] = (0.05, 0.05)) -> bool:
        """Handle outliers using winsorization."""
        from scipy.stats import mstats
        
        for col in numeric_cols:
            original_values = self.cleaned_data[col].copy()
            winsorized = mstats.winsorize(self.cleaned_data[col], limits=limits)
            
            changes = (original_values != winsorized).sum()
            if changes > 0:
                self.cleaned_data[col] = winsorized
                change_pct = (changes / len(self.cleaned_data)) * 100
                self.ai_engine.log_decision(f"  📊 {col}: winsorized {changes} values ({change_pct:.1f}%)")
        
        self.ai_engine.log_decision(f"✅ Winsorization complete for {len(numeric_cols)} columns")
        return True
    
    # Utility and export methods
    def get_cleaning_summary(self) -> Dict:
        """Get comprehensive cleaning summary with enhanced metrics."""
        if self.cleaned_data is None:
            return {"status": "no_cleaning_performed"}
        
        original_shape = self.original_data.shape
        cleaned_shape = self.cleaned_data.shape
        
        summary = {
            'original_shape': original_shape,
            'cleaned_shape': cleaned_shape,
            'rows_removed': original_shape[0] - cleaned_shape[0],
            'columns_added': cleaned_shape[1] - original_shape[1],
            'data_quality_improvement': self._calculate_quality_improvement(),
            'ai_decisions': len(self.ai_engine.reasoning_log),
            'recommendations_applied': len(self.cleaning_recommendations),
            'cleaning_log': self.ai_engine.reasoning_log,
            'feature_engineering_summary': self._get_feature_engineering_summary(),
            'lineage_summary': self.lineage_tracker.get_lineage_summary(),
            'processing_time': (datetime.now() - self.lineage_tracker.start_time).total_seconds()
        }
        
        return summary
    
    def _get_feature_engineering_summary(self) -> Dict:
        """Get feature engineering summary."""
        return {
            'generated_features': len(self.feature_engineer.generated_features),
            'feature_types': {
                'datetime': len([f for f in self.feature_engineer.generated_features if 'year' in f or 'month' in f]),
                'polynomial': len([f for f in self.feature_engineer.generated_features if 'poly_' in f]),
                'text': len([f for f in self.feature_engineer.generated_features if any(text_feat in f for text_feat in ['length', 'word_count', 'embed'])]),
                'statistical': len([f for f in self.feature_engineer.generated_features if any(stat in f for stat in ['mean', 'std', 'min', 'max'])])
            },
            'text_features_created': len(self.text_processor.text_features)
        }
    
    def _calculate_quality_improvement(self) -> Dict:
        """Calculate comprehensive data quality improvement metrics."""
        if self.cleaned_data is None:
            return {}
        
        original_missing = self.original_data.isnull().sum().sum()
        cleaned_missing = self.cleaned_data.isnull().sum().sum()
        
        original_total = self.original_data.shape[0] * self.original_data.shape[1]
        cleaned_total = self.cleaned_data.shape[0] * self.cleaned_data.shape[1]
        
        original_completeness = (1 - original_missing / original_total) * 100
        cleaned_completeness = (1 - cleaned_missing / cleaned_total) * 100 if cleaned_total > 0 else 0
        
        # Calculate additional quality metrics
        original_numeric_cols = len(self.original_data.select_dtypes(include=[np.number]).columns)
        cleaned_numeric_cols = len(self.cleaned_data.select_dtypes(include=[np.number]).columns)
        
        return {
            'original_completeness': original_completeness,
            'cleaned_completeness': cleaned_completeness,
            'improvement': cleaned_completeness - original_completeness,
            'missing_values_handled': max(0, original_missing - cleaned_missing),
            'numeric_columns_ratio': cleaned_numeric_cols / len(self.cleaned_data.columns) * 100,
            'original_numeric_ratio': original_numeric_cols / len(self.original_data.columns) * 100,
            'data_usability_score': self._calculate_usability_score(),
            'processing_efficiency': self._calculate_processing_efficiency()
        }
    
    def _calculate_usability_score(self) -> float:
        """Calculate overall data usability score."""
        if self.cleaned_data is None:
            return 0.0
        
        # Completeness score (40%)
        completeness_score = (1 - self.cleaned_data.isnull().sum().sum() / 
                             (self.cleaned_data.shape[0] * self.cleaned_data.shape[1])) * 40
        
        # Numeric ratio score (30%)
        numeric_ratio = len(self.cleaned_data.select_dtypes(include=[np.number]).columns) / len(self.cleaned_data.columns)
        numeric_score = numeric_ratio * 30
        
        # Feature richness score (20%)
        feature_richness = min(len(self.cleaned_data.columns) / 50, 1.0) * 20  # Optimal around 50 features
        
        # Data size adequacy score (10%)
        size_adequacy = min(len(self.cleaned_data) / 1000, 1.0) * 10  # Adequate at 1000+ rows
        
        return completeness_score + numeric_score + feature_richness + size_adequacy
    
    def _calculate_processing_efficiency(self) -> float:
        """Calculate processing efficiency score."""
        if not hasattr(self.lineage_tracker, 'transformations'):
            return 100.0
        
        total_time = (datetime.now() - self.lineage_tracker.start_time).total_seconds()
        data_size_mb = self.original_data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Efficiency: MB processed per second
        if total_time > 0:
            efficiency = data_size_mb / total_time
        else:
            efficiency = float('inf')
        
        # Normalize to 0-100 scale (assume 1 MB/s is baseline good performance)
        return min(efficiency * 100, 100.0)
    
    def export_cleaned_data(self, file_path: str, format: str = 'csv') -> Tuple[bool, str]:
        """Export cleaned data in various formats."""
        if self.cleaned_data is None:
            return False, "No cleaned data available"
        
        try:
            file_path = Path(file_path)
            
            if format.lower() == 'csv':
                self.cleaned_data.to_csv(file_path, index=False)
            elif format.lower() == 'excel' and EXCEL_AVAILABLE:
                self.cleaned_data.to_excel(file_path, index=False, engine='openpyxl')
            elif format.lower() == 'parquet' and PARQUET_AVAILABLE:
                self.cleaned_data.to_parquet(file_path, index=False)
            elif format.lower() == 'json':
                self.cleaned_data.to_json(file_path, orient='records', indent=2)
            else:
                return False, f"Unsupported format: {format}"
            
            self.ai_engine.log_decision(f"💾 Cleaned data exported to: {file_path}")
            
            # Export metadata
            metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'original_shape': self.original_data.shape,
                'cleaned_shape': self.cleaned_data.shape,
                'data_source': self.data_source,
                'cleaning_summary': self.get_cleaning_summary(),
                'generated_features': self.feature_engineer.generated_features,
                'lineage': self.lineage_tracker.get_lineage_summary()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return True, f"Data and metadata exported successfully to {file_path}"
            
        except Exception as e:
            error_msg = f"Error exporting cleaned data: {str(e)}"
            self.ai_engine.log_decision(f"❌ {error_msg}")
            return False, error_msg
    
    def export_lineage(self, file_path: str) -> bool:
        """Export complete data lineage for audit purposes."""
        try:
            self.lineage_tracker.export_lineage(file_path)
            self.ai_engine.log_decision(f"📋 Data lineage exported to: {file_path}")
            return True
        except Exception as e:
            self.ai_engine.log_decision(f"❌ Error exporting lineage: {str(e)}")
            return False
    
    def get_column_analysis_summary(self) -> List[Dict]:
        """Get formatted column analysis summary for display."""
        if not self.column_analyses:
            return []
        
        summary = []
        for col, analysis in self.column_analyses.items():
            summary.append({
                'column': col,
                'type': analysis['ai_type'],
                'missing_pct': round(analysis['null_percentage'], 1),
                'unique_count': analysis['unique_count'],
                'unique_pct': round(analysis['unique_percentage'], 1),
                'priority': analysis['priority']['level'],
                'strategy': analysis['missing_strategy']['strategy'],
                'reasoning': analysis['reasoning'][:100] + '...' if len(analysis['reasoning']) > 100 else analysis['reasoning']
            })
        
        return summary
    
    def get_meta_features(self) -> Dict:
        """Get dataset meta-features for intelligent processing."""
        return self.meta_features
    
    def get_data_lineage(self) -> Dict:
        """Get complete data transformation lineage."""
        return self.lineage_tracker.get_lineage_summary()
    
    def proceed_with_ml_processing(self, target_column: str = None, problem_type: str = 'auto') -> bool:
        """Continue with ML processing after cleaning (enhanced version)."""
        if self.cleaned_data is None:
            self.ai_engine.log_decision("❌ No cleaned data available for ML processing")
            return False
        
        self.processed_data = self.cleaned_data.copy()
        
        if target_column:
            if target_column not in self.processed_data.columns:
                self.ai_engine.log_decision(f"❌ Target column '{target_column}' not found in cleaned data")
                return False
            
            self.target_column = target_column
            
            if problem_type == 'auto':
                self.problem_type = self._detect_problem_type(self.processed_data[target_column])
            else:
                self.problem_type = problem_type
        else:
            self.problem_type = 'unsupervised'
        
        # Final preparation for ML
        self._prepare_for_ml()
        
        self.ai_engine.log_decision(f"🎯 Ready for ML processing - Problem type: {self.problem_type}")
        
        self.lineage_tracker.log_transformation(
            'ml_preparation',
            {
                'target_column': self.target_column,
                'problem_type': self.problem_type,
                'final_features': len(self.processed_data.columns)
            },
            self.cleaned_data.shape,
            self.processed_data.shape
        )
        
        return True
    
    def _prepare_for_ml(self):
        """Final preparation steps for machine learning."""
        # Ensure all columns are numeric
        non_numeric_cols = self.processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in non_numeric_cols:
            if col != self.target_column:  # Don't encode target here
                # Apply label encoding for remaining categorical columns
                if self.processed_data[col].nunique() <= 100:
                    le = LabelEncoder()
                    self.processed_data[col] = le.fit_transform(self.processed_data[col].astype(str))
                    self.encoders[col] = le
                else:
                    # Drop very high cardinality columns
                    self.processed_data.drop(col, axis=1, inplace=True)
        
        # Handle any remaining infinite values
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(self.processed_data[col]).any():
                finite_values = self.processed_data[col][np.isfinite(self.processed_data[col])]
                if len(finite_values) > 0:
                    max_val = finite_values.max()
                    min_val = finite_values.min()
                    self.processed_data[col] = np.where(np.isposinf(self.processed_data[col]), max_val, self.processed_data[col])
                    self.processed_data[col] = np.where(np.isneginf(self.processed_data[col]), min_val, self.processed_data[col])
        
        # Store feature names for later reference
        self.feature_names = [col for col in self.processed_data.columns if col != self.target_column]
    
    def _detect_problem_type(self, target_series: pd.Series) -> str:
        """Enhanced problem type detection with AI reasoning."""
        unique_vals = target_series.nunique()
        total_vals = len(target_series.dropna())
        
        # Handle edge cases
        if total_vals == 0:
            self.ai_engine.log_decision("⚠️ AI detected: No valid target values - unsupervised learning")
            return 'unsupervised'
        
        if unique_vals == 1:
            self.ai_engine.log_decision("⚠️ AI detected: Only one unique target value - problematic for supervised learning")
            return 'single_class'
        
        # Enhanced detection logic
        if target_series.dtype == 'object' or target_series.dtype.name == 'category':
            self.ai_engine.log_decision(f"🤖 AI detected: Classification problem ({unique_vals} text classes)")
            return 'classification'
        
        # Numeric target analysis
        unique_ratio = unique_vals / total_vals
        
        # Binary classification
        if unique_vals == 2:
            self.ai_engine.log_decision(f"🤖 AI detected: Binary classification ({unique_vals} classes)")
            return 'classification'
        
        # Multi-class classification
        elif unique_vals <= 20 and unique_ratio < 0.05:
            self.ai_engine.log_decision(f"🤖 AI detected: Multi-class classification ({unique_vals} classes, {unique_ratio:.3f} ratio)")
            return 'classification'
        
        # Time series detection
        elif hasattr(self, 'data_info') and len(self.data_info.get('datetime_columns', [])) > 0:
            self.ai_engine.log_decision(f"🤖 AI detected: Time series regression ({unique_vals} unique values with datetime features)")
            return 'time_series'
        
        # Regression
        else:
            self.ai_engine.log_decision(f"🤖 AI detected: Regression problem ({unique_vals} unique values, {unique_ratio:.3f} ratio)")
            return 'regression'
    
    # Legacy compatibility methods
    def log(self, message: str):
        """Add message to processing log (legacy compatibility)."""
        self.processing_log.append(message)
        self.ai_engine.log_decision(message)
    
    def get_processing_log(self) -> List[str]:
        """Get full processing log including AI decisions."""
        combined_log = self.processing_log.copy()
        combined_log.extend([entry['decision'] for entry in self.ai_engine.reasoning_log])
        return combined_log
    
    # Additional utility methods
    def reset(self):
        """Reset the processor to initial state."""
        self.original_data = None
        self.cleaned_data = None
        self.processed_data = None
        self.target_column = None
        self.problem_type = None
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
        self.processing_log = []
        self.data_info = {}
        self.column_analyses = {}
        self.cleaning_recommendations = {}
        self.meta_features = {}
        self.data_source = {'type': None, 'path': None, 'metadata': {}}
        
        # Reset components
        self.ai_engine = EnhancedAIDecisionEngine()
        self.lineage_tracker = DataLineageTracker()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.text_processor = AdvancedTextProcessor()
        
        self.ai_engine.log_decision("🔄 SmartDataProcessor reset to initial state")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        memory_info = {}
        
        if self.original_data is not None:
            memory_info['original_data_mb'] = self.original_data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if self.cleaned_data is not None:
            memory_info['cleaned_data_mb'] = self.cleaned_data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if self.processed_data is not None:
            memory_info['processed_data_mb'] = self.processed_data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        memory_info['total_mb'] = sum(memory_info.values())
        
        return memory_info
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity after processing."""
        if self.cleaned_data is None:
            return {'status': 'no_data', 'valid': False}
        
        integrity_report = {
            'status': 'validated',
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for missing values
        missing_count = self.cleaned_data.isnull().sum().sum()
        if missing_count > 0:
            integrity_report['warnings'].append(f"{missing_count} missing values remain")
        
        # Check for duplicate rows
        duplicate_count = self.cleaned_data.duplicated().sum()
        if duplicate_count > 0:
            integrity_report['warnings'].append(f"{duplicate_count} duplicate rows found")
        
        # Check for infinite values
        numeric_data = self.cleaned_data.select_dtypes(include=[np.number])
        infinite_count = np.isinf(numeric_data).sum().sum()
        if infinite_count > 0:
            integrity_report['issues'].append(f"{infinite_count} infinite values found")
            integrity_report['valid'] = False
        
        # Check for very low variance columns
        low_variance_cols = []
        for col in numeric_data.columns:
            if numeric_data[col].var() < 1e-10:
                low_variance_cols.append(col)
        
        if low_variance_cols:
            integrity_report['warnings'].append(f"Low variance columns: {low_variance_cols}")
        
        # Statistics
        integrity_report['statistics'] = {
            'shape': self.cleaned_data.shape,
            'completeness': (1 - missing_count / (self.cleaned_data.shape[0] * self.cleaned_data.shape[1])) * 100,
            'numeric_columns': len(numeric_data.columns),
            'categorical_columns': len(self.cleaned_data.select_dtypes(include=['object', 'category']).columns)
        }
        
        return integrity_report