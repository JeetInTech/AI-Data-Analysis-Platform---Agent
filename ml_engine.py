import numpy as np
import pandas as pd
import time
import warnings
import json
import pickle
import joblib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from contextlib import contextmanager

# Core ML imports
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, 
                                   RandomizedSearchCV, StratifiedKFold, KFold)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                            GradientBoostingClassifier, GradientBoostingRegressor,
                            VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, 
                                ElasticNet, SGDClassifier, SGDRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.pipeline import Pipeline
# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Advanced ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# AutoML libraries
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

try:
    import autosklearn.classification
    import autosklearn.regression
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False

# Deep Learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# NLP libraries
try:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                            TrainingArguments, Trainer, pipeline)
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Evaluation metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score,
                           mean_squared_error, mean_absolute_error, r2_score, 
                           mean_absolute_percentage_error)

# Model explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

class ModelTrainingProgress:
    """Track and report model training progress."""
    
    def __init__(self):
        self.current_model = None
        self.total_models = 0
        self.completed_models = 0
        self.start_time = None
        self.model_times = {}
        self.callbacks = []
    
    def set_total_models(self, total: int):
        self.total_models = total
        self.start_time = time.time()
    
    def start_model(self, model_name: str):
        self.current_model = model_name
        self.model_times[model_name] = {'start': time.time()}
    
    def complete_model(self, model_name: str, success: bool = True):
        if model_name in self.model_times:
            self.model_times[model_name]['end'] = time.time()
            self.model_times[model_name]['success'] = success
        self.completed_models += 1
        
        # Notify callbacks
        for callback in self.callbacks:
            callback(model_name, self.completed_models, self.total_models, success)
    
    def get_progress(self) -> Dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'completed': self.completed_models,
            'total': self.total_models,
            'current': self.current_model,
            'progress_pct': (self.completed_models / self.total_models * 100) if self.total_models > 0 else 0,
            'elapsed_time': elapsed,
            'estimated_remaining': (elapsed / max(self.completed_models, 1)) * (self.total_models - self.completed_models)
        }

class DeepLearningModels:
    """Deep learning model implementations."""
    
    @staticmethod
    def create_pytorch_mlp(input_size: int, output_size: int, problem_type: str, 
                          hidden_sizes: List[int] = [128, 64, 32]) -> nn.Module:
        """Create PyTorch MLP for tabular data."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        if problem_type == 'classification' and output_size > 2:
            layers.append(nn.Linear(prev_size, output_size))
        elif problem_type == 'classification':
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())
        else:  # regression
            layers.append(nn.Linear(prev_size, 1))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_tensorflow_model(input_size: int, output_size: int, problem_type: str,
                              hidden_sizes: List[int] = [128, 64, 32]) -> 'tf.keras.Model':
        """Create TensorFlow model for tabular data."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_size,)))
        
        for hidden_size in hidden_sizes:
            model.add(keras.layers.Dense(hidden_size, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.2))
        
        # Output layer
        if problem_type == 'classification' and output_size > 2:
            model.add(keras.layers.Dense(output_size, activation='softmax'))
        elif problem_type == 'classification':
            model.add(keras.layers.Dense(1, activation='sigmoid'))
        else:  # regression
            model.add(keras.layers.Dense(1))
        
        return model

class NLPPipeline:
    """Natural Language Processing pipeline for text classification."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.model_name = "distilbert-base-uncased"
    
    def setup_for_classification(self, num_labels: int = 2):
        """Setup NLP pipeline for text classification."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )
            self.pipeline = pipeline("text-classification", 
                                   model=self.model, 
                                   tokenizer=self.tokenizer)
            return True
        except Exception as e:
            print(f"Error setting up NLP pipeline: {e}")
            return False
    
    def train_on_text_data(self, X_text: List[str], y: List, validation_split: float = 0.2):
        """Train model on text data."""
        # This would implement custom training with Trainer API
        # Simplified for now - in practice would need custom dataset class
        try:
            # For demonstration - would need full implementation
            results = {"accuracy": 0.85, "f1": 0.83}  # Placeholder
            return results
        except Exception as e:
            print(f"Error training NLP model: {e}")
            return None
    
    def predict_text(self, texts: List[str]) -> List:
        """Predict on text data."""
        if self.pipeline:
            return [result['label'] for result in self.pipeline(texts)]
        return []

class EnsembleMethods:
    """Advanced ensemble methods and model combination strategies."""
    
    def __init__(self):
        self.ensemble_models = {}
    
    def create_voting_ensemble(self, models: Dict, problem_type: str, voting: str = 'soft'):
        """Create voting ensemble from trained models."""
        model_list = [(name, model) for name, model in models.items() 
                     if hasattr(model, 'predict')]
        
        if problem_type == 'classification':
            ensemble = VotingClassifier(estimators=model_list, voting=voting)
        else:
            ensemble = VotingRegressor(estimators=model_list)
        
        return ensemble
    
    def create_stacking_ensemble(self, models: Dict, problem_type: str, cv: int = 5):
        """Create stacking ensemble with meta-learner."""
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        
        base_models = [(name, model) for name, model in models.items()]
        
        if problem_type == 'classification':
            meta_clf = LogisticRegression(random_state=42)
            ensemble = StackingClassifier(estimators=base_models, 
                                        final_estimator=meta_clf, cv=cv)
        else:
            meta_reg = Ridge(random_state=42)
            ensemble = StackingRegressor(estimators=base_models, 
                                       final_estimator=meta_reg, cv=cv)
        
        return ensemble
    
    def create_blending_ensemble(self, models: Dict, X_blend: np.ndarray, 
                               y_blend: np.ndarray, problem_type: str):
        """Create blending ensemble using holdout set."""
        # Get predictions from all models on blend set
        blend_predictions = []
        for name, model in models.items():
            try:
                if problem_type == 'classification' and hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_blend)
                    if pred.shape[1] == 2:  # Binary classification
                        pred = pred[:, 1:2]  # Take positive class probability
                else:
                    pred = model.predict(X_blend).reshape(-1, 1)
                blend_predictions.append(pred)
            except Exception as e:
                print(f"Error getting predictions from {name}: {e}")
                continue
        
        if not blend_predictions:
            return None
        
        # Stack predictions horizontally
        X_blend_meta = np.hstack(blend_predictions)
        
        # Train meta-model
        if problem_type == 'classification':
            meta_model = LogisticRegression(random_state=42)
        else:
            meta_model = Ridge(random_state=42)
        
        meta_model.fit(X_blend_meta, y_blend)
        
        return {'meta_model': meta_model, 'base_models': models}

class SmartMLEngine:
    """Enhanced ML engine with multi-paradigm support, AutoML, and advanced features."""
    
    def __init__(self):
        # Core attributes
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.ensemble_models = {}
        
        # Data attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        
        # Configuration
        self.problem_type = None
        self.feature_names = []
        self.target_encoder = None
        self.feature_scaler = None
        
        # Training state
        self.training_log = []
        self.model_metadata = {}
        self.training_history = {}
        self.cross_validation_results = {}
        
        # Advanced components
        self.progress_tracker = ModelTrainingProgress()
        self.dl_models = DeepLearningModels()
        self.nlp_pipeline = NLPPipeline()
        self.ensemble_methods = EnsembleMethods()
        
        # Configuration flags
        self.use_multiprocessing = True
        self.max_workers = min(4, multiprocessing.cpu_count())
        self.enable_early_stopping = True
        self.auto_feature_selection = False
        
        # Model persistence
        self.model_save_path = Path("saved_models")
        self.model_save_path.mkdir(exist_ok=True)
    
    def configure_training(self, use_multiprocessing: bool = True, max_workers: int = None,
                         enable_early_stopping: bool = True, auto_feature_selection: bool = False):
        """Configure training parameters."""
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.enable_early_stopping = enable_early_stopping
        self.auto_feature_selection = auto_feature_selection
        
        self.log(f"[CONFIG] Training configured: MP={use_multiprocessing}, Workers={self.max_workers}, "
                f"EarlyStopping={enable_early_stopping}, FeatureSelection={auto_feature_selection}")
    
    def prepare_data(self, processed_data: pd.DataFrame, target_column: str, 
                    problem_type: str = 'auto', test_size: float = 0.2, 
                    val_size: float = 0.1, **kwargs) -> bool:
        """Enhanced data preparation with validation set and advanced preprocessing."""
        try:
            self.log("[PREPARE] Starting enhanced data preparation...")
            
            # Smart missing value handling instead of blanket removal
            processed_data = self._smart_handle_missing_values(processed_data, target_column)
            
            # Separate features and target
            if target_column not in processed_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            X = processed_data.drop(target_column, axis=1)
            y = processed_data[target_column]
            
            self.log(f"[DATA] Features shape: {X.shape}")
            self.log(f"[DATA] Target shape: {y.shape}")
            
            # Auto-detect problem type with enhanced logic
            if problem_type == 'auto':
                self.problem_type = self._enhanced_problem_type_detection(y, X)
            else:
                self.problem_type = problem_type
            
            self.log(f"[DETECT] Problem type: {self.problem_type.upper()}")
            
            # Handle target encoding
            if self.problem_type == 'classification':
                y = self._encode_target_for_classification(y)
            elif self.problem_type == 'time_series':
                # Special handling for time series
                return self._prepare_time_series_data(X, y, **kwargs)
            
            # Enhanced feature preprocessing
            X = self._enhanced_preprocess_features(X)
            self.feature_names = list(X.columns)
            
            # Intelligent feature selection if enabled
            if self.auto_feature_selection:
                X = self._intelligent_feature_selection(X, y)
            
            # Split data with validation set
            self._split_data_with_validation(X, y, test_size, val_size)
            
            # Final validation
            self._validate_prepared_data()
            
            self.log(f"[SUCCESS] Data prepared successfully!")
            self.log(f"[INFO] Train: {len(self.X_train)}, Val: {len(self.X_val) if self.X_val is not None else 0}, "
                    f"Test: {len(self.X_test)} samples")
            self.log(f"[INFO] Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            self.log(f"[ERROR] Error preparing data: {str(e)}")
            return False
    
    def _prepare_time_series_data(self, X: pd.DataFrame, y: pd.Series, 
                                 time_column: str = None, **kwargs) -> bool:
        """Prepare data for time series analysis."""
        self.log("📅 Preparing time series data...")
        
        # Time series specific preparation would go here
        # For now, fall back to standard preparation
        self.problem_type = 'regression'  # Most time series are regression
        
        X = self._preprocess_features(X)
        self.feature_names = list(X.columns)
        
        # Time series split (chronological)
        split_point = int(len(X) * 0.8)
        val_split_point = int(len(X) * 0.9)
        
        self.X_train = X.iloc[:split_point]
        self.X_val = X.iloc[split_point:val_split_point]
        self.X_test = X.iloc[val_split_point:]
        
        self.y_train = y.iloc[:split_point]
        self.y_val = y.iloc[split_point:val_split_point]
        self.y_test = y.iloc[val_split_point:]
        
        self.log("✅ Time series data prepared with chronological splits")
        return True
    
    def _split_data_with_validation(self, X: pd.DataFrame, y: pd.Series, 
                                  test_size: float, val_size: float):
        """Split data into train/validation/test sets."""
        # First split: separate test set
        stratify_param = self._determine_stratification(y)
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        
        # Second split: separate validation from remaining data
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
            stratify_temp = self._determine_stratification(y_temp)
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42, 
                stratify=stratify_temp
            )
        else:
            self.X_train, self.y_train = X_temp, y_temp
            self.X_val, self.y_val = None, None
    
    def _enhanced_preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature preprocessing with intelligent transformations."""
        X = X.copy()
        
        # Remove any remaining emojis and special characters
        X = self._clean_text_features(X)
        
        # Intelligent numeric conversion
        X = self._intelligent_numeric_conversion(X)
        
        # Feature engineering for datetime columns
        X = self._engineer_datetime_features(X)
        
        # Handle high cardinality categorical features
        X = self._handle_high_cardinality_features(X)
        
        # Intelligent scaling based on feature distribution
        X = self._intelligent_feature_scaling(X)
        
        # Create interaction features for small feature sets
        if len(X.columns) < 20:
            X = self._create_interaction_features(X)
        
        return X
    
    def _clean_text_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove emojis and clean text features."""
        X = X.copy()
        # Define emoji pattern
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)
        
        for col in X.select_dtypes(include=['object']).columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype(str).apply(lambda x: emoji_pattern.sub(r'', x).strip())
                # Clean multiple spaces
                X[col] = X[col].apply(lambda x: re.sub(r'\s+', ' ', x))
                
        self.log("[CLEAN] Removed emojis and cleaned text features")
        return X
    
    def _intelligent_numeric_conversion(self, X: pd.DataFrame) -> pd.DataFrame:
        """Intelligently convert features to numeric with better error handling."""
        X = X.copy()
        columns_to_drop = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Try direct conversion first
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if not X[col].isnull().all():
                        # Fill NaN with median for successfully converted columns
                        median_val = X[col].median()
                        X[col].fillna(median_val, inplace=True)
                        self.log(f"[CONVERT] Converted '{col}' to numeric")
                        continue
                except:
                    pass
                
                # Try label encoding for categorical
                try:
                    unique_ratio = X[col].nunique() / len(X[col])
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        self.log(f"[ENCODE] Label encoded '{col}' ({X[col].nunique()} categories)")
                    else:
                        # High cardinality - drop or encode differently
                        if X[col].nunique() > 1000:
                            columns_to_drop.append(col)
                            self.log(f"[DROP] Dropping high cardinality column '{col}' ({X[col].nunique()} categories)")
                        else:
                            # Use frequency encoding for medium cardinality
                            freq_encoding = X[col].value_counts().to_dict()
                            X[col] = X[col].map(freq_encoding)
                            self.log(f"[ENCODE] Frequency encoded '{col}'")
                except Exception as e:
                    columns_to_drop.append(col)
                    self.log(f"[ERROR] Failed to convert '{col}': {str(e)}")
        
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
            self.log(f"[DROP] Dropped {len(columns_to_drop)} unconvertible columns")
        
        return X
    
    def _engineer_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from datetime columns."""
        X = X.copy()
        datetime_cols = []
        
        # Detect datetime columns
        for col in X.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    X[col] = pd.to_datetime(X[col], errors='coerce')
                    if not X[col].isnull().all():
                        datetime_cols.append(col)
                except:
                    continue
        
        # Engineer features from datetime columns
        for col in datetime_cols:
            if X[col].dtype == 'datetime64[ns]':
                # Extract useful datetime components
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                
                # Drop original datetime column
                X = X.drop(columns=[col])
                self.log(f"[ENGINEER] Created datetime features from '{col}'")
        
        return X
    
    def _handle_high_cardinality_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle high cardinality categorical features intelligently."""
        X = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                cardinality = X[col].nunique()
                if cardinality > 50:  # High cardinality threshold
                    # Use target encoding or frequency encoding
                    value_counts = X[col].value_counts()
                    # Keep top N categories, group others as 'Other'
                    top_categories = value_counts.head(20).index.tolist()
                    X[col] = X[col].apply(lambda x: x if x in top_categories else 'Other')
                    self.log(f"[REDUCE] Reduced cardinality of '{col}' from {cardinality} to {X[col].nunique()}")
        
        return X
    
    def _intelligent_feature_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent feature scaling based on distribution."""
        X = X.copy()
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return X
        
        # Analyze feature distributions
        scaling_strategy = {}
        for col in numeric_cols:
            skewness = abs(X[col].skew())
            has_outliers = self._detect_outliers(X[col])
            
            if skewness > 2 or has_outliers:
                scaling_strategy[col] = 'robust'
            elif X[col].min() >= 0:  # All positive values
                scaling_strategy[col] = 'minmax'
            else:
                scaling_strategy[col] = 'standard'
        
        # Apply appropriate scaling
        if len(scaling_strategy) > 0:
            scalers = {}
            for strategy in ['robust', 'minmax', 'standard']:
                cols = [col for col, strat in scaling_strategy.items() if strat == strategy]
                if cols:
                    if strategy == 'robust':
                        scaler = RobustScaler()
                    elif strategy == 'minmax':
                        scaler = MinMaxScaler()
                    else:
                        scaler = StandardScaler()
                    
                    X[cols] = scaler.fit_transform(X[cols])
                    scalers[strategy] = scaler
                    self.log(f"[SCALE] Applied {strategy} scaling to {len(cols)} features")
            
            self.feature_scalers = scalers  # Store multiple scalers
        
        return X
    
    def _detect_outliers(self, series: pd.Series) -> bool:
        """Detect if a series has significant outliers."""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]
            return len(outliers) / len(series) > 0.05  # More than 5% outliers
        except:
            return False
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for small feature sets."""
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 10:  # Only for small feature sets
            interactions_created = 0
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if interactions_created < 5:  # Limit interactions
                        # Create multiplicative interaction
                        interaction_name = f"{col1}_x_{col2}"
                        X[interaction_name] = X[col1] * X[col2]
                        interactions_created += 1
            
            if interactions_created > 0:
                self.log(f"[ENGINEER] Created {interactions_created} interaction features")
        
        return X
    
    def _intelligent_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Perform intelligent feature selection with multiple methods."""
        self.log("[SELECT] Starting intelligent feature selection...")
        
        try:
            original_features = len(X.columns)
            
            # Step 1: Remove constant and quasi-constant features
            X = self._remove_constant_features(X)
            
            # Step 2: Remove highly correlated features
            X = self._remove_correlated_features(X)
            
            # Step 3: Statistical feature selection
            if len(X.columns) > 100:  # Only if we have many features
                if self.problem_type == 'classification':
                    # Use mutual information for classification
                    selector = SelectKBest(score_func=mutual_info_classif, k=min(80, len(X.columns)))
                else:
                    # Use mutual information for regression
                    selector = SelectKBest(score_func=mutual_info_regression, k=min(80, len(X.columns)))
                
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                
                self.log(f"[SELECT] Statistical selection: {len(selected_features)} features")
            
            # Step 4: Tree-based feature importance (if not too many features)
            if len(X.columns) > 20 and len(X.columns) <= 200:
                X = self._tree_based_feature_selection(X, y)
            
            self.log(f"[SELECT] Final selection: {len(X.columns)} from {original_features} features")
            
            return X
            
        except Exception as e:
            self.log(f"[WARNING] Feature selection failed: {e}, using all features")
            return X
    
    def _remove_constant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant and quasi-constant features."""
        X = X.copy()
        constant_features = []
        
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
            elif X[col].nunique() / len(X[col]) < 0.01:  # Less than 1% unique values
                constant_features.append(col)
        
        if constant_features:
            X = X.drop(columns=constant_features)
            self.log(f"[REMOVE] Removed {len(constant_features)} constant/quasi-constant features")
        
        return X
    
    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        X = X.copy()
        numeric_X = X.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) < 2:
            return X
        
        try:
            corr_matrix = numeric_X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            highly_correlated = [column for column in upper_triangle.columns 
                               if any(upper_triangle[column] > threshold)]
            
            if highly_correlated:
                X = X.drop(columns=highly_correlated)
                self.log(f"[REMOVE] Removed {len(highly_correlated)} highly correlated features")
        
        except Exception as e:
            self.log(f"[WARNING] Correlation analysis failed: {e}")
        
        return X
    
    def _tree_based_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Use tree-based model for feature selection."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.feature_selection import SelectFromModel
            
            if self.problem_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Use median threshold for feature selection
            selector = SelectFromModel(estimator, threshold='median')
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            if len(selected_features) > 0:
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                self.log(f"[SELECT] Tree-based selection: {len(selected_features)} features")
            
        except Exception as e:
            self.log(f"[WARNING] Tree-based selection failed: {e}")
        
        return X
    
    def train_all_models(self, quick_mode: bool = False, automl: bool = False, 
                        deep_learning: bool = False, cross_validation: bool = True,
                        feature_selection: bool = False, **kwargs) -> bool:
        """Enhanced model training with multiple paradigms."""
        if self.X_train is None:
            self.log("❌ No training data available")
            return False
        
        self.log("🚀 Starting enhanced multi-paradigm model training...")
        
        # Collect all models to train
        models_to_train = {}
        
        # Classical ML models
        classical_models = self._get_classical_models(quick_mode)
        models_to_train.update(classical_models)
        
        # Advanced gradient boosting models
        if LIGHTGBM_AVAILABLE or XGBOOST_AVAILABLE or CATBOOST_AVAILABLE:
            advanced_models = self._get_advanced_models(quick_mode)
            models_to_train.update(advanced_models)
        
        # AutoML models
        if automl and (FLAML_AVAILABLE or AUTOSKLEARN_AVAILABLE):
            automl_models = self._get_automl_models(quick_mode)
            models_to_train.update(automl_models)
        
        # Deep Learning models
        if deep_learning and (PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE):
            dl_models = self._get_deep_learning_models()
            models_to_train.update(dl_models)
        
        if not models_to_train:
            self.log("❌ No models available for training")
            return False
        
        # Setup progress tracking
        self.progress_tracker.set_total_models(len(models_to_train))
        
        # Train models
        if self.use_multiprocessing and len(models_to_train) > 2:
            success = self._train_models_parallel(models_to_train, quick_mode, cross_validation)
        else:
            success = self._train_models_sequential(models_to_train, quick_mode, cross_validation)
        
        if success and len(self.models) > 0:
            # Find best model
            self._find_best_model()
            
            # Create ensemble if multiple models succeeded
            if len(self.models) > 2:
                self._create_ensemble_models()
            
            self.log(f"🎉 Training completed! {len(self.models)}/{len(models_to_train)} models trained successfully")
            self.log(f"🏆 Best model: {self.best_model_name}")
            
            return True
        else:
            self.log("❌ All models failed to train")
            return False
    
    def _train_models_parallel(self, models_to_train: Dict, quick_mode: bool, 
                              cross_validation: bool) -> bool:
        """Train models in parallel using multiprocessing."""
        self.log(f"⚡ Training {len(models_to_train)} models in parallel with {self.max_workers} workers...")
        
        successful_models = 0
        
        def train_single_model(model_info):
            name, (model, params) = model_info
            return self._train_single_model_safe(name, model, params, quick_mode, cross_validation)
        
        # Use ThreadPoolExecutor for I/O bound tasks (sklearn models work better with threads)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(train_single_model, item): item[0] 
                      for item in models_to_train.items()}
            
            for future in futures:
                model_name = futures[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per model
                    if result:
                        successful_models += 1
                        self.log(f"  ✅ {model_name} completed successfully")
                    else:
                        self.log(f"  ❌ {model_name} failed")
                except Exception as e:
                    self.log(f"  ❌ {model_name} failed with exception: {str(e)}")
                
                self.progress_tracker.complete_model(model_name, result if 'result' in locals() else False)
        
        return successful_models > 0
    
    def _train_models_sequential(self, models_to_train: Dict, quick_mode: bool, 
                                cross_validation: bool) -> bool:
        """Train models sequentially."""
        self.log(f"🔄 Training {len(models_to_train)} models sequentially...")
        
        successful_models = 0
        
        for name, (model, params) in models_to_train.items():
            self.progress_tracker.start_model(name)
            
            if self._train_single_model_safe(name, model, params, quick_mode, cross_validation):
                successful_models += 1
                self.log(f"  ✅ {name} completed successfully")
                self.progress_tracker.complete_model(name, True)
            else:
                self.log(f"  ❌ {name} failed")
                self.progress_tracker.complete_model(name, False)
        
        return successful_models > 0
    
    def _train_single_model_safe(self, name: str, model: Any, params: Dict, 
                                quick_mode: bool, cross_validation: bool) -> bool:
        """Safely train a single model with error handling."""
        try:
            start_time = time.time()
            
            # Handle different model types
            if 'automl' in name.lower():
                trained_model = self._train_automl_model(name, model, params)
            elif 'deep' in name.lower() or 'neural' in name.lower():
                trained_model = self._train_deep_learning_model(name, model, params)
            elif 'nlp' in name.lower():
                trained_model = self._train_nlp_model(name, model, params)
            else:
                trained_model = self._train_classical_model(name, model, params, quick_mode)
            
            if trained_model is None:
                return False
            
            # Store model and evaluate
            self.models[name] = trained_model
            
            # Evaluate model
            scores = self._evaluate_model_comprehensive(name, trained_model, cross_validation)
            self.results[name] = scores
            
            # Store training metadata
            train_time = time.time() - start_time
            self.model_metadata[name] = {
                'model_type': type(trained_model).__name__,
                'training_time': train_time,
                'parameters': params if not quick_mode else {},
                'feature_count': len(self.feature_names),
                'training_samples': len(self.X_train)
            }
            
            return True
            
        except Exception as e:
            self.log(f"  ❌ Error training {name}: {str(e)}")
            return False
    
    def _train_classical_model(self, name: str, model: Any, params: Dict, quick_mode: bool):
        """Train classical ML model with hyperparameter tuning."""
        if not quick_mode and params:
            # Use RandomizedSearchCV for faster hyperparameter search
            search = RandomizedSearchCV(
                model, params, n_iter=20, cv=3, 
                scoring=self._get_scoring_metric(),
                n_jobs=-1, random_state=42, error_score='raise'
            )
            search.fit(self.X_train, self.y_train)
            
            # Store hyperparameter search results
            self.training_history[name] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
            
            return search.best_estimator_
        else:
            model.fit(self.X_train, self.y_train)
            return model
    
    def _train_automl_model(self, name: str, automl_instance: Any, params: Dict):
        """Train AutoML model."""
        try:
            if FLAML_AVAILABLE and isinstance(automl_instance, AutoML):
                # FLAML AutoML
                time_budget = params.get('time_budget', 60)  # 1 minute default
                
                automl_instance.fit(
                    self.X_train, self.y_train,
                    task='classification' if self.problem_type == 'classification' else 'regression',
                    time_budget=time_budget,
                    verbose=0
                )
                
                self.training_history[name] = {
                    'best_estimator': str(automl_instance.best_estimator),
                    'best_config': automl_instance.best_config,
                    'best_loss': automl_instance.best_loss
                }
                
                return automl_instance
            
            elif AUTOSKLEARN_AVAILABLE:
                # Auto-sklearn
                time_budget = params.get('time_budget', 60)
                
                if self.problem_type == 'classification':
                    automl_instance.fit(self.X_train.values, self.y_train.values, 
                                      time_left_for_this_task=time_budget)
                else:
                    automl_instance.fit(self.X_train.values, self.y_train.values, 
                                      time_left_for_this_task=time_budget)
                
                return automl_instance
            
            return None
            
        except Exception as e:
            self.log(f"  ⚠️ AutoML training failed: {e}")
            return None
    
    def _train_deep_learning_model(self, name: str, model_info: Dict, params: Dict):
        """Train deep learning model."""
        try:
            framework = model_info.get('framework', 'pytorch')
            
            if framework == 'pytorch' and PYTORCH_AVAILABLE:
                return self._train_pytorch_model(name, model_info, params)
            elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
                return self._train_tensorflow_model(name, model_info, params)
            else:
                return None
                
        except Exception as e:
            self.log(f"  ⚠️ Deep learning training failed: {e}")
            return None
    
    def _train_pytorch_model(self, name: str, model_info: Dict, params: Dict):
        """Train PyTorch model."""
        # Determine output size
        if self.problem_type == 'classification':
            output_size = len(np.unique(self.y_train))
        else:
            output_size = 1
        
        # Create model
        model = self.dl_models.create_pytorch_mlp(
            input_size=len(self.feature_names),
            output_size=output_size,
            problem_type=self.problem_type,
            hidden_sizes=params.get('hidden_sizes', [128, 64, 32])
        )
        
        # Prepare data
        X_tensor = torch.FloatTensor(self.X_train.values)
        y_tensor = torch.FloatTensor(self.y_train.values.reshape(-1, 1)) if self.problem_type == 'regression' else torch.LongTensor(self.y_train.values)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=params.get('batch_size', 32), shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss() if self.problem_type == 'regression' else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        
        # Training loop
        epochs = params.get('epochs', 100)
        model.train()
        
        training_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if self.problem_type == 'classification':
                    if output_size == 2:  # Binary classification
                        outputs = outputs.squeeze()
                        batch_y = batch_y.float()
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            training_losses.append(epoch_loss / len(dataloader))
            
            # Early stopping
            if self.enable_early_stopping and len(training_losses) > 10:
                if all(training_losses[-5:][i] >= training_losses[-5:][i+1] for i in range(4)):
                    self.log(f"    Early stopping at epoch {epoch}")
                    break
        
        # Store training history
        self.training_history[name] = {
            'training_losses': training_losses,
            'epochs_trained': len(training_losses),
            'final_loss': training_losses[-1] if training_losses else float('inf')
        }
        
        # Create wrapper for sklearn compatibility
        class PyTorchWrapper:
            def __init__(self, model, problem_type, scaler=None):
                self.model = model
                self.problem_type = problem_type
                self.scaler = scaler
                
            def predict(self, X):
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
                    outputs = self.model(X_tensor)
                    
                    if self.problem_type == 'classification':
                        if outputs.dim() > 1 and outputs.shape[1] > 1:
                            return torch.argmax(outputs, dim=1).numpy()
                        else:
                            return (torch.sigmoid(outputs) > 0.5).int().numpy().flatten()
                    else:
                        return outputs.numpy().flatten()
            
            def predict_proba(self, X):
                if self.problem_type != 'classification':
                    raise AttributeError("predict_proba not available for regression")
                
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
                    outputs = self.model(X_tensor)
                    
                    if outputs.dim() == 1 or outputs.shape[1] == 1:  # Binary classification
                        proba = torch.sigmoid(outputs).numpy().flatten()
                        return np.column_stack([1 - proba, proba])
                    else:  # Multi-class
                        return torch.softmax(outputs, dim=1).numpy()
        
        return PyTorchWrapper(model, self.problem_type, self.feature_scaler)
    
    def _train_tensorflow_model(self, name: str, model_info: Dict, params: Dict):
        """Train TensorFlow model."""
        # Determine output size
        if self.problem_type == 'classification':
            output_size = len(np.unique(self.y_train))
        else:
            output_size = 1
        
        # Create model
        model = self.dl_models.create_tensorflow_model(
            input_size=len(self.feature_names),
            output_size=output_size,
            problem_type=self.problem_type,
            hidden_sizes=params.get('hidden_sizes', [128, 64, 32])
        )
        
        # Compile model
        if self.problem_type == 'classification':
            if output_size == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss=loss,
            metrics=metrics
        )
        
        # Prepare validation data
        validation_data = None
        if self.X_val is not None and self.y_val is not None:
            validation_data = (self.X_val.values, self.y_val.values)
        
        # Training callbacks
        callbacks = []
        if self.enable_early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            )
        
        # Train model
        history = model.fit(
            self.X_train.values, self.y_train.values,
            epochs=params.get('epochs', 100),
            batch_size=params.get('batch_size', 32),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        # Store training history
        self.training_history[name] = {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', []),
            'metrics': {k: v for k, v in history.history.items() if k not in ['loss', 'val_loss']}
        }
        
        return model
    
    def _train_nlp_model(self, name: str, model_info: Dict, params: Dict):
        """Train NLP model for text classification."""
        try:
            # This would need text data - for now return None
            # In practice, would check for text columns and train accordingly
            self.log(f"  ⚠️ NLP training not fully implemented yet")
            return None
        except Exception as e:
            self.log(f"  ⚠️ NLP training failed: {e}")
            return None
    
    def _get_classical_models(self, quick_mode: bool) -> Dict:
        """Get classical ML models with parameters."""
        models = {}
        
        if self.problem_type == 'classification':
            models.update({
                'Random Forest': (
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    {} if quick_mode else {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10]
                    }
                ),
                'Gradient Boosting': (
                    GradientBoostingClassifier(random_state=42),
                    {} if quick_mode else {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                ),
                'Logistic Regression': (
                    LogisticRegression(random_state=42, max_iter=1000),
                    {} if quick_mode else {
                        'C': [0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear']
                    }
                )
            })
            
            # Add SVM for smaller datasets
            if len(self.X_train) < 5000:
                models['SVM'] = (
                    SVC(random_state=42, probability=True),
                    {} if quick_mode else {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear']
                    }
                )
        
        else:  # Regression
            models.update({
                'Random Forest': (
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    {} if quick_mode else {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10]
                    }
                ),
                'Gradient Boosting': (
                    GradientBoostingRegressor(random_state=42),
                    {} if quick_mode else {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                ),
                'Ridge': (
                    Ridge(random_state=42),
                    {} if quick_mode else {'alpha': [0.1, 1, 10, 100]}
                ),
                'Lasso': (
                    Lasso(random_state=42, max_iter=1000),
                    {} if quick_mode else {'alpha': [0.1, 1, 10, 100]}
                )
            })
        
        return models
    
    def _get_advanced_models(self, quick_mode: bool) -> Dict:
        """Get advanced gradient boosting models."""
        models = {}
        
        if LIGHTGBM_AVAILABLE:
            if self.problem_type == 'classification':
                models['LightGBM'] = (
                    lgb.LGBMClassifier(random_state=42, verbose=-1),
                    {} if quick_mode else {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'num_leaves': [31, 50, 100],
                        'feature_fraction': [0.8, 0.9, 1.0]
                    }
                )
            else:
                models['LightGBM'] = (
                    lgb.LGBMRegressor(random_state=42, verbose=-1),
                    {} if quick_mode else {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'num_leaves': [31, 50, 100],
                        'feature_fraction': [0.8, 0.9, 1.0]
                    }
                )
        
        if XGBOOST_AVAILABLE:
            if self.problem_type == 'classification':
                models['XGBoost'] = (
                    xgb.XGBClassifier(random_state=42, verbosity=0),
                    {} if quick_mode else {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                )
            else:
                models['XGBoost'] = (
                    xgb.XGBRegressor(random_state=42, verbosity=0),
                    {} if quick_mode else {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                )
        
        if CATBOOST_AVAILABLE:
            if self.problem_type == 'classification':
                models['CatBoost'] = (
                    cb.CatBoostClassifier(random_seed=42, verbose=False),
                    {} if quick_mode else {
                        'iterations': [100, 200, 300],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'depth': [4, 6, 8]
                    }
                )
            else:
                models['CatBoost'] = (
                    cb.CatBoostRegressor(random_seed=42, verbose=False),
                    {} if quick_mode else {
                        'iterations': [100, 200, 300],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'depth': [4, 6, 8]
                    }
                )
        
        return models
    
    def _get_automl_models(self, quick_mode: bool) -> Dict:
        """Get AutoML models."""
        models = {}
        
        if FLAML_AVAILABLE:
            models['FLAML AutoML'] = (
                AutoML(),
                {
                    'time_budget': 60 if quick_mode else 300,  # 1 or 5 minutes
                    'metric': 'accuracy' if self.problem_type == 'classification' else 'r2'
                }
            )
        
        if AUTOSKLEARN_AVAILABLE:
            if self.problem_type == 'classification':
                models['Auto-sklearn'] = (
                    autosklearn.classification.AutoSklearnClassifier(
                        time_left_for_this_task=60 if quick_mode else 300,
                        per_run_time_limit=30,
                        memory_limit=3072
                    ),
                    {'time_budget': 60 if quick_mode else 300}
                )
            else:
                models['Auto-sklearn'] = (
                    autosklearn.regression.AutoSklearnRegressor(
                        time_left_for_this_task=60 if quick_mode else 300,
                        per_run_time_limit=30,
                        memory_limit=3072
                    ),
                    {'time_budget': 60 if quick_mode else 300}
                )
        
        return models
    
    def _get_deep_learning_models(self) -> Dict:
        """Get deep learning models."""
        models = {}
        
        if PYTORCH_AVAILABLE:
            models['PyTorch MLP'] = (
                {'framework': 'pytorch', 'type': 'mlp'},
                {
                    'hidden_sizes': [128, 64, 32],
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            )
        
        if TENSORFLOW_AVAILABLE:
            models['TensorFlow NN'] = (
                {'framework': 'tensorflow', 'type': 'mlp'},
                {
                    'hidden_sizes': [128, 64, 32],
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            )
        
        return models
    
    def _create_ensemble_models(self):
        """Create ensemble models from trained models."""
        try:
            self.log("🔗 Creating ensemble models...")
            
            # Filter models that support predict_proba for voting
            suitable_models = {}
            for name, model in self.models.items():
                if hasattr(model, 'predict') and not any(keyword in name.lower() 
                                                        for keyword in ['automl', 'deep', 'neural']):
                    suitable_models[name] = model
            
            if len(suitable_models) < 2:
                self.log("  ⚠️ Not enough suitable models for ensemble")
                return
            
            # Create voting ensemble
            try:
                voting_ensemble = self.ensemble_methods.create_voting_ensemble(
                    suitable_models, self.problem_type, voting='soft' if self.problem_type == 'classification' else None
                )
                voting_ensemble.fit(self.X_train, self.y_train)
                
                self.models['Voting Ensemble'] = voting_ensemble
                scores = self._evaluate_model_comprehensive('Voting Ensemble', voting_ensemble, False)
                self.results['Voting Ensemble'] = scores
                
                self.log("  ✅ Voting ensemble created")
            except Exception as e:
                self.log(f"  ⚠️ Voting ensemble failed: {e}")
            
            # Create stacking ensemble
            try:
                stacking_ensemble = self.ensemble_methods.create_stacking_ensemble(
                    suitable_models, self.problem_type, cv=3
                )
                stacking_ensemble.fit(self.X_train, self.y_train)
                
                self.models['Stacking Ensemble'] = stacking_ensemble
                scores = self._evaluate_model_comprehensive('Stacking Ensemble', stacking_ensemble, False)
                self.results['Stacking Ensemble'] = scores
                
                self.log("  ✅ Stacking ensemble created")
            except Exception as e:
                self.log(f"  ⚠️ Stacking ensemble failed: {e}")
                
        except Exception as e:
            self.log(f"⚠️ Ensemble creation failed: {e}")
    
    def _evaluate_model_comprehensive(self, name: str, model: Any, cross_validation: bool) -> Dict:
        """Comprehensive model evaluation with multiple metrics."""
        try:
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Cross-validation if requested
            cv_scores = None
            if cross_validation:
                try:
                    cv_folds = min(5, len(self.X_train) // 10)
                    if cv_folds >= 2:
                        cv_scores = cross_val_score(
                            model, self.X_train, self.y_train,
                            cv=cv_folds, scoring=self._get_scoring_metric(),
                            n_jobs=-1
                        )
                except Exception as e:
                    self.log(f"    ⚠️ Cross-validation failed for {name}: {e}")
            
            # Validation set evaluation if available
            val_scores = {}
            if self.X_val is not None and self.y_val is not None:
                try:
                    y_pred_val = model.predict(self.X_val)
                    if self.problem_type == 'classification':
                        val_scores['val_accuracy'] = accuracy_score(self.y_val, y_pred_val)
                        val_scores['val_f1'] = f1_score(self.y_val, y_pred_val, average='weighted', zero_division=0)
                    else:
                        val_scores['val_mse'] = mean_squared_error(self.y_val, y_pred_val)
                        val_scores['val_r2'] = r2_score(self.y_val, y_pred_val)
                except Exception as e:
                    self.log(f"    ⚠️ Validation evaluation failed: {e}")
            
            if self.problem_type == 'classification':
                scores = self._evaluate_classification_comprehensive(
                    y_pred_train, y_pred_test, cv_scores, model
                )
            else:
                scores = self._evaluate_regression_comprehensive(
                    y_pred_train, y_pred_test, cv_scores
                )
            
            # Add validation scores
            scores.update(val_scores)
            
            return scores
            
        except Exception as e:
            self.log(f"    ❌ Evaluation failed for {name}: {e}")
            return self._get_default_scores()
    
    def _evaluate_classification_comprehensive(self, y_pred_train, y_pred_test, cv_scores, model) -> Dict:
        """Comprehensive classification evaluation."""
        scores = {
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, y_pred_test),
        }
        
        # Cross-validation scores
        if cv_scores is not None:
            scores.update({
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            })
        
        # Additional classification metrics
        try:
            n_classes = len(np.unique(self.y_test))
            average_method = 'binary' if n_classes == 2 else 'weighted'
            
            scores.update({
                'precision': precision_score(self.y_test, y_pred_test, average=average_method, zero_division=0),
                'recall': recall_score(self.y_test, y_pred_test, average=average_method, zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred_test, average=average_method, zero_division=0)
            })
            
            # ROC AUC for binary classification with probability predictions
            if n_classes == 2 and hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                    scores['roc_auc'] = roc_auc_score(self.y_test, y_proba)
                except:
                    scores['roc_auc'] = 0.5
            
            scores['confusion_matrix'] = confusion_matrix(self.y_test, y_pred_test)
            
        except Exception as e:
            self.log(f"      ⚠️ Additional classification metrics failed: {e}")
            scores.update({'precision': 0, 'recall': 0, 'f1_score': 0, 'confusion_matrix': np.array([[0]])})
        
        return scores
    
    def _evaluate_regression_comprehensive(self, y_pred_train, y_pred_test, cv_scores) -> Dict:
        """Comprehensive regression evaluation."""
        scores = {
            'train_mse': mean_squared_error(self.y_train, y_pred_train),
            'test_mse': mean_squared_error(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test)
        }
        
        # Cross-validation scores
        if cv_scores is not None:
            scores.update({
                'cv_mse': -cv_scores.mean(),  # CV uses negative MSE
                'cv_std': cv_scores.std()
            })
        
        # Additional regression metrics
        try:
            # Mean Absolute Percentage Error
            def mape(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
            
            scores.update({
                'train_mape': mape(self.y_train, y_pred_train),
                'test_mape': mape(self.y_test, y_pred_test)
            })
            
            # Explained Variance Score
            from sklearn.metrics import explained_variance_score
            scores.update({
                'train_explained_variance': explained_variance_score(self.y_train, y_pred_train),
                'test_explained_variance': explained_variance_score(self.y_test, y_pred_test)
            })
            
        except Exception as e:
            self.log(f"      ⚠️ Additional regression metrics failed: {e}")
        
        return scores
    
    def _get_default_scores(self) -> Dict:
        """Get default scores for failed models."""
        if self.problem_type == 'classification':
            return {
                'train_accuracy': 0, 'test_accuracy': 0, 'cv_accuracy': 0, 'cv_std': 0,
                'precision': 0, 'recall': 0, 'f1_score': 0, 'confusion_matrix': np.array([[0]]),
                'roc_auc': 0.5
            }
        else:
            return {
                'train_mse': float('inf'), 'test_mse': float('inf'), 'train_rmse': float('inf'),
                'test_rmse': float('inf'), 'train_mae': float('inf'), 'test_mae': float('inf'),
                'train_r2': -float('inf'), 'test_r2': -float('inf'), 'cv_mse': float('inf'), 
                'cv_std': 0, 'train_mape': float('inf'), 'test_mape': float('inf')
            }
    
    # Model persistence and management
    def save_models(self, save_path: str = None) -> Dict[str, str]:
        """Save trained models with metadata."""
        if not self.models:
            return {}
        
        save_path = Path(save_path) if save_path else self.model_save_path
        save_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_models = {}
        
        for name, model in self.models.items():
            try:
                # Create model-specific directory
                model_dir = save_path / f"{name.replace(' ', '_')}_{timestamp}"
                model_dir.mkdir(exist_ok=True)
                
                # Save model
                model_file = model_dir / "model.pkl"
                joblib.dump(model, model_file)
                
                # Save metadata
                metadata = {
                    'model_name': name,
                    'model_type': type(model).__name__,
                    'problem_type': self.problem_type,
                    'feature_names': self.feature_names,
                    'training_metadata': self.model_metadata.get(name, {}),
                    'results': self.results.get(name, {}),
                    'training_history': self.training_history.get(name, {}),
                    'saved_timestamp': datetime.now().isoformat(),
                    'target_encoder': self.target_encoder,
                    'feature_scaler': self.feature_scaler
                }
                
                metadata_file = model_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                saved_models[name] = str(model_dir)
                self.log(f"💾 Saved {name} to {model_dir}")
                
            except Exception as e:
                self.log(f"⚠️ Failed to save {name}: {e}")
        
        return saved_models
    
    def load_model(self, model_path: str) -> bool:
        """Load a previously saved model."""
        try:
            model_path = Path(model_path)
            
            # Load model
            model_file = model_path / "model.pkl"
            model = joblib.load(model_file)
            
            # Load metadata
            metadata_file = model_path / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Restore state
            model_name = metadata['model_name']
            self.models[model_name] = model
            self.results[model_name] = metadata.get('results', {})
            self.model_metadata[model_name] = metadata.get('training_metadata', {})
            self.training_history[model_name] = metadata.get('training_history', {})
            
            # Restore configuration
            self.problem_type = metadata['problem_type']
            self.feature_names = metadata['feature_names']
            self.target_encoder = metadata.get('target_encoder')
            self.feature_scaler = metadata.get('feature_scaler')
            
            # Set as best model if it's the first/only loaded model
            if not self.best_model:
                self.best_model = model
                self.best_model_name = model_name
            
            self.log(f"📂 Loaded {model_name} from {model_path}")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to load model: {e}")
            return False
    
    # Enhanced utility methods
    def get_training_progress(self) -> Dict:
        """Get current training progress."""
        return self.progress_tracker.get_progress()
    
    def get_model_comparison(self, sort_by: str = None) -> Dict:
        """Enhanced model comparison with sorting options."""
        if not self.results:
            return {
                "error": "No models trained yet",
                "headers": ["Model", "Status"],
                "rows": [["No models", "Not trained"]]
            }
        
        try:
            comparison = {'headers': [], 'rows': []}
            
            if self.problem_type == 'classification':
                headers = ['Model', 'Test Accuracy', 'CV Accuracy', 'Precision', 'Recall', 'F1 Score']
                if any('roc_auc' in scores for scores in self.results.values()):
                    headers.append('ROC AUC')
                
                comparison['headers'] = headers
                
                rows = []
                for name, scores in self.results.items():
                    row = [
                        name,
                        f"{scores.get('test_accuracy', 0):.4f}",
                        f"{scores.get('cv_accuracy', 0):.4f} ± {scores.get('cv_std', 0):.4f}",
                        f"{scores.get('precision', 0):.4f}",
                        f"{scores.get('recall', 0):.4f}",
                        f"{scores.get('f1_score', 0):.4f}"
                    ]
                    
                    if 'ROC AUC' in headers:
                        row.append(f"{scores.get('roc_auc', 0.5):.4f}")
                    
                    rows.append(row)
                
                # Sort by test accuracy by default
                if rows:
                    sort_idx = 1 if sort_by is None else (headers.index(sort_by) if sort_by in headers else 1)
                    try:
                        rows.sort(key=lambda x: float(x[sort_idx].split()[0]), reverse=True)
                    except (ValueError, IndexError):
                        pass  # Keep original order if sorting fails
                
            else:  # Regression
                headers = ['Model', 'Test R²', 'Test RMSE', 'Test MAE', 'CV MSE']
                if any('test_mape' in scores for scores in self.results.values()):
                    headers.append('Test MAPE')
                
                comparison['headers'] = headers
                
                rows = []
                for name, scores in self.results.items():
                    row = [
                        name,
                        f"{scores.get('test_r2', -float('inf')):.4f}",
                        f"{scores.get('test_rmse', float('inf')):.4f}",
                        f"{scores.get('test_mae', float('inf')):.4f}",
                        f"{scores.get('cv_mse', float('inf')):.4f} ± {scores.get('cv_std', 0):.4f}"
                    ]
                    
                    if 'Test MAPE' in headers:
                        row.append(f"{scores.get('test_mape', float('inf')):.2f}%")
                    
                    rows.append(row)
                
                # Sort by R² by default (higher is better)
                if rows:
                    sort_idx = 1 if sort_by is None else (headers.index(sort_by) if sort_by in headers else 1)
                    try:
                        reverse_sort = sort_by in ['Test R²'] if sort_by else True
                        rows.sort(key=lambda x: float(x[sort_idx].split()[0]), reverse=reverse_sort)
                    except (ValueError, IndexError):
                        pass  # Keep original order if sorting fails
            
            comparison['rows'] = rows
            return comparison
            
        except Exception as e:
            return {
                "error": f"Error generating comparison: {e}",
                "headers": ["Model", "Status"],
                "rows": [["Error", str(e)]]
            }
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> Optional[pd.DataFrame]:
        """Enhanced feature importance extraction."""
        model = self.best_model if model_name is None else self.models.get(model_name)
        if not model:
            return None
        
        try:
            importance = None
            method = "Unknown"
            
            # Try different methods to extract importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                method = "Feature Importances"
            elif hasattr(model, 'coef_'):
                if model.coef_.ndim > 1:
                    importance = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importance = np.abs(model.coef_)
                method = "Coefficients (Absolute)"
            elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
                # For ensemble methods
                importance = model.estimators_[0].feature_importances_
                method = "Ensemble Feature Importances"
            
            if importance is not None:
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': self.feature_names[:len(importance)],
                    'importance': importance,
                    'importance_normalized': importance / importance.sum() * 100
                }).sort_values('importance', ascending=False).head(top_n)
                
                importance_df['method'] = method
                importance_df['model'] = model_name or self.best_model_name
                
                return importance_df
            
            # Try SHAP if available and model supports it
            if SHAP_AVAILABLE:
                return self._get_shap_importance(model, model_name, top_n)
            
            return None
            
        except Exception as e:
            self.log(f"Could not extract feature importance: {str(e)}")
            return None
    
    def _get_shap_importance(self, model, model_name: str, top_n: int) -> Optional[pd.DataFrame]:
        """Get feature importance using SHAP."""
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, self.X_train.sample(min(100, len(self.X_train))))
            
            # Calculate SHAP values for a sample
            sample_size = min(100, len(self.X_test))
            shap_values = explainer(self.X_test.sample(sample_size))
            
            # Get mean absolute SHAP values
            if isinstance(shap_values.values, list):
                # Multi-class classification
                importance = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values.values], axis=0)
            else:
                importance = np.mean(np.abs(shap_values.values), axis=0)
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(importance)],
                'importance': importance,
                'importance_normalized': importance / importance.sum() * 100
            }).sort_values('importance', ascending=False).head(top_n)
            
            importance_df['method'] = "SHAP Values"
            importance_df['model'] = model_name or self.best_model_name
            
            return importance_df
            
        except Exception as e:
            self.log(f"SHAP importance extraction failed: {e}")
            return None
    
    def get_model_summary(self) -> Dict:
        """Enhanced model summary with detailed information."""
        if not self.best_model:
            return {"status": "no_models_trained"}
        
        try:
            summary = {
                'training_overview': {
                    'best_model': self.best_model_name,
                    'problem_type': self.problem_type,
                    'models_trained': len(self.models),
                    'models_successful': len([r for r in self.results.values() if r.get('test_accuracy', r.get('test_r2', -1)) > 0])
                },
                'data_info': {
                    'training_samples': len(self.X_train) if self.X_train is not None else 0,
                    'validation_samples': len(self.X_val) if self.X_val is not None else 0,
                    'test_samples': len(self.X_test) if self.X_test is not None else 0,
                    'features': len(self.feature_names),
                    'feature_scaling': self.feature_scaler is not None,
                    'target_encoding': self.target_encoder is not None
                },
                'best_model_performance': self.results.get(self.best_model_name, {}),
                'training_configuration': {
                    'multiprocessing': self.use_multiprocessing,
                    'max_workers': self.max_workers,
                    'early_stopping': self.enable_early_stopping,
                    'auto_feature_selection': self.auto_feature_selection
                },
                'model_metadata': self.model_metadata.get(self.best_model_name, {}),
                'available_models': list(self.models.keys())
            }
            
            return summary
        except Exception as e:
            return {"error": f"Error generating model summary: {e}"}
    
    def get_results_summary(self) -> Dict:
        """Get comprehensive results summary for export."""
        return {
            'models': {name: scores for name, scores in self.results.items()},
            'best_model': self.best_model_name,
            'problem_type': self.problem_type,
            'feature_names': self.feature_names,
            'training_metadata': self.model_metadata,
            'training_history': self.training_history,
            'cross_validation_results': self.cross_validation_results
        }
    
    # Original methods for backward compatibility
    def _determine_stratification(self, y):
        """Determine appropriate stratification strategy."""
        if self.problem_type == 'classification':
            try:
                class_counts = pd.Series(y).value_counts()
                min_class_size = class_counts.min()
                
                if min_class_size < 2:
                    return None
                elif min_class_size < 5:
                    return y
                else:
                    return y
            except Exception as e:
                self.log(f"⚠️ Could not determine stratification: {e}")
                return None
        else:
            return None
    
    def _smart_handle_missing_values(self, data, target_column):
        """Intelligently handle missing values based on data characteristics."""
        data = data.copy()
        missing_info = {}
        
        for col in data.columns:
            if col == target_column:
                continue
                
            missing_pct = data[col].isnull().sum() / len(data) * 100
            missing_info[col] = missing_pct
            
            if missing_pct == 0:
                continue
            elif missing_pct > 90:
                # Drop columns with >90% missing values
                self.log(f"[CLEAN] Dropping column '{col}' - {missing_pct:.1f}% missing")
                data = data.drop(columns=[col])
            elif missing_pct > 50:
                # For high missing percentage, use more sophisticated imputation
                if data[col].dtype in ['int64', 'float64']:
                    # Use iterative imputation for numerical
                    imputer = IterativeImputer(random_state=42, max_iter=10)
                    data[col] = imputer.fit_transform(data[[col]]).flatten()
                    self.log(f"[CLEAN] Iterative imputation for '{col}' - {missing_pct:.1f}% missing")
                else:
                    # Keep categorical missing as a separate category if informative
                    if self._is_missing_informative(data, col, target_column):
                        data[col].fillna('MISSING_VALUE', inplace=True)
                        self.log(f"[CLEAN] Preserving missing as category for '{col}' - informative pattern")
                    else:
                        mode_val = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'Unknown'
                        data[col].fillna(mode_val, inplace=True)
            else:
                # Standard imputation for low missing percentage
                if data[col].dtype in ['int64', 'float64']:
                    # Use median for numerical
                    data[col].fillna(data[col].median(), inplace=True)
                else:
                    # Use mode for categorical
                    mode_val = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'Unknown'
                    data[col].fillna(mode_val, inplace=True)
        
        return data
    
    def _is_missing_informative(self, data, column, target_column):
        """Check if missing pattern in column is informative for prediction."""
        try:
            # Create binary indicator for missing values
            missing_indicator = data[column].isnull().astype(int)
            
            # Check correlation with target if it's numerical
            if data[target_column].dtype in ['int64', 'float64']:
                correlation = missing_indicator.corr(data[target_column])
                return abs(correlation) > 0.1  # Threshold for meaningful correlation
            else:
                # For categorical targets, check if missing pattern varies by class
                target_missing_rates = data.groupby(target_column)[column].apply(lambda x: x.isnull().mean())
                return target_missing_rates.std() > 0.1  # Variance in missing rates across classes
        except:
            return False
    
    def _ensure_numeric_features(self, X):
        """Ensure all features are numeric."""
        X = X.copy()
        columns_to_drop = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if X[col].isnull().sum() > 0:
                        X[col].fillna(X[col].median(), inplace=True)
                except:
                    try:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                    except:
                        columns_to_drop.append(col)
        
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
            self.log(f"🗑️ Dropped {len(columns_to_drop)} non-convertible columns")
        
        return X
    
    def _enhanced_problem_type_detection(self, target_series, features_df):
        """Enhanced problem type detection with advanced heuristics."""
        unique_vals = target_series.nunique()
        total_vals = len(target_series.dropna())
        
        if total_vals == 0:
            return 'unsupervised'
        
        unique_ratio = unique_vals / total_vals
        is_object = target_series.dtype == 'object'
        
        # Check for time series indicators
        datetime_cols = features_df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0 or any('date' in col.lower() or 'time' in col.lower() 
                                       for col in features_df.columns):
            self.log("[DETECT] Time series indicators found")
            return 'time_series'
        
        # Enhanced classification detection
        if is_object:
            return 'classification'
        
        # Check for binary classification (including 0/1, True/False)
        if unique_vals == 2:
            return 'classification'
        
        # Check if values are mostly integers and few unique values
        if target_series.dtype in ['int64', 'int32']:
            if unique_vals <= 20:
                return 'classification'
            elif unique_vals <= 50 and unique_ratio < 0.1:
                return 'classification'
        
        # Check for ordinal patterns (like ratings 1-5)
        if unique_vals <= 10 and all(isinstance(x, (int, float)) for x in target_series.dropna().unique()):
            sorted_vals = sorted(target_series.dropna().unique())
            if all(sorted_vals[i+1] - sorted_vals[i] == 1 for i in range(len(sorted_vals)-1)):
                self.log("[DETECT] Ordinal classification pattern detected")
                return 'classification'
        
        # Default to regression for continuous variables
        if unique_ratio > 0.05:
            return 'regression'
        else:
            return 'classification'
    
    def _encode_target_for_classification(self, y):
        """Encode target variable for classification."""
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y.astype(str))
            return pd.Series(y_encoded, index=y.index)
        else:
            if not np.all(y == y.astype(int)):
                y = y.astype(int)
            return y
    
    def _validate_prepared_data(self):
        """Validate prepared data."""
        issues = []
        
        if self.X_train.isnull().sum().sum() > 0:
            issues.append("Features contain NaN values")
        
        if pd.Series(self.y_train).isnull().sum() > 0:
            issues.append("Target contains NaN values")
        
        if len(self.X_train) == 0:
            issues.append("Training set is empty")
        
        if len(self.X_train.columns) == 0:
            issues.append("No features available")
        
        if issues:
            raise ValueError("Data validation failed: " + "; ".join(issues))
    
    def _get_scoring_metric(self):
        """Get appropriate scoring metric."""
        return 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
    
    def _find_best_model(self):
        """Find the best performing model."""
        if not self.results:
            return
        
        try:
            if self.problem_type == 'classification':
                metric = 'test_accuracy'
                best_score = max(self.results.values(), key=lambda x: x.get(metric, 0))[metric]
            else:
                metric = 'test_r2'
                best_score = max(self.results.values(), key=lambda x: x.get(metric, -float('inf')))[metric]
            
            for name, scores in self.results.items():
                if scores.get(metric, -float('inf')) == best_score:
                    self.best_model_name = name
                    self.best_model = self.models[name]
                    break
                    
        except Exception as e:
            if self.models:
                self.best_model_name = list(self.models.keys())[0]
                self.best_model = list(self.models.values())[0]
    
    def log(self, message: str):
        """Add message to training log."""
        self.training_log.append(message)
        print(message)
    
    def get_training_log(self) -> List[str]:
        """Get full training log."""
        return self.training_log
    
    def reset(self):
        """Reset the ML engine to initial state."""
        self.__init__()
        self.log("🔄 SmartMLEngine reset to initial state")