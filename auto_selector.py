import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class IntelligentProcessSelector:
    """AI-powered pipeline selection system that analyzes dataset characteristics 
    and recommends optimal preprocessing and modeling strategies."""
    
    def __init__(self):
        self.decision_rules = self._initialize_decision_rules()
        self.benchmark_results = {}
        self.pipeline_recommendations = {}
        self.confidence_threshold = 0.7
        
        # Gaming elements
        self.achievement_system = {
            'data_explorer': {'threshold': 1, 'unlocked': False, 'description': 'Analyzed first dataset'},
            'pattern_master': {'threshold': 5, 'unlocked': False, 'description': 'Identified 5 different data patterns'},
            'pipeline_guru': {'threshold': 10, 'unlocked': False, 'description': 'Recommended 10 optimal pipelines'},
            'efficiency_expert': {'threshold': 3, 'unlocked': False, 'description': 'Achieved 90%+ confidence 3 times'}
        }
        self.analysis_count = 0
        self.high_confidence_count = 0
        
    def _initialize_decision_rules(self):
        """Initialize comprehensive decision rules for pipeline selection."""
        return {
            'data_size_rules': {
                'small': {'threshold': 1000, 'recommended_models': ['linear', 'tree', 'svm']},
                'medium': {'threshold': 100000, 'recommended_models': ['ensemble', 'tree', 'neural']},
                'large': {'threshold': float('inf'), 'recommended_models': ['lightgbm', 'xgboost', 'neural']}
            },
            'feature_type_rules': {
                'text_heavy': {'threshold': 30, 'pipeline': 'nlp', 'models': ['bert', 'tfidf_svm', 'lstm']},
                'categorical_heavy': {'threshold': 70, 'pipeline': 'categorical', 'models': ['catboost', 'target_encoding']},
                'numeric_heavy': {'threshold': 80, 'pipeline': 'numeric', 'models': ['linear', 'tree', 'neural']},
                'mixed': {'threshold': 0, 'pipeline': 'mixed', 'models': ['ensemble', 'preprocessing_heavy']}
            },
            'quality_rules': {
                'high_missing': {'threshold': 30, 'preprocessing': ['iterative_imputer', 'knn_imputer', 'indicator_features']},
                'high_cardinality': {'threshold': 100, 'preprocessing': ['target_encoding', 'frequency_encoding', 'embedding']},
                'outliers': {'threshold': 10, 'preprocessing': ['robust_scaling', 'outlier_removal', 'winsorization']},
                'imbalanced': {'threshold': 80, 'preprocessing': ['smote', 'class_weights', 'undersampling']}
            },
            'time_series_rules': {
                'datetime_present': {'preprocessing': ['time_features', 'lag_features', 'seasonal_decomposition']},
                'temporal_patterns': {'models': ['arima', 'lstm', 'prophet']}
            }
        }
    
    def select_pipeline(self, meta_features, data_sample=None):
        """Main method to select optimal pipeline based on dataset characteristics."""
        try:
            self.analysis_count += 1
            
            # Update achievement progress
            self._update_achievements()
            
            # Step 1: Quick rule-based analysis
            rule_based_recommendation = self._rule_based_selection(meta_features)
            
            # Step 2: Benchmark-based validation (if sklearn available)
            benchmark_recommendation = None
            if SKLEARN_AVAILABLE and data_sample is not None:
                benchmark_recommendation = self._benchmark_based_selection(data_sample, meta_features)
            
            # Step 3: Combine recommendations
            final_recommendation = self._combine_recommendations(
                rule_based_recommendation, 
                benchmark_recommendation, 
                meta_features
            )
            
            # Step 4: Calculate confidence and add gaming elements
            confidence = self._calculate_confidence(final_recommendation, meta_features)
            final_recommendation['confidence'] = confidence
            final_recommendation['confidence_level'] = self._get_confidence_level(confidence)
            
            # Update high confidence achievement
            if confidence > 0.9:
                self.high_confidence_count += 1
                self._update_achievements()
            
            # Add gaming elements
            final_recommendation['achievements'] = self._get_recent_achievements()
            final_recommendation['analysis_stats'] = {
                'total_analyses': self.analysis_count,
                'high_confidence_analyses': self.high_confidence_count,
                'success_rate': (self.high_confidence_count / self.analysis_count) * 100
            }
            
            # Store recommendation for learning
            self.pipeline_recommendations[datetime.now().isoformat()] = final_recommendation
            
            return final_recommendation
            
        except Exception as e:
            return {
                'pipeline': 'standard_ml',
                'confidence': 0.5,
                'error': str(e),
                'reasoning': ['Error in pipeline selection - defaulting to standard ML pipeline'],
                'fallback': True
            }
    
    def _rule_based_selection(self, meta_features):
        """Rule-based pipeline selection using dataset characteristics."""
        recommendations = {
            'pipeline': 'standard_ml',
            'preprocessing_steps': [],
            'model_family': [],
            'reasoning': [],
            'priority_score': 0
        }
        
        # Data size analysis
        data_size = meta_features.get('dataset_size', 0)
        n_rows = meta_features.get('n_rows', 0)
        
        if n_rows < self.decision_rules['data_size_rules']['small']['threshold']:
            size_category = 'small'
            recommendations['reasoning'].append(f"🔍 Small dataset ({n_rows:,} rows) - Using lightweight models")
        elif n_rows < self.decision_rules['data_size_rules']['medium']['threshold']:
            size_category = 'medium'
            recommendations['reasoning'].append(f"📊 Medium dataset ({n_rows:,} rows) - Using ensemble methods")
        else:
            size_category = 'large'
            recommendations['reasoning'].append(f"🚀 Large dataset ({n_rows:,} rows) - Using scalable algorithms")
            recommendations['pipeline'] = 'big_data_ml'
        
        recommendations['model_family'] = self.decision_rules['data_size_rules'][size_category]['recommended_models']
        
        # Feature type analysis
        text_percentage = meta_features.get('pct_text', 0)
        categorical_percentage = meta_features.get('pct_categorical', 0)
        numeric_percentage = meta_features.get('pct_numeric', 0)
        
        if text_percentage > self.decision_rules['feature_type_rules']['text_heavy']['threshold']:
            recommendations['pipeline'] = 'nlp'
            recommendations['model_family'] = self.decision_rules['feature_type_rules']['text_heavy']['models']
            recommendations['preprocessing_steps'].extend(['text_cleaning', 'tokenization', 'vectorization'])
            recommendations['reasoning'].append(f"📝 Text-heavy dataset ({text_percentage:.1f}% text) - NLP pipeline required")
            recommendations['priority_score'] += 30
            
        elif categorical_percentage > self.decision_rules['feature_type_rules']['categorical_heavy']['threshold']:
            recommendations['pipeline'] = 'categorical_ml'
            recommendations['preprocessing_steps'].extend(['encoding', 'feature_selection'])
            recommendations['reasoning'].append(f"🏷️ High categorical features ({categorical_percentage:.1f}%) - Advanced encoding needed")
            recommendations['priority_score'] += 20
            
        elif numeric_percentage > self.decision_rules['feature_type_rules']['numeric_heavy']['threshold']:
            recommendations['preprocessing_steps'].extend(['scaling', 'normalization'])
            recommendations['reasoning'].append(f"🔢 Numeric-heavy dataset ({numeric_percentage:.1f}%) - Standard preprocessing")
            recommendations['priority_score'] += 10
        
        # Data quality analysis
        sparsity = meta_features.get('sparsity', 0)
        completeness = meta_features.get('completeness', 100)
        
        if sparsity > self.decision_rules['quality_rules']['high_missing']['threshold']:
            recommendations['preprocessing_steps'].extend(self.decision_rules['quality_rules']['high_missing']['preprocessing'])
            recommendations['reasoning'].append(f"🕳️ High sparsity ({sparsity:.1f}%) - Advanced imputation required")
            recommendations['priority_score'] += 25
        
        # Outlier analysis
        outlier_percentage = meta_features.get('outlier_percentage', 0)
        if outlier_percentage > self.decision_rules['quality_rules']['outliers']['threshold']:
            recommendations['preprocessing_steps'].extend(self.decision_rules['quality_rules']['outliers']['preprocessing'])
            recommendations['reasoning'].append(f"⚠️ High outliers ({outlier_percentage:.1f}%) - Robust preprocessing needed")
            recommendations['priority_score'] += 15
        
        # Class imbalance analysis
        class_imbalance = meta_features.get('class_imbalance', 0)
        if class_imbalance and class_imbalance > self.decision_rules['quality_rules']['imbalanced']['threshold']:
            recommendations['preprocessing_steps'].extend(self.decision_rules['quality_rules']['imbalanced']['preprocessing'])
            recommendations['reasoning'].append(f"⚖️ Class imbalance detected ({class_imbalance:.1f}%) - Balancing techniques needed")
            recommendations['priority_score'] += 20
        
        # Correlation analysis
        max_correlation = meta_features.get('max_correlation', 0)
        high_corr_pairs = meta_features.get('high_corr_pairs', 0)
        
        if high_corr_pairs > 5:
            recommendations['preprocessing_steps'].extend(['feature_selection', 'pca', 'correlation_removal'])
            recommendations['reasoning'].append(f"🔗 High correlations ({high_corr_pairs} pairs) - Dimensionality reduction needed")
            recommendations['priority_score'] += 10
        
        return recommendations
    
    def _benchmark_based_selection(self, data_sample, meta_features):
        """Quick benchmark to validate rule-based recommendations."""
        if not SKLEARN_AVAILABLE:
            return None
        
        try:
            # Prepare sample data for quick benchmark
            benchmark_data = data_sample.copy()
            
            # Find potential target column
            target_candidates = []
            for col in benchmark_data.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['target', 'label', 'class', 'y']):
                    target_candidates.append(col)
            
            # Also check for binary columns
            for col in benchmark_data.columns:
                if benchmark_data[col].nunique() == 2 and col not in target_candidates:
                    target_candidates.append(col)
            
            if not target_candidates:
                return {'error': 'No suitable target column found for benchmarking'}
            
            target_col = target_candidates[0]
            
            # Prepare features and target
            features = benchmark_data.drop(target_col, axis=1)
            target = benchmark_data[target_col]
            
            # Quick preprocessing
            features_processed = self._quick_preprocessing(features)
            
            if features_processed is None or len(features_processed.columns) == 0:
                return {'error': 'Feature preprocessing failed'}
            
            # Encode target if needed
            if target.dtype == 'object':
                le = LabelEncoder()
                target_encoded = le.fit_transform(target.fillna('missing'))
            else:
                target_encoded = target.fillna(target.median())
            
            # Quick benchmark with multiple models
            models = {
                'tree': self._quick_tree_model,
                'linear': self._quick_linear_model,
                'ensemble': self._quick_ensemble_model
            }
            
            benchmark_results = {}
            for model_name, model_func in models.items():
                try:
                    score = model_func(features_processed, target_encoded)
                    benchmark_results[model_name] = score
                except Exception as e:
                    benchmark_results[model_name] = 0.0
            
            # Select best performing model family
            if benchmark_results:
                best_model = max(benchmark_results, key=benchmark_results.get)
                best_score = benchmark_results[best_model]
                
                return {
                    'best_model_family': best_model,
                    'best_score': best_score,
                    'all_scores': benchmark_results,
                    'reasoning': f"Quick benchmark favors {best_model} (score: {best_score:.3f})"
                }
        
        except Exception as e:
            return {'error': f'Benchmark failed: {str(e)}'}
        
        return None
    
    def _quick_preprocessing(self, features):
        """Quick preprocessing for benchmark."""
        try:
            processed_features = features.copy()
            
            # Handle missing values
            numeric_cols = processed_features.select_dtypes(include=[np.number]).columns
            categorical_cols = processed_features.select_dtypes(include=['object']).columns
            
            # Fill numeric columns
            if len(numeric_cols) > 0:
                imputer_num = SimpleImputer(strategy='median')
                processed_features[numeric_cols] = imputer_num.fit_transform(processed_features[numeric_cols])
            
            # Handle categorical columns
            for col in categorical_cols:
                if processed_features[col].nunique() < 10:
                    # One-hot encode low cardinality
                    dummies = pd.get_dummies(processed_features[col], prefix=col)
                    processed_features = pd.concat([processed_features, dummies], axis=1)
                
                # Drop original categorical column
                processed_features = processed_features.drop(col, axis=1)
            
            # Ensure all features are numeric
            for col in processed_features.columns:
                if processed_features[col].dtype == 'object':
                    processed_features = processed_features.drop(col, axis=1)
            
            return processed_features
            
        except Exception as e:
            return None
    
    def _quick_tree_model(self, X, y):
        """Quick tree model benchmark."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        
        model = DecisionTreeClassifier(random_state=42, max_depth=5)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return np.mean(scores)
    
    def _quick_linear_model(self, X, y):
        """Quick linear model benchmark."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        model = LogisticRegression(random_state=42, max_iter=100)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return np.mean(scores)
    
    def _quick_ensemble_model(self, X, y):
        """Quick ensemble model benchmark."""
        model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return np.mean(scores)
    
    def _combine_recommendations(self, rule_based, benchmark_based, meta_features):
        """Combine rule-based and benchmark-based recommendations."""
        combined = rule_based.copy()
        
        # Add benchmark insights if available
        if benchmark_based and 'error' not in benchmark_based:
            combined['benchmark_results'] = benchmark_based
            combined['reasoning'].append(f"🏃‍♂️ Benchmark validation: {benchmark_based.get('reasoning', 'Completed')}")
            
            # Adjust model family based on benchmark
            if 'best_model_family' in benchmark_based:
                best_family = benchmark_based['best_model_family']
                if best_family not in combined['model_family']:
                    combined['model_family'].insert(0, best_family)
                    combined['reasoning'].append(f"🎯 Benchmark recommends {best_family} as primary choice")
        
        # Add final pipeline recommendation
        combined['final_pipeline'] = self._determine_final_pipeline(combined, meta_features)
        
        # Add estimated performance
        combined['estimated_performance'] = self._estimate_performance(combined, meta_features)
        
        # Add resource requirements
        combined['resource_requirements'] = self._estimate_resources(combined, meta_features)
        
        return combined
    
    def _determine_final_pipeline(self, recommendations, meta_features):
        """Determine the final pipeline configuration."""
        pipeline_config = {
            'name': recommendations['pipeline'],
            'preprocessing': recommendations['preprocessing_steps'],
            'models': recommendations['model_family'][:3],  # Top 3 models
            'evaluation_strategy': 'cross_validation',
            'optimization': 'grid_search'
        }
        
        # Add specific configurations based on data characteristics
        n_rows = meta_features.get('n_rows', 0)
        
        if n_rows > 100000:
            pipeline_config['optimization'] = 'random_search'
            pipeline_config['evaluation_strategy'] = 'holdout'
            pipeline_config['early_stopping'] = True
        
        if meta_features.get('pct_text', 0) > 30:
            pipeline_config['text_preprocessing'] = ['tokenization', 'lemmatization', 'vectorization']
            pipeline_config['models'] = ['bert', 'tfidf_svm', 'naive_bayes']
        
        return pipeline_config
    
    def _estimate_performance(self, recommendations, meta_features):
        """Estimate expected performance based on data characteristics."""
        base_score = 0.7  # Base expectation
        
        # Adjust based on data quality
        completeness = meta_features.get('completeness', 100)
        base_score += (completeness - 80) / 100  # Bonus for good completeness
        
        # Adjust based on data size
        n_rows = meta_features.get('n_rows', 0)
        if n_rows > 10000:
            base_score += 0.05  # More data generally helps
        
        # Adjust based on feature quality
        outlier_pct = meta_features.get('outlier_percentage', 0)
        base_score -= outlier_pct / 200  # Penalty for outliers
        
        # Benchmark bonus
        if 'benchmark_results' in recommendations:
            benchmark_score = max(recommendations['benchmark_results'].get('all_scores', {}).values())
            base_score = (base_score + benchmark_score) / 2
        
        return {
            'estimated_accuracy': min(0.99, max(0.5, base_score)),
            'confidence_interval': [base_score - 0.1, base_score + 0.1],
            'factors': {
                'data_quality': completeness,
                'data_size_factor': min(n_rows / 10000, 2.0),
                'outlier_penalty': -outlier_pct / 200
            }
        }
    
    def _estimate_resources(self, recommendations, meta_features):
        """Estimate computational resource requirements."""
        n_rows = meta_features.get('n_rows', 0)
        n_cols = meta_features.get('n_cols', 0)
        
        # Base resource calculation
        memory_gb = max(0.5, (n_rows * n_cols * 8) / (1024**3))  # Rough estimate
        training_time_minutes = max(1, n_rows / 10000)  # Rough estimate
        
        # Adjust for pipeline complexity
        if recommendations['pipeline'] == 'nlp':
            memory_gb *= 3
            training_time_minutes *= 5
        elif recommendations['pipeline'] == 'big_data_ml':
            memory_gb *= 0.5  # More efficient algorithms
            training_time_minutes *= 2
        
        cpu_cores = min(8, max(1, n_rows // 50000))
        
        return {
            'estimated_memory_gb': round(memory_gb, 2),
            'estimated_training_time_minutes': round(training_time_minutes, 1),
            'recommended_cpu_cores': int(cpu_cores),
            'gpu_recommended': recommendations['pipeline'] in ['nlp', 'deep_learning'],
            'scalability': 'high' if n_rows > 100000 else 'medium' if n_rows > 10000 else 'low'
        }
    
    def _calculate_confidence(self, recommendation, meta_features):
        """Calculate confidence score for the recommendation."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on clear patterns
        priority_score = recommendation.get('priority_score', 0)
        confidence += min(0.3, priority_score / 100)
        
        # Benchmark validation bonus
        if 'benchmark_results' in recommendation:
            best_score = max(recommendation['benchmark_results'].get('all_scores', {}).values())
            confidence += best_score * 0.3
        
        # Data quality bonus
        completeness = meta_features.get('completeness', 100)
        if completeness > 95:
            confidence += 0.1
        elif completeness < 70:
            confidence -= 0.1
        
        # Clear use case bonus
        text_pct = meta_features.get('pct_text', 0)
        if text_pct > 50 and recommendation['pipeline'] == 'nlp':
            confidence += 0.2
        
        return min(0.99, max(0.3, confidence))
    
    def _get_confidence_level(self, confidence):
        """Convert confidence score to human-readable level."""
        if confidence >= 0.9:
            return "🏆 VERY HIGH"
        elif confidence >= 0.8:
            return "⭐ HIGH"
        elif confidence >= 0.7:
            return "🎯 GOOD"
        elif confidence >= 0.6:
            return "🟡 MODERATE"
        else:
            return "⚠️ LOW"
    
    def _update_achievements(self):
        """Update achievement system."""
        achievements = self.achievement_system
        
        # Data Explorer
        if self.analysis_count >= 1 and not achievements['data_explorer']['unlocked']:
            achievements['data_explorer']['unlocked'] = True
        
        # Pattern Master (simplified - unlock after 5 analyses)
        if self.analysis_count >= 5 and not achievements['pattern_master']['unlocked']:
            achievements['pattern_master']['unlocked'] = True
        
        # Pipeline Guru
        if self.analysis_count >= 10 and not achievements['pipeline_guru']['unlocked']:
            achievements['pipeline_guru']['unlocked'] = True
        
        # Efficiency Expert
        if self.high_confidence_count >= 3 and not achievements['efficiency_expert']['unlocked']:
            achievements['efficiency_expert']['unlocked'] = True
    
    def _get_recent_achievements(self):
        """Get recently unlocked achievements."""
        recent = []
        for name, achievement in self.achievement_system.items():
            if achievement['unlocked']:
                recent.append({
                    'name': name.replace('_', ' ').title(),
                    'description': achievement['description'],
                    'icon': self._get_achievement_icon(name)
                })
        return recent[-3:]  # Return last 3 achievements
    
    def _get_achievement_icon(self, achievement_name):
        """Get icon for achievement."""
        icons = {
            'data_explorer': '🔍',
            'pattern_master': '🧩',
            'pipeline_guru': '🏗️',
            'efficiency_expert': '⚡'
        }
        return icons.get(achievement_name, '🏆')
    
    def get_pipeline_history(self):
        """Get history of pipeline recommendations."""
        return self.pipeline_recommendations
    
    def get_achievement_status(self):
        """Get current achievement status."""
        return {
            'achievements': self.achievement_system,
            'stats': {
                'total_analyses': self.analysis_count,
                'high_confidence_analyses': self.high_confidence_count,
                'success_rate': (self.high_confidence_count / max(1, self.analysis_count)) * 100
            }
        }
    
    def export_recommendations(self, filepath):
        """Export recommendations to file."""
        export_data = {
            'recommendations': self.pipeline_recommendations,
            'achievements': self.achievement_system,
            'stats': {
                'total_analyses': self.analysis_count,
                'high_confidence_analyses': self.high_confidence_count
            },
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return True