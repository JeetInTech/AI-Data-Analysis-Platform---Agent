import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class ModelExplainer:
    """Advanced model explainability and interpretability system with gamified insights."""
    
    def __init__(self):
        self.explanations = {}
        self.analysis_cache = {}
        
        # Gamification elements
        self.insight_points = 0
        self.explanation_badges = {
            'shap_master': {'points': 100, 'earned': False, 'description': 'Generated 5 SHAP explanations'},
            'lime_expert': {'points': 75, 'earned': False, 'description': 'Created detailed LIME analysis'},
            'error_detective': {'points': 150, 'earned': False, 'description': 'Identified model weaknesses'},
            'confidence_analyst': {'points': 125, 'earned': False, 'description': 'Analyzed prediction confidence'},
            'feature_hunter': {'points': 90, 'earned': False, 'description': 'Discovered key features'}
        }
        self.explanation_count = 0
        
        # Styling for visualizations
        self.colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c', 
            'neutral': '#95a5a6',
            'accent': '#3498db',
            'warning': '#f39c12',
            'background': '#2c3e50',
            'text': '#ecf0f1'
        }
        
        # Set visualization style
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    def generate_shap_analysis(self, ml_engine):
        """Generate comprehensive SHAP analysis with gamified insights."""
        if not SHAP_AVAILABLE:
            return "❌ SHAP not available. Install with: pip install shap"
        
        try:
            self.explanation_count += 1
            analysis_results = []
            
            analysis_results.append("🧩 SHAP ANALYSIS REPORT")
            analysis_results.append("=" * 50)
            analysis_results.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            analysis_results.append(f"🎯 Analysis #{self.explanation_count}")
            analysis_results.append("")
            
            if not hasattr(ml_engine, 'models') or not ml_engine.models:
                return "❌ No trained models available for SHAP analysis"
            
            # Analyze each model
            for model_name, model_info in ml_engine.models.items():
                if 'model' not in model_info:
                    continue
                    
                model = model_info['model']
                analysis_results.append(f"🤖 Model: {model_name.upper()}")
                analysis_results.append("-" * 30)
                
                try:
                    # Create SHAP explainer
                    X_sample = ml_engine.X_test.sample(min(100, len(ml_engine.X_test)))
                    
                    # Choose appropriate explainer
                    if hasattr(model, 'predict_proba'):
                        explainer = shap.Explainer(model.predict, X_sample)
                    else:
                        explainer = shap.Explainer(model.predict, X_sample)
                    
                    shap_values = explainer(X_sample)
                    
                    # Global feature importance
                    feature_importance = np.abs(shap_values.values).mean(0)
                    feature_names = ml_engine.X_test.columns
                    
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                    
                    analysis_results.append("🎯 TOP FEATURE IMPACTS:")
                    for i, row in importance_df.head(10).iterrows():
                        impact_level = self._get_impact_level(row['importance'])
                        analysis_results.append(f"  {impact_level} {row['feature']}: {row['importance']:.4f}")
                    
                    analysis_results.append("")
                    
                    # Sample prediction explanation
                    if len(X_sample) > 0:
                        sample_idx = 0
                        sample_shap = shap_values.values[sample_idx]
                        
                        analysis_results.append(f"📋 SAMPLE PREDICTION BREAKDOWN (Instance {sample_idx}):")
                        for feature, shap_val in zip(feature_names, sample_shap):
                            contribution = "📈 Positive" if shap_val > 0 else "📉 Negative" if shap_val < 0 else "➖ Neutral"
                            analysis_results.append(f"  {contribution} {feature}: {shap_val:.4f}")
                    
                    # Store for visualization
                    self.explanations[f"{model_name}_shap"] = {
                        'shap_values': shap_values,
                        'feature_importance': importance_df,
                        'explainer': explainer,
                        'timestamp': datetime.now()
                    }
                    
                    # Award points
                    self.insight_points += 20
                    analysis_results.append(f"🎮 +20 Insight Points! Total: {self.insight_points}")
                    
                except Exception as e:
                    analysis_results.append(f"❌ SHAP analysis failed for {model_name}: {str(e)}")
                
                analysis_results.append("")
            
            # Check for badges
            self._check_shap_badges()
            
            # Add insights summary
            analysis_results.append("💡 KEY INSIGHTS:")
            insights = self._generate_shap_insights()
            for insight in insights:
                analysis_results.append(f"  🔍 {insight}")
            
            return "\n".join(analysis_results)
            
        except Exception as e:
            return f"❌ SHAP analysis error: {str(e)}"
    
    def generate_lime_explanations(self, ml_engine, num_instances=5):
        """Generate LIME explanations for sample instances."""
        if not LIME_AVAILABLE:
            return "❌ LIME not available. Install with: pip install lime"
        
        try:
            analysis_results = []
            analysis_results.append("🔍 LIME EXPLANATIONS REPORT")
            analysis_results.append("=" * 50)
            analysis_results.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            analysis_results.append("")
            
            if not hasattr(ml_engine, 'models') or not ml_engine.models:
                return "❌ No trained models available for LIME analysis"
            
            # Get best model
            best_model_name = getattr(ml_engine, 'best_model_name', list(ml_engine.models.keys())[0])
            model_info = ml_engine.models[best_model_name]
            model = model_info['model']
            
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                ml_engine.X_train.values,
                feature_names=ml_engine.X_train.columns,
                class_names=['Class 0', 'Class 1'] if hasattr(model, 'predict_proba') else ['Target'],
                mode='classification' if hasattr(model, 'predict_proba') else 'regression',
                discretize_continuous=True
            )
            
            analysis_results.append(f"🤖 Analyzing model: {best_model_name.upper()}")
            analysis_results.append(f"🎯 Explaining {num_instances} sample predictions")
            analysis_results.append("")
            
            # Explain sample instances
            sample_instances = ml_engine.X_test.sample(min(num_instances, len(ml_engine.X_test)))
            
            for i, (idx, instance) in enumerate(sample_instances.iterrows()):
                analysis_results.append(f"📊 INSTANCE {i+1} (Index: {idx})")
                analysis_results.append("-" * 25)
                
                try:
                    # Generate explanation
                    explanation = explainer.explain_instance(
                        instance.values,
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        num_features=min(10, len(instance))
                    )
                    
                    # Get prediction
                    if hasattr(model, 'predict_proba'):
                        prediction = model.predict_proba([instance.values])[0]
                        predicted_class = np.argmax(prediction)
                        confidence = prediction[predicted_class]
                        analysis_results.append(f"🎯 Prediction: Class {predicted_class} (Confidence: {confidence:.3f})")
                    else:
                        prediction = model.predict([instance.values])[0]
                        analysis_results.append(f"🎯 Prediction: {prediction:.3f}")
                    
                    # Feature contributions
                    feature_contributions = explanation.as_list()
                    analysis_results.append("🔍 Feature Contributions:")
                    
                    for feature_desc, contribution in feature_contributions:
                        contribution_type = "🟢 Supporting" if contribution > 0 else "🔴 Opposing"
                        analysis_results.append(f"  {contribution_type} {feature_desc}: {contribution:.4f}")
                    
                    # Store explanation
                    self.explanations[f"lime_instance_{i}"] = {
                        'explanation': explanation,
                        'instance': instance,
                        'prediction': prediction,
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    analysis_results.append(f"❌ LIME explanation failed for instance {i}: {str(e)}")
                
                analysis_results.append("")
            
            # Award points and check badges
            self.insight_points += 15 * num_instances
            analysis_results.append(f"🎮 +{15 * num_instances} Insight Points! Total: {self.insight_points}")
            
            self._check_lime_badges()
            
            return "\n".join(analysis_results)
            
        except Exception as e:
            return f"❌ LIME analysis error: {str(e)}"
    
    def perform_error_analysis(self, ml_engine):
        """Comprehensive error analysis with gamified insights."""
        try:
            analysis_results = []
            analysis_results.append("🕵️ ERROR ANALYSIS DETECTIVE REPORT")
            analysis_results.append("=" * 50)
            analysis_results.append(f"📅 Investigation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            analysis_results.append("")
            
            if not hasattr(ml_engine, 'models') or not ml_engine.models:
                return "❌ No trained models available for error analysis"
            
            detective_points = 0
            
            for model_name, model_info in ml_engine.models.items():
                if 'model' not in model_info:
                    continue
                    
                model = model_info['model']
                analysis_results.append(f"🔍 INVESTIGATING MODEL: {model_name.upper()}")
                analysis_results.append("-" * 40)
                
                try:
                    # Get predictions
                    if hasattr(model, 'predict_proba'):
                        predictions = model.predict(ml_engine.X_test)
                        probabilities = model.predict_proba(ml_engine.X_test)
                        
                        # Classification analysis
                        if SKLEARN_AVAILABLE:
                            cm = confusion_matrix(ml_engine.y_test, predictions)
                            analysis_results.append("📊 CONFUSION MATRIX ANALYSIS:")
                            analysis_results.append(f"True Negatives: {cm[0,0]}")
                            analysis_results.append(f"False Positives: {cm[0,1]} ⚠️")
                            analysis_results.append(f"False Negatives: {cm[1,0]} ⚠️")
                            analysis_results.append(f"True Positives: {cm[1,1]}")
                            
                            # Error rates
                            total_errors = cm[0,1] + cm[1,0]
                            total_predictions = cm.sum()
                            error_rate = total_errors / total_predictions
                            
                            analysis_results.append(f"🎯 Overall Error Rate: {error_rate:.3f} ({error_rate*100:.1f}%)")
                            
                            # Type I and Type II errors
                            type1_error = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
                            type2_error = cm[1,0] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
                            
                            analysis_results.append(f"🔴 Type I Error (False Positive Rate): {type1_error:.3f}")
                            analysis_results.append(f"🔴 Type II Error (False Negative Rate): {type2_error:.3f}")
                            
                            detective_points += 25
                        
                        # Confidence analysis
                        max_probs = np.max(probabilities, axis=1)
                        low_confidence_mask = max_probs < 0.7
                        low_confidence_count = np.sum(low_confidence_mask)
                        
                        analysis_results.append("")
                        analysis_results.append("🎲 CONFIDENCE ANALYSIS:")
                        analysis_results.append(f"Low Confidence Predictions: {low_confidence_count} ({low_confidence_count/len(max_probs)*100:.1f}%)")
                        analysis_results.append(f"Average Confidence: {np.mean(max_probs):.3f}")
                        analysis_results.append(f"Confidence Std Dev: {np.std(max_probs):.3f}")
                        
                        detective_points += 15
                        
                    else:
                        # Regression analysis
                        predictions = model.predict(ml_engine.X_test)
                        residuals = ml_engine.y_test - predictions
                        
                        analysis_results.append("📈 REGRESSION ERROR ANALYSIS:")
                        analysis_results.append(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")
                        analysis_results.append(f"Root Mean Square Error: {np.sqrt(np.mean(residuals**2)):.4f}")
                        analysis_results.append(f"Mean Residual: {np.mean(residuals):.4f}")
                        analysis_results.append(f"Residual Std Dev: {np.std(residuals):.4f}")
                        
                        # Large error analysis
                        large_errors = np.abs(residuals) > 2 * np.std(residuals)
                        large_error_count = np.sum(large_errors)
                        
                        analysis_results.append(f"🚨 Large Errors (>2σ): {large_error_count} ({large_error_count/len(residuals)*100:.1f}%)")
                        
                        detective_points += 20
                    
                    # Feature error correlation (if available)
                    if len(ml_engine.X_test.columns) > 0:
                        analysis_results.append("")
                        analysis_results.append("🔗 ERROR PATTERN ANALYSIS:")
                        
                        # Find features correlated with errors
                        if hasattr(model, 'predict_proba'):
                            error_mask = predictions != ml_engine.y_test
                        else:
                            error_mask = np.abs(residuals) > np.std(residuals)
                        
                        error_correlations = []
                        for col in ml_engine.X_test.columns:
                            if ml_engine.X_test[col].dtype in [np.number]:
                                corr = np.corrcoef(ml_engine.X_test[col], error_mask.astype(int))[0,1]
                                if not np.isnan(corr):
                                    error_correlations.append((col, abs(corr)))
                        
                        error_correlations.sort(key=lambda x: x[1], reverse=True)
                        
                        analysis_results.append("🎯 Features Most Associated with Errors:")
                        for feature, corr in error_correlations[:5]:
                            risk_level = "🔴 HIGH" if corr > 0.3 else "🟡 MEDIUM" if corr > 0.1 else "🟢 LOW"
                            analysis_results.append(f"  {risk_level} {feature}: {corr:.3f}")
                        
                        detective_points += 30
                
                except Exception as e:
                    analysis_results.append(f"❌ Error analysis failed for {model_name}: {str(e)}")
                
                analysis_results.append("")
            
            # Award detective points
            self.insight_points += detective_points
            analysis_results.append(f"🕵️ +{detective_points} Detective Points! Total: {self.insight_points}")
            
            # Check for error detective badge
            self._check_error_badges()
            
            # Summary recommendations
            analysis_results.append("💡 DETECTIVE RECOMMENDATIONS:")
            recommendations = self._generate_error_recommendations()
            for rec in recommendations:
                analysis_results.append(f"  🎯 {rec}")
            
            return "\n".join(analysis_results)
            
        except Exception as e:
            return f"❌ Error analysis failed: {str(e)}"
    
    def analyze_feature_contributions(self, ml_engine):
        """Analyze feature contributions across models."""
        try:
            analysis_results = []
            analysis_results.append("🎯 FEATURE CONTRIBUTION ANALYSIS")
            analysis_results.append("=" * 50)
            analysis_results.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            analysis_results.append("")
            
            if not hasattr(ml_engine, 'models') or not ml_engine.models:
                return "❌ No trained models available for feature analysis"
            
            feature_importance_summary = {}
            
            for model_name, model_info in ml_engine.models.items():
                if 'model' not in model_info:
                    continue
                    
                model = model_info['model']
                analysis_results.append(f"🔍 ANALYZING: {model_name.upper()}")
                analysis_results.append("-" * 30)
                
                try:
                    # Get feature importance
                    importance_scores = None
                    
                    # Try different methods to get feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance_scores = model.feature_importances_
                        method = "Built-in Feature Importance"
                    elif hasattr(model, 'coef_'):
                        importance_scores = np.abs(model.coef_).flatten()
                        method = "Coefficient Magnitude"
                    elif SKLEARN_AVAILABLE:
                        # Use permutation importance
                        perm_importance = permutation_importance(
                            model, ml_engine.X_test, ml_engine.y_test, 
                            n_repeats=5, random_state=42
                        )
                        importance_scores = perm_importance.importances_mean
                        method = "Permutation Importance"
                    
                    if importance_scores is not None:
                        # Create feature importance dataframe
                        feature_names = ml_engine.X_test.columns
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importance_scores,
                            'model': model_name
                        }).sort_values('importance', ascending=False)
                        
                        analysis_results.append(f"📊 Method: {method}")
                        analysis_results.append("🏆 TOP 10 FEATURES:")
                        
                        for i, row in importance_df.head(10).iterrows():
                            importance_level = self._get_importance_level(row['importance'])
                            analysis_results.append(f"  {i+1:2d}. {importance_level} {row['feature']}: {row['importance']:.4f}")
                        
                        # Store for summary
                        for _, row in importance_df.iterrows():
                            if row['feature'] not in feature_importance_summary:
                                feature_importance_summary[row['feature']] = []
                            feature_importance_summary[row['feature']].append(row['importance'])
                        
                        # Identify feature categories
                        high_impact = importance_df[importance_df['importance'] > importance_df['importance'].quantile(0.8)]
                        medium_impact = importance_df[
                            (importance_df['importance'] > importance_df['importance'].quantile(0.5)) &
                            (importance_df['importance'] <= importance_df['importance'].quantile(0.8))
                        ]
                        
                        analysis_results.append("")
                        analysis_results.append(f"🔴 High Impact Features: {len(high_impact)}")
                        analysis_results.append(f"🟡 Medium Impact Features: {len(medium_impact)}")
                        analysis_results.append(f"🟢 Low Impact Features: {len(importance_df) - len(high_impact) - len(medium_impact)}")
                    
                    else:
                        analysis_results.append("❌ Could not extract feature importance")
                
                except Exception as e:
                    analysis_results.append(f"❌ Feature analysis failed: {str(e)}")
                
                analysis_results.append("")
            
            # Cross-model feature importance summary
            if feature_importance_summary:
                analysis_results.append("🌟 CROSS-MODEL FEATURE RANKING")
                analysis_results.append("-" * 40)
                
                # Calculate average importance across models
                avg_importance = {}
                for feature, importances in feature_importance_summary.items():
                    avg_importance[feature] = np.mean(importances)
                
                # Sort by average importance
                sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                
                analysis_results.append("🏆 CONSENSUS TOP FEATURES:")
                for i, (feature, avg_imp) in enumerate(sorted_features[:15]):
                    consistency = len(feature_importance_summary[feature])
                    consistency_icon = "🔥" if consistency == len(ml_engine.models) else "⭐" if consistency > 1 else "📌"
                    analysis_results.append(f"  {i+1:2d}. {consistency_icon} {feature}: {avg_imp:.4f} (in {consistency} models)")
                
                # Award feature hunter points
                self.insight_points += 50
                analysis_results.append(f"🎮 +50 Feature Hunter Points! Total: {self.insight_points}")
                
                self._check_feature_badges()
            
            return "\n".join(analysis_results)
            
        except Exception as e:
            return f"❌ Feature analysis error: {str(e)}"
    
    def create_confidence_analysis(self, ml_engine):
        """Create prediction confidence analysis visualization."""
        try:
            if not hasattr(ml_engine, 'models') or not ml_engine.models:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No trained models available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                return fig
            
            # Get best model with probability predictions
            best_model = None
            best_model_name = None
            
            for model_name, model_info in ml_engine.models.items():
                if 'model' in model_info and hasattr(model_info['model'], 'predict_proba'):
                    best_model = model_info['model']
                    best_model_name = model_name
                    break
            
            if best_model is None:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No models with probability predictions available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            # Create confidence analysis
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.patch.set_facecolor(self.colors['background'])
            fig.suptitle(f'🎲 Prediction Confidence Analysis - {best_model_name}', 
                        fontsize=16, fontweight='bold', color=self.colors['text'])
            
            # Get predictions and probabilities
            probabilities = best_model.predict_proba(ml_engine.X_test)
            max_probs = np.max(probabilities, axis=1)
            predictions = best_model.predict(ml_engine.X_test)
            
            # 1. Confidence distribution
            ax1 = axes[0, 0]
            ax1.hist(max_probs, bins=30, alpha=0.7, color=self.colors['accent'], edgecolor='white')
            ax1.axvline(np.mean(max_probs), color=self.colors['positive'], linestyle='--', 
                       label=f'Mean: {np.mean(max_probs):.3f}')
            ax1.set_xlabel('Prediction Confidence', color=self.colors['text'])
            ax1.set_ylabel('Frequency', color=self.colors['text'])
            ax1.set_title('🎯 Confidence Distribution', fontweight='bold', color=self.colors['text'])
            ax1.legend()
            ax1.set_facecolor(self.colors['background'])
            
            # 2. Confidence vs Accuracy
            ax2 = axes[0, 1]
            correct_predictions = (predictions == ml_engine.y_test).astype(int)
            
            # Bin by confidence levels
            conf_bins = np.linspace(0, 1, 11)
            bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
            bin_accuracies = []
            
            for i in range(len(conf_bins) - 1):
                mask = (max_probs >= conf_bins[i]) & (max_probs < conf_bins[i+1])
                if np.sum(mask) > 0:
                    bin_accuracies.append(np.mean(correct_predictions[mask]))
                else:
                    bin_accuracies.append(0)
            
            ax2.plot(bin_centers, bin_accuracies, 'o-', color=self.colors['positive'], linewidth=2, markersize=8)
            ax2.plot([0, 1], [0, 1], '--', color=self.colors['neutral'], alpha=0.7, label='Perfect Calibration')
            ax2.set_xlabel('Prediction Confidence', color=self.colors['text'])
            ax2.set_ylabel('Actual Accuracy', color=self.colors['text'])
            ax2.set_title('📊 Calibration Curve', fontweight='bold', color=self.colors['text'])
            ax2.legend()
            ax2.set_facecolor(self.colors['background'])
            
            # 3. Low confidence instances
            ax3 = axes[1, 0]
            low_conf_threshold = 0.7
            low_conf_mask = max_probs < low_conf_threshold
            low_conf_correct = correct_predictions[low_conf_mask]
            
            if len(low_conf_correct) > 0:
                labels = ['Incorrect', 'Correct']
                counts = [np.sum(low_conf_correct == 0), np.sum(low_conf_correct == 1)]
                colors_pie = [self.colors['negative'], self.colors['positive']]
                
                wedges, texts, autotexts = ax3.pie(counts, labels=labels, colors=colors_pie, 
                                                  autopct='%1.1f%%', startangle=90)
                ax3.set_title(f'🚨 Low Confidence Predictions\n(<{low_conf_threshold} confidence)', 
                             fontweight='bold', color=self.colors['text'])
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                ax3.text(0.5, 0.5, '🎉 No Low Confidence\nPredictions!', ha='center', va='center',
                        fontsize=14, fontweight='bold', color=self.colors['positive'])
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
            
            ax3.set_facecolor(self.colors['background'])
            
            # 4. Confidence by class
            ax4 = axes[1, 1]
            unique_classes = np.unique(ml_engine.y_test)
            
            if len(unique_classes) <= 5:  # Only show if not too many classes
                class_confidences = []
                class_labels = []
                
                for cls in unique_classes:
                    mask = ml_engine.y_test == cls
                    class_conf = max_probs[mask]
                    class_confidences.append(class_conf)
                    class_labels.append(f'Class {cls}')
                
                box_plot = ax4.boxplot(class_confidences, labels=class_labels, patch_artist=True)
                
                # Color boxes
                for patch in box_plot['boxes']:
                    patch.set_facecolor(self.colors['accent'])
                    patch.set_alpha(0.7)
                
                ax4.set_ylabel('Prediction Confidence', color=self.colors['text'])
                ax4.set_title('📈 Confidence by Class', fontweight='bold', color=self.colors['text'])
                ax4.set_facecolor(self.colors['background'])
            else:
                ax4.text(0.5, 0.5, 'Too many classes\nfor visualization', ha='center', va='center',
                        fontsize=12, color=self.colors['text'])
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.set_facecolor(self.colors['background'])
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Award confidence analysis points
            self.insight_points += 35
            self._check_confidence_badges()
            
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Confidence analysis error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
    
    def create_shap_waterfall(self, ml_engine, instance_idx=0):
        """Create SHAP waterfall plot for a specific instance."""
        if not SHAP_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'SHAP not available\nInstall with: pip install shap', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        try:
            # Get model and create explainer
            best_model_name = getattr(ml_engine, 'best_model_name', list(ml_engine.models.keys())[0])
            model = ml_engine.models[best_model_name]['model']
            
            # Create SHAP explainer
            X_sample = ml_engine.X_test.head(100)
            explainer = shap.Explainer(model.predict, X_sample)
            
            # Get SHAP values for specific instance
            instance = ml_engine.X_test.iloc[instance_idx:instance_idx+1]
            shap_values = explainer(instance)
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor(self.colors['background'])
            
            shap.waterfall_plot(shap_values[0], show=False)
            
            plt.title(f'🌊 SHAP Waterfall - Instance {instance_idx}', 
                     fontsize=16, fontweight='bold', color=self.colors['text'])
            
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'SHAP waterfall error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
    
    def create_calibration_curves(self, ml_engine):
        """Create model calibration curves."""
        if not SKLEARN_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Sklearn not available for calibration analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor(self.colors['background'])
            ax.set_facecolor(self.colors['background'])
            
            # Plot calibration curves for models with probability predictions
            model_count = 0
            colors = [self.colors['positive'], self.colors['accent'], self.colors['warning'], 
                     self.colors['negative']]
            
            for i, (model_name, model_info) in enumerate(ml_engine.models.items()):
                if 'model' not in model_info:
                    continue
                    
                model = model_info['model']
                if not hasattr(model, 'predict_proba'):
                    continue
                
                # Get probabilities
                probabilities = model.predict_proba(ml_engine.X_test)
                
                if probabilities.shape[1] == 2:  # Binary classification
                    prob_pos = probabilities[:, 1]
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        ml_engine.y_test, prob_pos, n_bins=10
                    )
                    
                    color = colors[model_count % len(colors)]
                    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                           color=color, linewidth=2, markersize=8, 
                           label=f'{model_name} (Calibration)')
                    model_count += 1
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
            
            ax.set_xlabel('Mean Predicted Probability', fontsize=12, color=self.colors['text'])
            ax.set_ylabel('Fraction of Positives', fontsize=12, color=self.colors['text'])
            ax.set_title('📊 Model Calibration Curves', fontsize=16, fontweight='bold', 
                        color=self.colors['text'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Calibration analysis error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
    
    # Helper methods
    def _get_impact_level(self, importance):
        """Get impact level icon based on importance score."""
        if importance > 0.1:
            return "🔥 HIGH"
        elif importance > 0.05:
            return "⭐ MEDIUM"
        else:
            return "📌 LOW"
    
    def _get_importance_level(self, importance):
        """Get importance level icon."""
        return self._get_impact_level(importance)
    
    def _generate_shap_insights(self):
        """Generate insights from SHAP analysis."""
        insights = [
            "Feature interactions reveal complex model behavior patterns",
            "Top features drive majority of prediction variance",
            "SHAP values provide local explanation for each prediction",
            "Global feature importance may differ from local explanations"
        ]
        return insights
    
    def _generate_error_recommendations(self):
        """Generate recommendations based on error analysis."""
        recommendations = [
            "Focus on features most correlated with prediction errors",
            "Consider ensemble methods to reduce high-variance errors",
            "Implement confidence thresholding for uncertain predictions",
            "Collect more data for low-confidence prediction regions"
        ]
        return recommendations
    
    def _check_shap_badges(self):
        """Check and award SHAP-related badges."""
        if self.explanation_count >= 5 and not self.explanation_badges['shap_master']['earned']:
            self.explanation_badges['shap_master']['earned'] = True
            self.insight_points += self.explanation_badges['shap_master']['points']
    
    def _check_lime_badges(self):
        """Check and award LIME-related badges."""
        if not self.explanation_badges['lime_expert']['earned']:
            self.explanation_badges['lime_expert']['earned'] = True
            self.insight_points += self.explanation_badges['lime_expert']['points']
    
    def _check_error_badges(self):
        """Check and award error analysis badges."""
        if not self.explanation_badges['error_detective']['earned']:
            self.explanation_badges['error_detective']['earned'] = True
            self.insight_points += self.explanation_badges['error_detective']['points']
    
    def _check_confidence_badges(self):
        """Check and award confidence analysis badges."""
        if not self.explanation_badges['confidence_analyst']['earned']:
            self.explanation_badges['confidence_analyst']['earned'] = True
            self.insight_points += self.explanation_badges['confidence_analyst']['points']
    
    def _check_feature_badges(self):
        """Check and award feature analysis badges."""
        if not self.explanation_badges['feature_hunter']['earned']:
            self.explanation_badges['feature_hunter']['earned'] = True
            self.insight_points += self.explanation_badges['feature_hunter']['points']
    
    def get_explanation_summary(self):
        """Get summary of all explanations generated."""
        return {
            'total_explanations': len(self.explanations),
            'insight_points': self.insight_points,
            'badges_earned': {name: badge['earned'] for name, badge in self.explanation_badges.items()},
            'available_explanations': list(self.explanations.keys())
        }
    
    def export_explanations(self, filepath):
        """Export explanations to file."""
        export_data = {
            'summary': self.get_explanation_summary(),
            'explanations_metadata': {
                key: {
                    'timestamp': exp['timestamp'].isoformat(),
                    'type': key.split('_')[0]
                }
                for key, exp in self.explanations.items()
            },
            'exported_at': datetime.now().isoformat()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return True