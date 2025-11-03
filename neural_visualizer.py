import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False

class SmartVisualizer:
    """Enhanced data visualization system with 3D, interactive, and gamified features."""
    
    def __init__(self):
        self.fig_size = (14, 10)
        self.color_palette = sns.color_palette("husl", 12)
        self.gamification_colors = {
            'success': '#2ecc71',
            'warning': '#f39c12', 
            'error': '#e74c3c',
            'info': '#3498db',
            'achievement': '#9b59b6',
            'progress': '#1abc9c'
        }
        
        # Professional styling with gamification elements
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Enhanced matplotlib parameters for gamification
        plt.rcParams.update({
            'figure.facecolor': '#2c3e50',
            'axes.facecolor': '#34495e',
            'axes.edgecolor': '#ecf0f1',
            'axes.linewidth': 1.2,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'figure.titleweight': 'bold',
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'grid.alpha': 0.4,
            'text.color': '#ecf0f1',
            'axes.labelcolor': '#ecf0f1',
            'xtick.color': '#ecf0f1',
            'ytick.color': '#ecf0f1'
        })
        
        # Animation settings
        self.animation_speed = 50  # milliseconds
        self.achievement_count = 0
        
    def create_professional_data_overview(self, original_data, processed_data=None, achievements=None):
        """Create professional data overview with advanced analytics."""
        comparison_data = processed_data if processed_data is not None else original_data
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.patch.set_facecolor('#2c3e50')
        fig.suptitle('DATA ANALYTICS DASHBOARD - PROFESSIONAL INSIGHTS', fontsize=20, fontweight='bold', 
                     color='#ecf0f1', y=0.98)
        
        # Add achievement banner if achievements exist
        if achievements:
            self._add_achievement_banner(fig, achievements)
        
        # 1. Progress Ring - Dataset Completion
        ax1 = axes[0, 0]
        self._create_progress_ring(ax1, original_data, comparison_data)
        
        # 2. 3D Data Quality Analysis
        ax2 = axes[0, 1]
        ax2.remove()
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        self._create_3d_quality_analysis(ax2, original_data, comparison_data)
        
        # 3. Advanced Missing Values Analysis
        ax3 = axes[0, 2]
        self._create_advanced_missing_analysis(ax3, original_data, comparison_data)
        
        # 4. Data Distribution Flow
        ax4 = axes[1, 0]
        self._create_data_distribution_flow(ax4, comparison_data)
        
        # 5. Performance Metrics
        ax5 = axes[1, 1]
        self._create_performance_metrics(ax5, original_data, comparison_data)
        
        # 6. Quality Improvement Tracking
        ax6 = axes[1, 2]
        self._create_quality_tracking(ax6, original_data, comparison_data)
        
        # Add interactive elements
        self._add_hover_effects(fig, axes)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def _create_progress_ring(self, ax, original_data, comparison_data):
        """Create a progress ring showing data completion."""
        # Calculate completion metrics
        original_completeness = self._calculate_completeness(original_data)
        processed_completeness = self._calculate_completeness(comparison_data)
        
        # Create circular progress
        angles = np.linspace(0, 2*np.pi, 100)
        
        # Background ring
        r_outer = 1.0
        r_inner = 0.7
        ax.fill_between(angles, r_inner, r_outer, alpha=0.3, color='#34495e')
        
        # Progress ring
        progress_angle = 2 * np.pi * (processed_completeness / 100)
        progress_angles = angles[:int(len(angles) * processed_completeness / 100)]
        
        # Gradient colors for progress
        colors = plt.cm.viridis(np.linspace(0, 1, len(progress_angles)))
        for i in range(len(progress_angles)-1):
            ax.fill_between([progress_angles[i], progress_angles[i+1]], 
                           r_inner, r_outer, color=colors[i], alpha=0.8)
        
        # Center text
        ax.text(0, 0, f'{processed_completeness:.1f}%\nCOMPLETE', 
               ha='center', va='center', fontsize=16, fontweight='bold',
               color='#ecf0f1')
        
        # Quality indicator
        if processed_completeness > 95:
            ax.text(0, -0.3, 'EXCELLENT QUALITY', ha='center', va='center', 
                   fontsize=12, color=self.gamification_colors['achievement'])
        elif processed_completeness > 90:
            ax.text(0, -0.3, 'HIGH QUALITY', ha='center', va='center', 
                   fontsize=12, color=self.gamification_colors['success'])
        elif processed_completeness > 80:
            ax.text(0, -0.3, 'GOOD QUALITY', ha='center', va='center', 
                   fontsize=12, color=self.gamification_colors['progress'])
        else:
            ax.text(0, -0.3, 'NEEDS IMPROVEMENT', ha='center', va='center', 
                   fontsize=12, color=self.gamification_colors['warning'])
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('DATA COMPLETION PROGRESS', fontweight='bold', pad=20, color='#ecf0f1')
    
    def _create_3d_quality_analysis(self, ax, original_data, comparison_data):
        """Create 3D visualization of comprehensive data quality metrics."""
        # Enhanced quality metrics
        completeness = self._calculate_completeness(comparison_data) / 100
        consistency = self._calculate_enhanced_consistency(comparison_data)
        reliability = self._calculate_data_reliability(original_data, comparison_data)
        
        # Create sophisticated 3D visualization
        r = [0, 1]
        X, Y = np.meshgrid(r, r)
        
        # Create gradient surfaces
        Z_front = np.ones_like(X) * completeness
        colors_front = plt.cm.viridis(completeness)
        ax.plot_surface(X, Y, Z_front, alpha=0.7, color=colors_front)
        
        Z_side = np.ones_like(X) * consistency
        colors_side = plt.cm.plasma(consistency)
        ax.plot_surface(Z_side, X, Y, alpha=0.7, color=colors_side)
        
        Z_top = np.ones_like(X) * reliability
        colors_top = plt.cm.inferno(reliability)
        ax.plot_surface(X, Z_top, Y, alpha=0.7, color=colors_top)
        
        # Calculate overall quality score with weighted metrics
        overall_quality = (completeness * 0.4 + consistency * 0.3 + reliability * 0.3) * 100
        
        # Add quality assessment
        ax.text(0.5, 0.5, 1.2, f'OVERALL QUALITY\n{overall_quality:.1f}/100\n{self._get_quality_grade(overall_quality)}', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='#ecf0f1', bbox=dict(boxstyle="round,pad=0.3", facecolor='#2c3e50', alpha=0.8))
        
        # Enhanced labels
        ax.set_xlabel('Completeness', color='#ecf0f1', fontweight='bold')
        ax.set_ylabel('Consistency', color='#ecf0f1', fontweight='bold')
        ax.set_zlabel('Reliability', color='#ecf0f1', fontweight='bold')
        ax.set_title('3D DATA QUALITY ANALYSIS', fontweight='bold', color='#ecf0f1')
        
        # Style 3D plot with better aesthetics
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.2)
        ax.xaxis.pane.set_edgecolor('#ecf0f1')
        ax.yaxis.pane.set_edgecolor('#ecf0f1')
        ax.zaxis.pane.set_edgecolor('#ecf0f1')
    
    def _create_gamified_missing_values(self, ax, original_data, comparison_data):
        """Create gamified missing values visualization."""
        original_missing = original_data.isnull().sum()
        comparison_missing = comparison_data.isnull().sum()
        
        # Get top missing columns
        missing_cols = original_missing[original_missing > 0].sort_values(ascending=False).head(10)
        
        if len(missing_cols) == 0:
            # Achievement display for clean data
            ax.text(0.5, 0.5, '🏆 PERFECT DATA!\nZERO MISSING VALUES', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', color=self.gamification_colors['achievement'],
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#9b59b6', alpha=0.3))
            ax.set_title('🎯 MISSING VALUES CHALLENGE', fontweight='bold', color='#ecf0f1')
            return
        
        # Create heatmap-style visualization
        y_pos = np.arange(len(missing_cols))
        
        # Calculate improvement for each column
        improvements = []
        for col in missing_cols.index:
            original_count = missing_cols[col]
            current_count = comparison_missing.get(col, 0)
            improvement = ((original_count - current_count) / original_count) * 100
            improvements.append(improvement)
        
        # Create horizontal bars with gradient colors
        colors = []
        for improvement in improvements:
            if improvement >= 100:
                colors.append(self.gamification_colors['achievement'])
            elif improvement >= 75:
                colors.append(self.gamification_colors['success'])
            elif improvement >= 50:
                colors.append(self.gamification_colors['progress'])
            else:
                colors.append(self.gamification_colors['warning'])
        
        bars = ax.barh(y_pos, missing_cols.values, color=colors, alpha=0.8, height=0.6)
        
        # Add improvement indicators
        for i, (bar, improvement) in enumerate(zip(bars, improvements)):
            width = bar.get_width()
            if improvement >= 100:
                ax.text(width + max(missing_cols.values) * 0.02, i, '🏆 FIXED!', 
                       va='center', ha='left', fontweight='bold', 
                       color=self.gamification_colors['achievement'])
            elif improvement > 0:
                ax.text(width + max(missing_cols.values) * 0.02, i, f'⬆️ {improvement:.0f}%', 
                       va='center', ha='left', fontweight='bold', 
                       color=self.gamification_colors['success'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{col[:12]}...' if len(col) > 12 else col for col in missing_cols.index])
        ax.set_xlabel('Missing Values Count', fontweight='bold', color='#ecf0f1')
        ax.set_title('🎯 MISSING VALUES CHALLENGE', fontweight='bold', color='#ecf0f1')
        ax.invert_yaxis()
    
    def _create_data_types_flow(self, ax, data):
        """Create animated data types flow visualization."""
        dtype_counts = data.dtypes.value_counts()
        
        # Create flow-like visualization
        angles = np.linspace(0, 2*np.pi, len(dtype_counts), endpoint=False)
        
        # Create flowing particles effect
        for i, (dtype, count) in enumerate(dtype_counts.items()):
            angle = angles[i]
            
            # Create flowing lines
            r_values = np.linspace(0.3, 1.0, 50)
            x_flow = r_values * np.cos(angle + 0.1 * np.sin(10 * r_values))
            y_flow = r_values * np.sin(angle + 0.1 * np.sin(10 * r_values))
            
            # Color based on data type
            if 'int' in str(dtype) or 'float' in str(dtype):
                color = self.gamification_colors['info']
                icon = '🔢'
            elif 'object' in str(dtype):
                color = self.gamification_colors['achievement']
                icon = '📝'
            elif 'datetime' in str(dtype):
                color = self.gamification_colors['success']
                icon = '📅'
            else:
                color = self.gamification_colors['warning']
                icon = '❓'
            
            # Plot flowing line
            ax.plot(x_flow, y_flow, color=color, linewidth=3, alpha=0.7)
            
            # Add type label with icon
            ax.text(1.2 * np.cos(angle), 1.2 * np.sin(angle), 
                   f'{icon}\n{str(dtype)}\n({count})', 
                   ha='center', va='center', fontweight='bold',
                   fontsize=10, color='#ecf0f1')
        
        # Center hub
        circle = plt.Circle((0, 0), 0.2, color='#ecf0f1', alpha=0.8)
        ax.add_patch(circle)
        ax.text(0, 0, '🎯\nDATA\nTYPES', ha='center', va='center', 
               fontweight='bold', fontsize=10, color='#2c3e50')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('🌊 DATA TYPES FLOW', fontweight='bold', color='#ecf0f1')
    
    def _create_performance_speedometer(self, ax, original_data, comparison_data):
        """Create speedometer-style performance indicator."""
        # Calculate performance score
        performance_score = self._calculate_performance_score(original_data, comparison_data)
        
        # Speedometer parameters
        theta = np.linspace(np.pi, 0, 100)
        
        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), 'gray', linewidth=10, alpha=0.3)
        
        # Performance zones
        zones = [
            (0, 0.3, self.gamification_colors['error'], 'NEEDS WORK'),
            (0.3, 0.6, self.gamification_colors['warning'], 'GOOD'),
            (0.6, 0.8, self.gamification_colors['progress'], 'GREAT'),
            (0.8, 1.0, self.gamification_colors['achievement'], 'LEGENDARY!')
        ]
        
        for start, end, color, label in zones:
            zone_theta = theta[int(start*99):int(end*99)]
            ax.plot(np.cos(zone_theta), np.sin(zone_theta), color=color, 
                   linewidth=10, alpha=0.8)
        
        # Needle
        needle_angle = np.pi * (1 - performance_score)
        needle_x = [0, 0.8 * np.cos(needle_angle)]
        needle_y = [0, 0.8 * np.sin(needle_angle)]
        ax.plot(needle_x, needle_y, color='#ecf0f1', linewidth=4)
        
        # Center circle
        circle = plt.Circle((0, 0), 0.1, color='#ecf0f1')
        ax.add_patch(circle)
        
        # Score display
        ax.text(0, -0.3, f'{performance_score*100:.0f}/100\nPERFORMANCE', 
               ha='center', va='center', fontsize=14, fontweight='bold',
               color='#ecf0f1')
        
        # Achievement badge
        if performance_score > 0.9:
            ax.text(0, -0.6, '🏆 MASTER ANALYST!', ha='center', va='center', 
                   fontsize=12, color=self.gamification_colors['achievement'])
        elif performance_score > 0.8:
            ax.text(0, -0.6, '⭐ EXPERT LEVEL!', ha='center', va='center', 
                   fontsize=12, color=self.gamification_colors['success'])
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('⚡ PERFORMANCE METER', fontweight='bold', color='#ecf0f1')
    
    def _create_achievement_progress(self, ax, original_data, comparison_data):
        """Create achievement progress bars."""
        achievements = [
            ('Data Explorer', 'Load dataset', 100),
            ('Cleaner', 'Clean missing values', self._calculate_cleaning_progress(original_data, comparison_data)),
            ('Optimizer', 'Improve data quality', self._calculate_optimization_progress(original_data, comparison_data)),
            ('Analyst', 'Complete analysis', 75),
            ('ML Master', 'Train models', 0)
        ]
        
        y_positions = np.arange(len(achievements))
        
        for i, (name, description, progress) in enumerate(achievements):
            # Background bar
            ax.barh(i, 100, height=0.6, color='#34495e', alpha=0.5)
            
            # Progress bar with gradient
            if progress > 0:
                colors = plt.cm.viridis(progress / 100)
                bar = ax.barh(i, progress, height=0.6, color=colors, alpha=0.8)
            
            # Achievement icon
            if progress >= 100:
                icon = '🏆'
                color = self.gamification_colors['achievement']
            elif progress >= 75:
                icon = '⭐'
                color = self.gamification_colors['success']
            elif progress >= 50:
                icon = '🎯'
                color = self.gamification_colors['progress']
            elif progress > 0:
                icon = '🔄'
                color = self.gamification_colors['info']
            else:
                icon = '⭕'
                color = '#7f8c8d'
            
            # Labels
            ax.text(-5, i, icon, ha='center', va='center', fontsize=16)
            ax.text(2, i+0.15, name, ha='left', va='center', fontweight='bold', 
                   fontsize=11, color='#ecf0f1')
            ax.text(2, i-0.15, description, ha='left', va='center', 
                   fontsize=9, color='#bdc3c7', style='italic')
            ax.text(102, i, f'{progress:.0f}%', ha='left', va='center', 
                   fontweight='bold', color=color)
        
        ax.set_xlim(-10, 120)
        ax.set_ylim(-0.5, len(achievements) - 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Progress', fontweight='bold', color='#ecf0f1')
        ax.set_title('🎮 ACHIEVEMENT PROGRESS', fontweight='bold', color='#ecf0f1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def create_3d_visualization(self, data, features=None):
        """Create interactive 3D data visualization."""
        if not PCA_AVAILABLE:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'PCA not available\nInstall scikit-learn for 3D visualization', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Prepare data for 3D visualization
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_data.shape[1] < 3:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'Insufficient numeric features for 3D visualization\nNeed at least 3 numeric columns', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=3)
        transformed_data = pca.fit_transform(numeric_data)
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('#2c3e50')
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#34495e')
        
        # Create 3D scatter plot
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2],
                           c=range(len(transformed_data)), cmap='viridis', 
                           s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # Customize 3D plot
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                     color='#ecf0f1', fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                     color='#ecf0f1', fontweight='bold')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)', 
                     color='#ecf0f1', fontweight='bold')
        
        ax.set_title('🌐 3D Data Space Exploration\nPrincipal Component Analysis', 
                    fontsize=16, fontweight='bold', color='#ecf0f1', pad=20)
        
        # Style 3D plot
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Data Point Index', color='#ecf0f1', fontweight='bold')
        cbar.ax.yaxis.set_tick_params(color='#ecf0f1')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plotly_visualization(self, data):
        """Create interactive Plotly visualization (if available)."""
        if not PLOTLY_AVAILABLE:
            return None
        
        # Prepare numeric data
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_data.shape[1] < 3:
            return None
        
        # Apply PCA
        if PCA_AVAILABLE:
            pca = PCA(n_components=3)
            transformed_data = pca.fit_transform(numeric_data)
            
            # Create 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=transformed_data[:, 0],
                y=transformed_data[:, 1],
                z=transformed_data[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=range(len(transformed_data)),
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=[f'Point {i}' for i in range(len(transformed_data))],
                hovertemplate='<b>Point %{text}</b><br>' +
                            'PC1: %{x:.2f}<br>' +
                            'PC2: %{y:.2f}<br>' +
                            'PC3: %{z:.2f}<extra></extra>'
            )])
            
            fig.update_layout(
                title='🌐 Interactive 3D Data Exploration',
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=800,
                height=600,
                template='plotly_dark'
            )
            
            return fig
        
        return None
    
    def create_training_animation(self, ml_engine):
        """Create animated training progress visualization."""
        if not hasattr(ml_engine, 'training_history'):
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No training history available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('#2c3e50')
        
        # Animated training metrics
        history = ml_engine.training_history
        epochs = range(1, len(history.get('accuracy', [])) + 1)
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Training accuracy
            if 'accuracy' in history:
                current_acc = history['accuracy'][:frame+1]
                current_epochs = epochs[:frame+1]
                
                ax1.plot(current_epochs, current_acc, 
                        color=self.gamification_colors['success'], linewidth=3, marker='o')
                ax1.set_title('🎯 Training Accuracy Progress', fontweight='bold', color='#ecf0f1')
                ax1.set_xlabel('Epoch', color='#ecf0f1')
                ax1.set_ylabel('Accuracy', color='#ecf0f1')
                ax1.grid(True, alpha=0.3)
                ax1.set_facecolor('#34495e')
                
                # Add achievement markers
                if frame > 0 and current_acc[frame] > 0.9:
                    ax1.scatter(current_epochs[frame], current_acc[frame], 
                              s=200, color=self.gamification_colors['achievement'], 
                              marker='*', zorder=5)
                    ax1.text(current_epochs[frame], current_acc[frame] + 0.02, 
                           '🏆', ha='center', va='bottom', fontsize=20)
            
            # Training loss
            if 'loss' in history:
                current_loss = history['loss'][:frame+1]
                current_epochs = epochs[:frame+1]
                
                ax2.plot(current_epochs, current_loss, 
                        color=self.gamification_colors['error'], linewidth=3, marker='s')
                ax2.set_title('📉 Training Loss Reduction', fontweight='bold', color='#ecf0f1')
                ax2.set_xlabel('Epoch', color='#ecf0f1')
                ax2.set_ylabel('Loss', color='#ecf0f1')
                ax2.grid(True, alpha=0.3)
                ax2.set_facecolor('#34495e')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(epochs), 
                           interval=self.animation_speed, repeat=True)
        
        plt.tight_layout()
        return fig, anim
    
    def create_hyperparameter_landscape(self, ml_engine):
        """Create 3D hyperparameter optimization landscape."""
        if not hasattr(ml_engine, 'hyperparameter_results'):
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No hyperparameter optimization results available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('#2c3e50')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#34495e')
        
        # Get hyperparameter results
        results = ml_engine.hyperparameter_results
        
        # Create 3D surface plot
        if len(results) >= 3:
            # Example: plot first 3 hyperparameters vs performance
            param_names = list(results.keys())[:3]
            x = [results[param_names[0]][i] for i in range(len(results[param_names[0]]))]
            y = [results[param_names[1]][i] for i in range(len(results[param_names[1]]))]
            z = [results[param_names[2]][i] for i in range(len(results[param_names[2]]))]
            
            # Performance scores
            scores = results.get('scores', [0.5] * len(x))
            
            # Create 3D scatter
            scatter = ax.scatter(x, y, z, c=scores, cmap='viridis', 
                               s=100, alpha=0.8, edgecolors='white')
            
            ax.set_xlabel(param_names[0], color='#ecf0f1', fontweight='bold')
            ax.set_ylabel(param_names[1], color='#ecf0f1', fontweight='bold')
            ax.set_zlabel(param_names[2], color='#ecf0f1', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Performance Score', color='#ecf0f1', fontweight='bold')
            
            # Highlight best configuration
            best_idx = np.argmax(scores)
            ax.scatter([x[best_idx]], [y[best_idx]], [z[best_idx]], 
                      s=300, color='gold', marker='*', edgecolors='red', linewidth=2)
            ax.text(x[best_idx], y[best_idx], z[best_idx] + 0.1, '🏆 BEST!', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold', color='gold')
        
        ax.set_title('🏔️ Hyperparameter Optimization Landscape', 
                    fontsize=16, fontweight='bold', color='#ecf0f1', pad=20)
        
        plt.tight_layout()
        return fig
    
    def _add_achievement_banner(self, fig, achievements):
        """Add achievement banner to the top of the figure."""
        banner_text = "🏆 ACHIEVEMENTS: " + " | ".join(achievements[:3])
        fig.text(0.5, 0.95, banner_text, ha='center', va='top', 
                fontsize=12, fontweight='bold', color=self.gamification_colors['achievement'],
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#9b59b6', alpha=0.3))
    
    def _add_hover_effects(self, fig, axes):
        """Add hover effects to the visualization."""
        def on_hover(event):
            if event.inaxes:
                # Change cursor and add glow effect
                event.inaxes.set_facecolor('#3a4a5c')
                fig.canvas.draw_idle()
        
        def on_leave(event):
            if event.inaxes:
                # Reset appearance
                event.inaxes.set_facecolor('#34495e')
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('axes_enter_event', on_hover)
        fig.canvas.mpl_connect('axes_leave_event', on_leave)
    
    # Helper methods for calculations
    def _calculate_completeness(self, data):
        """Calculate data completeness percentage."""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        return ((total_cells - missing_cells) / total_cells) * 100
    
    def _calculate_enhanced_consistency(self, data):
        """Calculate enhanced data consistency score with multiple criteria."""
        consistency_scores = []
        
        # 1. Outlier-based consistency
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            outlier_consistency = self._calculate_outlier_consistency(numeric_data)
            consistency_scores.append(outlier_consistency)
        
        # 2. Pattern consistency for categorical data
        categorical_data = data.select_dtypes(include=['object'])
        if len(categorical_data.columns) > 0:
            pattern_consistency = self._calculate_pattern_consistency(categorical_data)
            consistency_scores.append(pattern_consistency)
        
        # 3. Distribution consistency
        if len(numeric_data.columns) > 0:
            distribution_consistency = self._calculate_distribution_consistency(numeric_data)
            consistency_scores.append(distribution_consistency)
        
        # Return weighted average or default
        if consistency_scores:
            return np.mean(consistency_scores)
        else:
            return 0.8  # Default for non-analyzable data
    
    def _calculate_outlier_consistency(self, numeric_data):
        """Calculate consistency based on outlier presence."""
        outlier_ratios = []
        
        for col in numeric_data.columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                outliers = numeric_data[col][(numeric_data[col] < (Q1 - 1.5 * IQR)) | 
                                           (numeric_data[col] > (Q3 + 1.5 * IQR))]
                outlier_ratio = len(outliers) / len(numeric_data[col])
                outlier_ratios.append(outlier_ratio)
        
        if outlier_ratios:
            avg_outlier_ratio = np.mean(outlier_ratios)
            return max(0, min(1, 1 - avg_outlier_ratio * 2))  # Scale and invert
        return 0.8
    
    def _calculate_pattern_consistency(self, categorical_data):
        """Calculate consistency of categorical patterns."""
        pattern_scores = []
        
        for col in categorical_data.columns:
            if categorical_data[col].dtype == 'object':
                # Check for consistent formatting patterns
                value_lengths = categorical_data[col].dropna().astype(str).str.len()
                length_cv = value_lengths.std() / value_lengths.mean() if value_lengths.mean() > 0 else 0
                
                # Lower coefficient of variation indicates more consistent patterns
                pattern_score = max(0, min(1, 1 - length_cv))
                pattern_scores.append(pattern_score)
        
        return np.mean(pattern_scores) if pattern_scores else 0.8
    
    def _calculate_distribution_consistency(self, numeric_data):
        """Calculate consistency of data distributions."""
        distribution_scores = []
        
        for col in numeric_data.columns:
            try:
                # Check for multi-modal distributions (less consistent)
                from scipy import stats
                data_clean = numeric_data[col].dropna()
                
                if len(data_clean) > 10:
                    # Use skewness and kurtosis as consistency indicators
                    skewness = abs(stats.skew(data_clean))
                    kurtosis = abs(stats.kurtosis(data_clean))
                    
                    # Normal-like distributions are more consistent
                    skew_score = max(0, 1 - skewness / 3)  # Normalize skewness
                    kurt_score = max(0, 1 - kurtosis / 5)  # Normalize kurtosis
                    
                    distribution_scores.append((skew_score + kurt_score) / 2)
            except:
                distribution_scores.append(0.7)  # Default for calculation failures
        
        return np.mean(distribution_scores) if distribution_scores else 0.8
    
    def _calculate_data_reliability(self, original_data, processed_data):
        """Calculate data reliability based on processing improvements."""
        try:
            # Compare data quality improvements
            original_missing = original_data.isnull().sum().sum()
            processed_missing = processed_data.isnull().sum().sum()
            
            # Calculate missing data improvement
            if original_missing > 0:
                missing_improvement = (original_missing - processed_missing) / original_missing
            else:
                missing_improvement = 1.0  # Perfect if no missing data initially
            
            # Calculate feature consistency
            feature_consistency = len(processed_data.columns) / len(original_data.columns)
            
            # Weighted reliability score
            reliability = (missing_improvement * 0.6 + feature_consistency * 0.4)
            return max(0, min(1, reliability))
            
        except:
            return 0.7  # Default reliability score
    
    def _get_quality_grade(self, score):
        """Convert quality score to letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 65:
            return "D+"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _calculate_accuracy(self, original_data, processed_data):
        """Calculate data accuracy improvement."""
        # Simplified accuracy calculation based on data quality improvement
        original_quality = self._calculate_completeness(original_data) / 100
        processed_quality = self._calculate_completeness(processed_data) / 100
        
        # Accuracy improves with processing
        accuracy = min(1.0, processed_quality + 0.1)
        return accuracy
    
    def _calculate_performance_score(self, original_data, processed_data):
        """Calculate overall performance score."""
        completeness = self._calculate_completeness(processed_data) / 100
        consistency = self._calculate_consistency(processed_data)
        accuracy = self._calculate_accuracy(original_data, processed_data)
        
        # Weight the metrics
        score = (completeness * 0.4 + consistency * 0.3 + accuracy * 0.3)
        return min(1.0, max(0.0, score))
    
    def _calculate_cleaning_progress(self, original_data, processed_data):
        """Calculate cleaning progress percentage."""
        original_missing = original_data.isnull().sum().sum()
        processed_missing = processed_data.isnull().sum().sum()
        
        if original_missing == 0:
            return 100  # No cleaning needed
        
        improvement = ((original_missing - processed_missing) / original_missing) * 100
        return max(0, min(100, improvement))
    
    def _calculate_optimization_progress(self, original_data, processed_data):
        """Calculate optimization progress percentage."""
        # Based on data quality improvements
        original_quality = self._calculate_completeness(original_data)
        processed_quality = self._calculate_completeness(processed_data)
        
        improvement = processed_quality - original_quality
        # Map improvement to 0-100 scale
        progress = max(0, min(100, improvement * 5))  # Scale factor
        return progress
    
    # Enhanced versions of existing methods
    def create_enhanced_data_overview(self, original_data, processed_data=None):
        """Enhanced version of the original data overview."""
        return self.create_gamified_data_overview(original_data, processed_data)
    
    def create_enhanced_cleaning_summary(self, data_processor):
        """Enhanced version of cleaning summary with gamification."""
        return self.create_data_cleaning_summary(data_processor)
    
    def create_interactive_before_after(self, original_data, cleaned_data):
        """Interactive before/after comparison."""
        if PLOTLY_AVAILABLE:
            return self._create_plotly_before_after(original_data, cleaned_data)
        else:
            return self.create_before_after_comparison(original_data, cleaned_data)
    
    def _create_plotly_before_after(self, original_data, cleaned_data):
        """Create interactive Plotly before/after comparison."""
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=['Original Data', 'Cleaned Data'],
                           specs=[[{"secondary_y": False}, {"secondary_y": False}]])
        
        # Original data metrics
        original_missing = original_data.isnull().sum()
        cleaned_missing = cleaned_data.isnull().sum()
        
        # Add bar charts
        fig.add_trace(
            go.Bar(x=original_missing.index[:10], y=original_missing.values[:10],
                  name='Original Missing', marker_color='#e74c3c'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=cleaned_missing.index[:10], y=cleaned_missing.values[:10],
                  name='Cleaned Missing', marker_color='#2ecc71'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="🔄 Interactive Before vs After Comparison",
            template='plotly_dark',
            showlegend=True
        )
        
        return fig
    
    # Keep all existing methods from original implementation
    def create_data_overview(self, original_data, processed_data=None):
        """Legacy method for backward compatibility."""
        return self.create_gamified_data_overview(original_data, processed_data)
    
    def create_data_cleaning_summary(self, data_processor):
        """Create visualization showing data cleaning results and AI decisions."""
        if not hasattr(data_processor, 'ai_engine') or not getattr(data_processor, 'column_analyses', None):
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No AI cleaning analysis available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🤖 AI Data Cleaning Summary & Decisions', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Cleaning Strategies Applied
        ax1 = axes[0, 0]
        self._create_cleaning_strategies_chart(ax1, data_processor)
        
        # 2. Column Priority Analysis
        ax2 = axes[0, 1]
        self._create_priority_analysis(ax2, data_processor)
        
        # 3. AI Decisions Timeline
        ax3 = axes[1, 0]
        self._create_ai_decisions_timeline(ax3, data_processor)
        
        # 4. Data Quality Improvement
        ax4 = axes[1, 1]
        self._create_quality_improvement_chart(ax4, data_processor)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig