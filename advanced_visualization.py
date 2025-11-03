"""
Advanced Multi-Dimensional Visualization Dashboard
Provides comprehensive 2D and 3D visualizations with unified controls.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import threading
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class AdvancedVisualizationDashboard:
    """
    Comprehensive visualization dashboard with 2D/3D support and unified controls.
    """
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.current_data = None
        self.visualization_mode = "2D"  # "2D" or "3D"
        self.color_palette = plt.cm.Set1
        self.theme = "professional"
        
        # Visualization containers
        self.canvases = {}
        self.figures = {}
        
        # Setup styling
        self.setup_styling()
        
        # Create dashboard
        self.create_dashboard_ui()
        
    def setup_styling(self):
        """Setup professional styling for visualizations."""
        # Professional color scheme
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'figure.facecolor': self.colors['primary'],
            'axes.facecolor': '#34495e',
            'axes.edgecolor': '#bdc3c7',
            'axes.labelcolor': '#ecf0f1',
            'text.color': '#ecf0f1',
            'xtick.color': '#bdc3c7',
            'ytick.color': '#bdc3c7',
            'grid.color': '#7f8c8d',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
        
    def create_dashboard_ui(self):
        """Create the dashboard user interface."""
        # Main container
        self.main_container = ttk.Frame(self.parent_frame)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel
        self.create_control_panel()
        
        # Visualization notebook
        self.viz_notebook = ttk.Notebook(self.main_container)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create visualization tabs
        self.create_visualization_tabs()
        
    def create_control_panel(self):
        """Create unified visualization controls."""
        control_frame = ttk.LabelFrame(self.main_container, text="Visualization Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # First row - Mode and theme controls
        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        # 2D/3D Mode toggle
        ttk.Label(row1, text="Mode:").pack(side=tk.LEFT, padx=(0, 5))
        self.mode_var = tk.StringVar(value="2D")
        mode_frame = ttk.Frame(row1)
        mode_frame.pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(mode_frame, text="2D", variable=self.mode_var, value="2D", 
                       command=self.on_mode_change).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(mode_frame, text="3D", variable=self.mode_var, value="3D", 
                       command=self.on_mode_change).pack(side=tk.LEFT)
        
        # Theme selector
        ttk.Label(row1, text="Theme:").pack(side=tk.LEFT, padx=(0, 5))
        self.theme_var = tk.StringVar(value="professional")
        theme_combo = ttk.Combobox(row1, textvariable=self.theme_var, width=15,
                                  values=["professional", "dark", "light", "colorful"])
        theme_combo.pack(side=tk.LEFT, padx=(0, 20))
        theme_combo.bind('<<ComboboxSelected>>', self.on_theme_change)
        
        # Refresh button
        ttk.Button(row1, text="[REFRESH] All", command=self.refresh_all_visualizations).pack(side=tk.RIGHT)
        
        # Second row - Specific controls
        row2 = ttk.Frame(control_frame) 
        row2.pack(fill=tk.X)
        
        # Color palette selector
        ttk.Label(row2, text="Colors:").pack(side=tk.LEFT, padx=(0, 5))
        self.palette_var = tk.StringVar(value="Set1")
        palette_combo = ttk.Combobox(row2, textvariable=self.palette_var, width=15,
                                   values=["Set1", "Set2", "tab10", "viridis", "plasma", "husl"])
        palette_combo.pack(side=tk.LEFT, padx=(0, 20))
        palette_combo.bind('<<ComboboxSelected>>', self.on_palette_change)
        
        # Export controls
        export_frame = ttk.Frame(row2)
        export_frame.pack(side=tk.RIGHT)
        ttk.Button(export_frame, text="[EXPORT] Current", command=self.export_current_viz).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="[EXPORT] All", command=self.export_all_viz).pack(side=tk.LEFT)
        
    def create_visualization_tabs(self):
        """Create all visualization tabs."""
        # Overview Tab
        self.overview_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.overview_frame, text="[DATA] Overview")
        
        # Distribution Tab
        self.distribution_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.distribution_frame, text="[CHART] Distributions")
        
        # Correlation Tab
        self.correlation_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.correlation_frame, text="[MATRIX] Correlations")
        
        # 3D Analysis Tab
        self.analysis_3d_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.analysis_3d_frame, text="[3D] Analysis")
        
        # Model Performance Tab
        self.performance_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.performance_frame, text="[ML] Performance")
        
        # Advanced Analytics Tab
        self.advanced_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.advanced_frame, text="[ADVANCED] Analytics")
        
    def update_data(self, data: pd.DataFrame):
        """Update data and refresh all visualizations."""
        self.current_data = data
        self.refresh_all_visualizations()
        
    def refresh_all_visualizations(self):
        """Refresh all visualizations with current data."""
        if self.current_data is None:
            return
            
        try:
            # Clear existing visualizations
            self.clear_all_canvases()
            
            # Generate new visualizations based on mode
            if self.visualization_mode == "2D":
                self.generate_2d_visualizations()
            else:
                self.generate_3d_visualizations()
                
        except Exception as e:
            print(f"Error refreshing visualizations: {e}")
            
    def clear_all_canvases(self):
        """Clear all existing canvas widgets."""
        for canvas in self.canvases.values():
            if canvas and hasattr(canvas, 'get_tk_widget'):
                canvas.get_tk_widget().destroy()
        self.canvases.clear()
        self.figures.clear()
        
    def generate_2d_visualizations(self):
        """Generate comprehensive 2D visualizations."""
        # Data Overview
        self.create_data_overview_2d()
        
        # Distribution Analysis
        self.create_distribution_analysis_2d()
        
        # Correlation Analysis
        self.create_correlation_analysis_2d()
        
        # Advanced Analytics
        self.create_advanced_analytics_2d()
        
    def generate_3d_visualizations(self):
        """Generate comprehensive 3D visualizations."""
        # 3D Scatter plots
        self.create_3d_scatter_analysis()
        
        # 3D Surface plots
        self.create_3d_surface_analysis()
        
        # 3D Volume visualization
        self.create_3d_volume_analysis()
        
    def create_data_overview_2d(self):
        """Create 2D data overview visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(self.colors['primary'])
        
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Box plot
            self.current_data[numeric_cols[:4]].boxplot(ax=ax1)
            ax1.set_title('Distribution Summary', color='white', fontsize=12)
            ax1.tick_params(colors='white')
            
            # Line plot of first numeric column
            if len(numeric_cols) > 0:
                ax2.plot(self.current_data[numeric_cols[0]], color=self.colors['secondary'], linewidth=2)
                ax2.set_title(f'Trend: {numeric_cols[0]}', color='white', fontsize=12)
                ax2.tick_params(colors='white')
                
            # Histogram
            if len(numeric_cols) > 1:
                ax3.hist(self.current_data[numeric_cols[1]], bins=30, alpha=0.7, 
                        color=self.colors['success'], edgecolor='white')
                ax3.set_title(f'Distribution: {numeric_cols[1]}', color='white', fontsize=12)
                ax3.tick_params(colors='white')
                
            # Scatter plot
            if len(numeric_cols) > 2:
                ax4.scatter(self.current_data[numeric_cols[0]], self.current_data[numeric_cols[2]], 
                           alpha=0.6, c=self.colors['warning'])
                ax4.set_title(f'{numeric_cols[0]} vs {numeric_cols[2]}', color='white', fontsize=12)
                ax4.tick_params(colors='white')
                
        plt.tight_layout()
        
        # Add to canvas
        canvas = FigureCanvasTkAgg(fig, self.overview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['overview_2d'] = canvas
        self.figures['overview_2d'] = fig
        
    def create_distribution_analysis_2d(self):
        """Create 2D distribution analysis."""
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
        n_cols = min(len(numeric_cols), 6)
        
        if n_cols == 0:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor(self.colors['primary'])
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:6]):
            if i < len(axes):
                # Histogram with KDE
                axes[i].hist(self.current_data[col], bins=30, alpha=0.7, density=True,
                           color=plt.cm.Set1(i/10), edgecolor='white')
                
                # Add KDE curve
                try:
                    from scipy import stats
                    x = np.linspace(self.current_data[col].min(), self.current_data[col].max(), 100)
                    kde = stats.gaussian_kde(self.current_data[col].dropna())
                    axes[i].plot(x, kde(x), color='red', linewidth=2)
                except:
                    pass
                    
                axes[i].set_title(f'Distribution: {col}', color='white', fontsize=10)
                axes[i].tick_params(colors='white')
                
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.distribution_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['distribution_2d'] = canvas
        self.figures['distribution_2d'] = fig
        
    def create_correlation_analysis_2d(self):
        """Create 2D correlation analysis."""
        numeric_data = self.current_data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor(self.colors['primary'])
        
        # Correlation heatmap
        corr_matrix = numeric_data.corr()
        im1 = ax1.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax1.set_title('Correlation Matrix', color='white', fontsize=12)
        ax1.set_xticks(range(len(corr_matrix.columns)))
        ax1.set_yticks(range(len(corr_matrix.columns)))
        ax1.set_xticklabels(corr_matrix.columns, rotation=45, color='white')
        ax1.set_yticklabels(corr_matrix.columns, color='white')
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white', fontsize=8)
                        
        # Clustermap-style dendrogram (simplified)
        if len(numeric_data.columns) > 2:
            # Pairplot for top correlated features
            top_cols = numeric_data.columns[:4]  # Take first 4 columns
            scatter_data = numeric_data[top_cols]
            
            # Create scatter matrix
            n_features = len(top_cols)
            colors_list = [self.colors['secondary'], self.colors['success'], 
                          self.colors['warning'], self.colors['danger']]
            
            for i, col1 in enumerate(top_cols):
                for j, col2 in enumerate(top_cols):
                    if i != j and i < 2 and j < 2:  # Simplified 2x2 scatter
                        ax2.scatter(scatter_data[col1], scatter_data[col2], 
                                  alpha=0.6, c=colors_list[i], s=20)
                        
            ax2.set_title('Feature Relationships', color='white', fontsize=12)
            ax2.tick_params(colors='white')
            
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.correlation_frame)
        canvas.draw()  
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['correlation_2d'] = canvas
        self.figures['correlation_2d'] = fig
        
    def create_advanced_analytics_2d(self):
        """Create advanced 2D analytics visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(self.colors['primary'])
        
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Time series-like plot
            ax1.plot(range(len(self.current_data)), self.current_data[numeric_cols[0]], 
                    color=self.colors['secondary'], alpha=0.8)
            ax1.fill_between(range(len(self.current_data)), self.current_data[numeric_cols[0]], 
                           alpha=0.3, color=self.colors['secondary'])
            ax1.set_title(f'Trend Analysis: {numeric_cols[0]}', color='white')
            ax1.tick_params(colors='white')
            
            # Violin plot
            if len(numeric_cols) > 1:
                data_to_plot = [self.current_data[col].dropna() for col in numeric_cols[:3]]
                parts = ax2.violinplot(data_to_plot, showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(self.colors['success'])
                    pc.set_alpha(0.7)
                ax2.set_title('Distribution Shapes', color='white')
                ax2.tick_params(colors='white')
                
            # Residual plot (if we have 2+ numeric columns)
            if len(numeric_cols) > 1:
                x, y = self.current_data[numeric_cols[0]], self.current_data[numeric_cols[1]]
                # Simple linear fit
                coeffs = np.polyfit(x.dropna(), y.dropna(), 1)
                fitted = np.polyval(coeffs, x)
                residuals = y - fitted
                
                ax3.scatter(fitted, residuals, alpha=0.6, color=self.colors['warning'])
                ax3.axhline(y=0, color='red', linestyle='--')
                ax3.set_title('Residual Analysis', color='white')
                ax3.tick_params(colors='white')
                
            # Statistical summary plot
            # Statistical summary plot
        numeric_data = self.current_data.select_dtypes(include=[np.number])
        stats_data = numeric_data.describe().T
        x_pos = range(len(stats_data))
        ax4.bar(x_pos, stats_data['mean'], alpha=0.7, color=self.colors['info'])
        ax4.set_title('Statistical Summary', color='white')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(stats_data.index, rotation=45, color='white')
        ax4.tick_params(colors='white')
            
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.advanced_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['advanced_2d'] = canvas
        self.figures['advanced_2d'] = fig
        
    def create_3d_scatter_analysis(self):
        """Create 3D scatter plot analysis."""
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor(self.colors['primary'])
        
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 3:
            ax = fig.add_subplot(121, projection='3d')
            
            # 3D scatter plot
            x, y, z = (self.current_data[numeric_cols[0]], 
                      self.current_data[numeric_cols[1]], 
                      self.current_data[numeric_cols[2]])
                      
            scatter = ax.scatter(x, y, z, c=range(len(x)), cmap='viridis', alpha=0.6, s=50)
            
            ax.set_xlabel(numeric_cols[0], color='white')
            ax.set_ylabel(numeric_cols[1], color='white') 
            ax.set_zlabel(numeric_cols[2], color='white')
            ax.set_title('3D Feature Space', color='white', fontsize=14)
            
            # Customize 3D plot appearance
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, alpha=0.3)
            
        # Second 3D plot - surface or wireframe
        if len(numeric_cols) >= 2:
            ax2 = fig.add_subplot(122, projection='3d')
            
            # Create a surface based on data density
            x_range = np.linspace(self.current_data[numeric_cols[0]].min(), 
                                 self.current_data[numeric_cols[0]].max(), 20)
            y_range = np.linspace(self.current_data[numeric_cols[1]].min(), 
                                 self.current_data[numeric_cols[1]].max(), 20)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Simple function for demonstration
            Z = np.sin(X/X.max()) * np.cos(Y/Y.max()) * (X.max() + Y.max()) / 2
            
            surf = ax2.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, antialiased=True)
            ax2.set_title('3D Surface Analysis', color='white', fontsize=14)
            ax2.set_xlabel(numeric_cols[0], color='white')
            ax2.set_ylabel(numeric_cols[1], color='white')
            
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.analysis_3d_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['3d_scatter'] = canvas
        self.figures['3d_scatter'] = fig
        
    def create_3d_surface_analysis(self):
        """Create additional 3D surface visualizations."""
        # This would be added to the 3D analysis tab as well
        pass
        
    def create_3d_volume_analysis(self):
        """Create 3D volume visualization if applicable.""" 
        # This would create volume renderings for applicable data
        pass
        
    def on_mode_change(self):
        """Handle visualization mode change."""
        self.visualization_mode = self.mode_var.get()
        self.refresh_all_visualizations()
        
    def on_theme_change(self, event=None):
        """Handle theme change."""
        self.theme = self.theme_var.get()
        self.update_theme()
        self.refresh_all_visualizations()
        
    def on_palette_change(self, event=None):
        """Handle color palette change."""
        palette_name = self.palette_var.get()
        try:
            self.color_palette = getattr(plt.cm, palette_name)
        except:
            self.color_palette = plt.cm.Set1
        self.refresh_all_visualizations()
        
    def update_theme(self):
        """Update visualization theme."""
        if self.theme == "dark":
            self.colors.update({
                'primary': '#1a1a1a',
                'secondary': '#4a90e2',
            })
        elif self.theme == "light":
            self.colors.update({
                'primary': '#ffffff',
                'secondary': '#007bff',
            })
        elif self.theme == "colorful":
            self.colors.update({
                'primary': '#2c3e50',
                'secondary': '#e74c3c',
            })
            
        # Update matplotlib theme
        plt.rcParams.update({
            'figure.facecolor': self.colors['primary'],
            'axes.facecolor': self.colors['primary'] if self.theme != 'light' else '#f8f9fa'
        })
        
    def export_current_viz(self):
        """Export currently selected visualization."""
        from pathlib import Path
        
        # Create export directory
        export_dir = Path("ai_analytics_storage/visualizations/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        current_tab = self.viz_notebook.select()
        tab_text = self.viz_notebook.tab(current_tab, "text")
        
        # Get timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean tab text for filename
        clean_tab_text = tab_text.replace("[", "").replace("]", "").replace(" ", "_")
        filename = export_dir / f"visualization_{clean_tab_text}_{timestamp}.png"
        
        try:
            # Find the corresponding figure
            for key, fig in self.figures.items():
                if key in tab_text.lower():
                    fig.savefig(filename, dpi=300, bbox_inches='tight', 
                              facecolor=self.colors['primary'])
                    print(f"✅ Exported: {filename}")
                    break
        except Exception as e:
            print(f"❌ Export failed: {e}")
            
    def export_all_viz(self):
        """Export all visualizations to separate folder."""
        from pathlib import Path
        
        # Create export directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(f"ai_analytics_storage/visualizations/exports/batch_{timestamp}")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        failed_count = 0
        
        for name, fig in self.figures.items():
            try:
                # Clean name for filename
                clean_name = name.replace(" ", "_").replace("/", "_")
                filename = export_dir / f"viz_{clean_name}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight',
                          facecolor=self.colors['primary'])
                print(f"✅ Exported: {filename}")
                exported_count += 1
            except Exception as e:
                print(f"❌ Failed to export {name}: {e}")
                failed_count += 1
        
        # Summary message
        print(f"\n📊 Export Summary:")
        print(f"   ✅ Successfully exported: {exported_count} visualizations")
        if failed_count > 0:
            print(f"   ❌ Failed: {failed_count} visualizations")
        print(f"   📁 Location: {export_dir.absolute()}")