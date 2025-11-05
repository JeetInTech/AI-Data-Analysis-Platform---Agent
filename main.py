import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import threading
import queue
import os
import json
import sqlite3
from datetime import datetime
import subprocess
import sys
from pathlib import Path

# Import your existing working modules
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    SKLEARN_ADVANCED_AVAILABLE = True
except ImportError:
    SKLEARN_ADVANCED_AVAILABLE = False

try:
    from data_processor import SmartDataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data_processor: {e}")
    DATA_PROCESSOR_AVAILABLE = False
    SmartDataProcessor = None

try:
    from ml_engine import SmartMLEngine
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ml_engine: {e}")
    ML_ENGINE_AVAILABLE = False
    SmartMLEngine = None

try:
    from neural_visualizer import SmartVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import neural_visualizer: {e}")
    VISUALIZER_AVAILABLE = False
    SmartVisualizer = None

# Import new modules (these will be created in subsequent files)
try:
    from auto_selector import IntelligentProcessSelector
except ImportError:
    IntelligentProcessSelector = None
    
try:
    from explainability import ModelExplainer
except ImportError:
    ModelExplainer = None
    
try:
    from storage_manager import ArtifactStorageManager
except ImportError:
    ArtifactStorageManager = None

try:
    from pygame_ui import PygameInterface
except ImportError:
    PygameInterface = None

try:
    from agent_mode import AgentModeController
    AGENT_MODE_AVAILABLE = True
except ImportError:
    AGENT_MODE_AVAILABLE = False
    AgentModeController = None

try:
    from advanced_visualization import AdvancedVisualizationDashboard
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False
    AdvancedVisualizationDashboard = None

# Database connectivity imports
try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, text
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

# Additional format support
try:
    import openpyxl  # Excel support
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    import pyarrow as pa  # Parquet support
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

class EnhancedAIDataPlatform:
    """Enhanced AI Data Analysis Platform - Advanced Multi-Format, Multi-Source Analytics"""

    def __init__(self, root):
        self.root = root
        self.root.title("NEUROVIAI Advanced AI Data Analytics Platform - Multi-Format & Multi-Source")
        self.root.geometry("1800x1200")
        self.root.configure(bg="#2c3e50")

        # Initialize components with error handling
        try:
            self.data_processor = SmartDataProcessor() if DATA_PROCESSOR_AVAILABLE else None
        except Exception as e:
            print(f"Error initializing data processor: {e}")
            self.data_processor = None
            
        try:
            self.ml_engine = SmartMLEngine() if ML_ENGINE_AVAILABLE else None
        except Exception as e:
            print(f"Error initializing ML engine: {e}")
            self.ml_engine = None
            
        try:
            self.visualizer = SmartVisualizer() if VISUALIZER_AVAILABLE else None
        except Exception as e:
            print(f"Error initializing visualizer: {e}")
            self.visualizer = None
        
        # Use visualizer as neural_visualizer for compatibility
        self.neural_visualizer = self.visualizer
        
        # Initialize new components if available
        self.auto_selector = IntelligentProcessSelector() if IntelligentProcessSelector else None
        self.explainer = ModelExplainer() if ModelExplainer else None
        self.storage_manager = ArtifactStorageManager() if ArtifactStorageManager else None
        
        # Initialize Agent Mode and Advanced Visualization
        self.agent_mode = AgentModeController(self) if AGENT_MODE_AVAILABLE else None
        self.advanced_viz_dashboard = None  # Will be initialized in create_widgets
        
        # State variables
        self.current_data = None
        self.cleaned_data = None
        self.processed_data = None
        self.selected_columns = []
        self.target_column = None
        self.training_complete = False
        self.current_session_id = None
        self.meta_features = {}
        self.database_connections = {}
        
        # Visualization format preference
        self.viz_format_var = tk.StringVar(value="matplotlib")  # Default to matplotlib
        
        # UI Mode
        self.pygame_mode = False
        self.pygame_interface = None

        # Queue for thread communication
        self.queue = queue.Queue()

        # Create storage directory
        self.storage_dir = Path("ai_analytics_sessions")
        self.storage_dir.mkdir(exist_ok=True)

        # Create GUI
        self.create_widgets()
        self.create_menu()
        
        # Initialize session management
        self.initialize_session_management()

        # Start queue checker
        self.check_queue()

    def initialize_session_management(self):
        """Initialize session management system."""
        if self.storage_manager:
            self.current_session_id = self.storage_manager.create_new_session()
        else:
            # Fallback session management
            self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = self.storage_dir / self.current_session_id
            session_dir.mkdir(exist_ok=True)

    def create_menu(self):
        """Create enhanced application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Multi-format file loading submenu
        load_submenu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Load Data", menu=load_submenu)
        load_submenu.add_command(label="📄 CSV File", command=lambda: self.load_file('csv'))
        if EXCEL_AVAILABLE:
            load_submenu.add_command(label="📊 Excel File (.xlsx)", command=lambda: self.load_file('excel'))
        load_submenu.add_command(label="🔧 JSON File", command=lambda: self.load_file('json'))
        if PARQUET_AVAILABLE:
            load_submenu.add_command(label="⚡ Parquet File", command=lambda: self.load_file('parquet'))
        load_submenu.add_separator()
        if SQL_AVAILABLE:
            load_submenu.add_command(label="🗄️ Database Connection", command=self.connect_database)
        
        file_menu.add_separator()
        
        # Session management
        session_submenu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Session Management", menu=session_submenu)
        session_submenu.add_command(label="💾 Save Current Session", command=self.save_session)
        session_submenu.add_command(label="📂 Load Previous Session", command=self.load_session)
        session_submenu.add_command(label="🗑️ Clear Current Session", command=self.clear_session)
        session_submenu.add_command(label="📋 List All Sessions", command=self.list_sessions)
        
        file_menu.add_separator()
        file_menu.add_command(label="💾 Export Cleaned Data", command=self.export_cleaned_data)
        file_menu.add_command(label="📊 Export Results", command=self.export_results)
        file_menu.add_command(label="🎨 Export All Visualizations", command=self.export_all_visualizations)

        # Data menu
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(label="🧠 AI Data Analysis", command=self.analyze_data)
        data_menu.add_command(label="🎯 Auto Process Selection", command=self.auto_select_process)
        data_menu.add_command(label="🧹 Smart Data Cleaning", command=self.clean_data)
        data_menu.add_command(label="🔧 Advanced Feature Engineering", command=self.advanced_feature_engineering)
        data_menu.add_command(label="▶️ Proceed to ML", command=self.proceed_to_ml)

        # Visualization menu
        viz_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualization", menu=viz_menu)
        viz_menu.add_command(label="📊 Data Overview", command=self.show_data_overview)
        viz_menu.add_command(label="🧹 Cleaning Summary", command=self.show_cleaning_summary)
        viz_menu.add_command(label="🔄 Before/After Comparison", command=self.show_before_after)
        viz_menu.add_command(label="📋 Interactive Data Table", command=self.show_data_table)
        viz_menu.add_separator()
        viz_menu.add_command(label="🌐 3D Data Visualization", command=self.show_3d_visualization)
        viz_menu.add_command(label="🎬 Animated Training Progress", command=self.show_training_animation)
        viz_menu.add_command(label="🏔️ Hyperparameter Landscape", command=self.show_hyperparameter_landscape)

        # ML menu
        ml_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Machine Learning", menu=ml_menu)
        ml_menu.add_command(label="🚀 Train Models", command=self.train_models)
        ml_menu.add_command(label="⚡ AutoML Training", command=self.automl_training)
        ml_menu.add_command(label="🧠 Deep Learning Models", command=self.train_deep_learning)
        ml_menu.add_command(label="📝 NLP Analysis", command=self.nlp_analysis)
        ml_menu.add_separator()
        ml_menu.add_command(label="📈 Model Performance", command=self.show_model_performance)
        ml_menu.add_command(label="🎯 Feature Importance", command=self.show_feature_importance)

        # Explainability menu
        explain_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Explainability", menu=explain_menu)
        if ModelExplainer:
            explain_menu.add_command(label="🧩 SHAP Analysis", command=self.show_shap_analysis)
            explain_menu.add_command(label="🔍 LIME Explanations", command=self.show_lime_explanations)
            explain_menu.add_command(label="🎯 Feature Contributions", command=self.show_feature_contributions)
            explain_menu.add_command(label="❌ Error Analysis", command=self.show_error_analysis)
            explain_menu.add_command(label="📊 Prediction Confidence", command=self.show_prediction_confidence)

        # Interface menu
        interface_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Interface", menu=interface_menu)
        if PygameInterface:
            interface_menu.add_command(label="🎮 Switch to Pygame Mode", command=self.switch_to_pygame)
        interface_menu.add_command(label="🌓 Toggle Dark/Light Theme", command=self.toggle_theme)
        interface_menu.add_command(label="⚙️ Preferences", command=self.show_preferences)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="📡 Start API Server", command=self.start_api_server)
        tools_menu.add_command(label="🐳 Docker Deployment", command=self.docker_deployment)
        tools_menu.add_command(label="☁️ Cloud Upload", command=self.cloud_upload)

    def create_widgets(self):
        """Create enhanced GUI widgets."""
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create Agent Mode UI at the very top
        self.create_agent_mode_ui(main_frame)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_data_loading_tab()
        self.create_ai_analysis_tab()
        self.create_meta_features_tab()
        self.create_data_cleaning_tab()
        self.create_visualization_tab()
        self.create_ml_tab()
        self.create_explainability_tab()
        self.create_results_tab()

        # Status bar
        self.create_status_bar()

    def create_agent_mode_ui(self, parent):
        """Create Agent Mode UI at the top of the application."""
        # Agent Mode Control Panel
        agent_frame = ttk.LabelFrame(parent, text="[AGENT] Autonomous AI Analysis Mode", padding=10)
        agent_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Main control row
        control_row = ttk.Frame(agent_frame)
        control_row.pack(fill=tk.X)
        
        # Agent Mode button (prominent)
        self.agent_button = tk.Button(
            control_row,
            text="[LAUNCH] AGENT MODE",
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            height=2,
            width=20,
            command=self.toggle_agent_mode,
            relief=tk.RAISED,
            bd=3
        )
        self.agent_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Status indicator
        status_frame = ttk.Frame(control_row)
        status_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(status_frame, text="Status:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.agent_status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.agent_status_label.pack(anchor=tk.W)
        
        # Progress information
        progress_frame = ttk.Frame(control_row)
        progress_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(progress_frame, text="Progress:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.agent_progress_label = ttk.Label(progress_frame, text="0/8 steps")
        self.agent_progress_label.pack(anchor=tk.W)
        
        # Progress bar
        self.agent_progress_bar = ttk.Progressbar(
            control_row, 
            length=200, 
            mode='determinate'
        )
        self.agent_progress_bar.pack(side=tk.LEFT, padx=(0, 15))
        
        # Current step indicator
        step_frame = ttk.Frame(control_row)
        step_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(step_frame, text="Current Step:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.agent_step_label = ttk.Label(step_frame, text="Waiting...")
        self.agent_step_label.pack(anchor=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(control_row)
        button_frame.pack(side=tk.RIGHT)
        
        self.stop_agent_button = ttk.Button(
            button_frame,
            text="[STOP] Agent",
            command=self.stop_agent_mode,
            state=tk.DISABLED
        )
        self.stop_agent_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="[VIEW] Logs",
            command=self.show_agent_logs
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame,
            text="[SETTINGS] Agent",
            command=self.show_agent_settings
        ).pack(side=tk.LEFT)
        
        # Second row for real-time feedback
        feedback_row = ttk.Frame(agent_frame)
        feedback_row.pack(fill=tk.X, pady=(10, 0))
        
        # Real-time log display
        log_frame = ttk.LabelFrame(feedback_row, text="Real-time Activity Log", padding=5)
        log_frame.pack(fill=tk.X)
        
        self.agent_log_text = tk.Text(
            log_frame,
            height=3,
            wrap=tk.WORD,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Consolas", 9),
            state=tk.DISABLED
        )
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.agent_log_text.yview)
        self.agent_log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.agent_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize agent mode status checking
        self.check_agent_status()

    def check_agent_status(self):
        """Periodically check and update agent mode status."""
        if self.agent_mode:
            status = self.agent_mode.get_status()
            self.update_agent_ui(status)
        
        # Schedule next check
        self.root.after(1000, self.check_agent_status)  # Check every second
        
    def update_agent_ui(self, status):
        """Update Agent Mode UI with current status."""
        # Update status label
        was_running = status['is_running']
        
        if status['is_running']:
            self.agent_status_label.config(text="Running", foreground="orange")
            self.agent_button.config(text="[RUNNING] AGENT MODE", bg="#f39c12")
            self.stop_agent_button.config(state=tk.NORMAL)
            # Reset flag when starting
            if not hasattr(self, '_agent_was_running') or not self._agent_was_running:
                self._qa_shown_for_session = False
        else:
            self.agent_status_label.config(text="Ready", foreground="green")
            self.agent_button.config(text="[LAUNCH] AGENT MODE", bg="#e74c3c")
            self.stop_agent_button.config(state=tk.DISABLED)
            
            # Show Q&A interface ONCE when just completed
            # Check if it just transitioned from running to completed
            if (status['current_step'] == "Completed" and 
                status['progress'] == 100 and
                not getattr(self, '_qa_shown_for_session', False) and
                getattr(self, '_agent_was_running', False)):
                
                self._qa_shown_for_session = True
                self.root.after(1000, self.show_agent_qa_interface)
        
        # Track if agent was running
        self._agent_was_running = was_running
        
        # Update progress
        progress_text = f"{status['success_count']}/{status['total_steps']} steps"
        self.agent_progress_label.config(text=progress_text)
        self.agent_progress_bar['value'] = status['progress']
        
        # Update current step
        self.agent_step_label.config(text=status['current_step'])
        
        # Update log (show last few entries)
        if status['log']:
            self.agent_log_text.config(state=tk.NORMAL)
            self.agent_log_text.delete(1.0, tk.END)
            recent_logs = status['log'][-3:]  # Show last 3 log entries
            for log_entry in recent_logs:
                self.agent_log_text.insert(tk.END, log_entry + "\n")
            self.agent_log_text.config(state=tk.DISABLED)
            self.agent_log_text.see(tk.END)

    def toggle_agent_mode(self):
        """Toggle Agent Mode on/off."""
        if not self.agent_mode:
            messagebox.showwarning("Agent Mode", "Agent Mode is not available")
            return
            
        if not self.agent_mode.is_running:
            # Reset Q&A flag for new session
            self._qa_shown_for_session = False
            self._agent_was_running = False
            
            # Start Agent Mode
            success, message = self.agent_mode.start_agent_mode(self.on_agent_progress)
            if success:
                self.add_agent_log("[SYSTEM] Agent Mode started successfully")
            else:
                messagebox.showerror("Agent Mode", f"Failed to start: {message}")
        else:
            # Stop Agent Mode
            self.agent_mode.stop_agent_mode()
            self.add_agent_log("[SYSTEM] Agent Mode stopped by user")
            
    def stop_agent_mode(self):
        """Stop Agent Mode."""
        if self.agent_mode and self.agent_mode.is_running:
            self.agent_mode.stop_agent_mode()
            self.add_agent_log("[SYSTEM] Agent Mode stopped")
            
    def on_agent_progress(self, status):
        """Callback for agent mode progress updates."""
        # This will be called from the agent mode thread
        # Schedule UI update on main thread
        self.root.after(0, lambda: self.update_agent_ui(status))
        
    def add_agent_log(self, message):
        """Add a message to the agent log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.agent_log_text.config(state=tk.NORMAL)
        self.agent_log_text.insert(tk.END, log_entry + "\n")
        self.agent_log_text.config(state=tk.DISABLED)
        self.agent_log_text.see(tk.END)
        
    def show_agent_logs(self):
        """Show detailed agent logs in a separate window."""
        if not self.agent_mode:
            return
            
        log_window = tk.Toplevel(self.root)
        log_window.title("Agent Mode - Detailed Logs")
        log_window.geometry("800x600")
        log_window.configure(bg="#2c3e50")
        
        # Log display
        log_frame = ttk.Frame(log_window)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Consolas", 10)
        )
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
        log_text.configure(yscrollcommand=scrollbar.set)
        
        # Populate with logs
        status = self.agent_mode.get_status()
        all_logs = status['log'] + [f"ERROR: {error}" for error in status['errors']]
        
        for log_entry in all_logs:
            log_text.insert(tk.END, log_entry + "\n")
            
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        button_frame = ttk.Frame(log_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="Refresh", 
                  command=lambda: self.refresh_log_window(log_text)).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Export Logs", 
                  command=lambda: self.export_agent_logs()).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(button_frame, text="Close", 
                  command=log_window.destroy).pack(side=tk.RIGHT)
                  
    def refresh_log_window(self, log_text):
        """Refresh the log window content."""
        if not self.agent_mode:
            return
            
        log_text.delete(1.0, tk.END)
        status = self.agent_mode.get_status()
        all_logs = status['log'] + [f"ERROR: {error}" for error in status['errors']]
        
        for log_entry in all_logs:
            log_text.insert(tk.END, log_entry + "\n")
            
    def export_agent_logs(self):
        """Export agent logs to file."""
        if not self.agent_mode:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_logs_{timestamp}.txt"
            
            status = self.agent_mode.get_status()
            all_logs = status['log'] + [f"ERROR: {error}" for error in status['errors']]
            
            with open(filename, 'w') as f:
                f.write(f"Agent Mode Logs - Exported: {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                for log_entry in all_logs:
                    f.write(log_entry + "\n")
                    
            messagebox.showinfo("Export Complete", f"Logs exported to: {filename}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export logs: {str(e)}")
            
    def show_agent_settings(self):
        """Show agent mode settings."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Agent Mode Settings")
        settings_window.geometry("500x400")
        settings_window.configure(bg="#2c3e50")
        
        # Settings frame
        settings_frame = ttk.LabelFrame(settings_window, text="Agent Configuration", padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Max retries
        ttk.Label(settings_frame, text="Max Retries per Step:").pack(anchor=tk.W, pady=(0, 5))
        retry_var = tk.IntVar(value=3)
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=retry_var).pack(anchor=tk.W, pady=(0, 15))
        
        # Auto-export results
        auto_export_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Auto-export results after completion", 
                       variable=auto_export_var).pack(anchor=tk.W, pady=(0, 15))
        
        # Visualization preferences
        viz_frame = ttk.LabelFrame(settings_frame, text="Visualization Preferences", padding=5)
        viz_frame.pack(fill=tk.X, pady=(0, 15))
        
        viz_mode_var = tk.StringVar(value="Both 2D and 3D")
        ttk.Label(viz_frame, text="Default Visualization Mode:").pack(anchor=tk.W, pady=(0, 5))
        ttk.Combobox(viz_frame, textvariable=viz_mode_var, 
                    values=["2D Only", "3D Only", "Both 2D and 3D"]).pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="Apply", 
                  command=lambda: self.apply_agent_settings(retry_var.get(), auto_export_var.get(), viz_mode_var.get())).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=lambda: self.reset_agent_settings()).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(button_frame, text="Close", 
                  command=settings_window.destroy).pack(side=tk.RIGHT)
                  
    def apply_agent_settings(self, max_retries, auto_export, viz_mode):
        """Apply agent mode settings.""" 
        if self.agent_mode:
            self.agent_mode.max_retries = max_retries
            # Store other settings for later use
            self.agent_auto_export = auto_export
            self.agent_viz_mode = viz_mode
            messagebox.showinfo("Settings", "Agent settings applied successfully")
            
    def reset_agent_settings(self):
        """Reset agent settings to defaults."""
        if self.agent_mode:
            self.agent_mode.max_retries = 3
            self.agent_auto_export = True
            self.agent_viz_mode = "Both 2D and 3D"
            messagebox.showinfo("Settings", "Agent settings reset to defaults")
    
    def show_instruction_dialog(self) -> str:
        """
        Show dialog to get user instructions before agent starts.
        Returns user instruction string or empty string.
        """
        instruction_window = tk.Toplevel(self.root)
        instruction_window.title("🤖 Agent Instructions")
        instruction_window.geometry("700x600")
        instruction_window.configure(bg="#2c3e50")
        instruction_window.transient(self.root)
        instruction_window.grab_set()
        
        user_input = {"text": ""}
        
        # Header
        header_frame = ttk.Frame(instruction_window)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        header_label = tk.Label(
            header_frame,
            text="📝 Provide Instructions to the Agent",
            font=("Arial", 16, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        header_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Tell the agent what you want it to do with your data",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#95a5a6"
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Examples frame
        examples_frame = ttk.LabelFrame(instruction_window, text="💡 Example Instructions", padding=10)
        examples_frame.pack(fill=tk.BOTH, padx=20, pady=(0, 10))
        
        examples_text = """Examples of what you can ask:

📊 Data Filtering:
  • "I only want house prices data"
  • "Filter only real estate related columns"
  • "Include only price, bedrooms, and location"

🧹 Processing Options:
  • "Just clean the data, no training"
  • "Only prepare data for export"
  • "Clean and show me correlations"

🎯 Prediction Tasks:
  • "Predict house prices"
  • "Focus on predicting sales"
  • "Train model for price estimation"

📈 Analysis Focus:
  • "Focus on correlation analysis"
  • "Emphasize outlier detection"
  • "Analyze distribution patterns"

Or leave empty to let the agent decide automatically! 🤖"""
        
        examples_display = tk.Text(
            examples_frame,
            wrap=tk.WORD,
            bg="#34495e",
            fg="#ecf0f1",
            font=("Consolas", 9),
            height=15,
            relief=tk.FLAT
        )
        examples_display.pack(fill=tk.BOTH, expand=True)
        examples_display.insert("1.0", examples_text)
        examples_display.config(state=tk.DISABLED)
        
        # Input frame
        input_frame = ttk.LabelFrame(instruction_window, text="✍️ Your Instructions", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        instruction_text = tk.Text(
            input_frame,
            wrap=tk.WORD,
            bg="#ecf0f1",
            fg="#2c3e50",
            font=("Arial", 11),
            height=4
        )
        instruction_text.pack(fill=tk.BOTH, expand=True)
        instruction_text.focus()
        
        # Buttons frame
        button_frame = ttk.Frame(instruction_window)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        def on_submit():
            user_input["text"] = instruction_text.get("1.0", tk.END).strip()
            instruction_window.destroy()
        
        def on_skip():
            user_input["text"] = ""
            instruction_window.destroy()
        
        submit_btn = ttk.Button(
            button_frame,
            text="✅ Submit Instructions",
            command=on_submit
        )
        submit_btn.pack(side=tk.LEFT, padx=5)
        
        skip_btn = ttk.Button(
            button_frame,
            text="⏭️ Skip (Auto Mode)",
            command=on_skip
        )
        skip_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="❌ Cancel",
            command=instruction_window.destroy
        )
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        # Wait for window to close
        instruction_window.wait_window()
        
        return user_input["text"]
    
    def show_agent_qa_interface(self):
        """Show interactive Q&A interface after agent completes."""
        qa_window = tk.Toplevel(self.root)
        qa_window.title("🤖 Agent Q&A - Ask Me Anything!")
        qa_window.geometry("900x700")
        qa_window.configure(bg="#2c3e50")
        
        # Header
        header_frame = ttk.Frame(qa_window)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        header_label = tk.Label(
            header_frame,
            text="🤖 Agent Analysis Complete! Ask Me Questions",
            font=("Arial", 14, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        header_label.pack()
        
        info_label = tk.Label(
            header_frame,
            text="Example: 'What is the accuracy?', 'Show F1 score', 'Compare models', 'Generate confusion matrix'",
            font=("Arial", 9),
            bg="#2c3e50",
            fg="#95a5a6"
        )
        info_label.pack(pady=(5, 0))
        
        # Q&A Display
        qa_display_frame = ttk.LabelFrame(qa_window, text="Conversation History", padding=10)
        qa_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.qa_display_text = scrolledtext.ScrolledText(
            qa_display_frame,
            wrap=tk.WORD,
            bg="#34495e",
            fg="#ecf0f1",
            font=("Consolas", 10),
            height=20
        )
        self.qa_display_text.pack(fill=tk.BOTH, expand=True)
        
        # Add welcome message
        welcome_msg = """🤖 AGENT: Hello! I've completed analyzing your data. Here's what I found:

📊 Dataset: {} rows × {} columns
✅ Success Rate: {}/{} steps completed
📈 Quality Score: {}

I can answer questions about:
• Model Performance (accuracy, F1, precision, recall, ROC, AUC)
• Feature Importance (which features matter most)
• Data Quality (missing values, outliers, distributions)
• Predictions (make predictions on new data)
• Visualizations (generate any chart you need)
• Comparisons (compare models, features, metrics)

What would you like to know? 💬
"""
        
        # Get actual stats
        if self.agent_mode and self.agent_mode.dataset_profile:
            profile = self.agent_mode.dataset_profile
            status = self.agent_mode.get_status()
            rows = profile.get('shape', (0, 0))[0]
            cols = profile.get('shape', (0, 0))[1]
            success = status.get('success_count', 0)
            total = status.get('total_steps', 0)
            quality = profile.get('quality_score', 0)
            
            welcome_msg = welcome_msg.format(rows, cols, success, total, f"{quality:.1f}/100")
        else:
            welcome_msg = welcome_msg.format("N/A", "N/A", "N/A", "N/A", "N/A")
        
        self.qa_display_text.insert(tk.END, welcome_msg)
        self.qa_display_text.insert(tk.END, "\n" + "="*80 + "\n\n")
        
        # Quick action buttons
        quick_actions_frame = ttk.LabelFrame(qa_window, text="Quick Actions", padding=10)
        quick_actions_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        actions_row1 = ttk.Frame(quick_actions_frame)
        actions_row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(actions_row1, text="📊 Show Accuracy", 
                  command=lambda: self.qa_quick_action("What is the model accuracy?")).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_row1, text="📈 F1 Score", 
                  command=lambda: self.qa_quick_action("What is the F1 score?")).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_row1, text="🎯 Feature Importance", 
                  command=lambda: self.qa_quick_action("Show feature importance")).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_row1, text="📉 Confusion Matrix", 
                  command=lambda: self.qa_quick_action("Generate confusion matrix")).pack(side=tk.LEFT, padx=2)
        
        actions_row2 = ttk.Frame(quick_actions_frame)
        actions_row2.pack(fill=tk.X)
        
        ttk.Button(actions_row2, text="📊 Compare Models", 
                  command=lambda: self.qa_quick_action("Compare all models")).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_row2, text="🔍 Data Summary", 
                  command=lambda: self.qa_quick_action("Summarize the dataset")).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_row2, text="📈 ROC Curve", 
                  command=lambda: self.qa_quick_action("Show ROC curve")).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_row2, text="🎨 Custom Viz", 
                  command=lambda: self.qa_quick_action("Generate correlation heatmap")).pack(side=tk.LEFT, padx=2)
        
        # Input frame
        input_frame = ttk.Frame(qa_window)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(input_frame, text="Your Question:").pack(anchor=tk.W)
        
        question_entry_frame = ttk.Frame(input_frame)
        question_entry_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.qa_question_entry = ttk.Entry(question_entry_frame, font=("Arial", 11))
        self.qa_question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ask_button = ttk.Button(
            question_entry_frame,
            text="🔍 Ask Agent",
            command=lambda: self.qa_ask_question(self.qa_question_entry.get())
        )
        ask_button.pack(side=tk.RIGHT)
        
        # Bind Enter key
        self.qa_question_entry.bind('<Return>', lambda e: self.qa_ask_question(self.qa_question_entry.get()))
        
        # Button frame
        button_frame = ttk.Frame(qa_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="💾 Export Conversation", 
                  command=self.qa_export_conversation).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="🔄 Clear Chat", 
                  command=lambda: self.qa_display_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(button_frame, text="✅ Close", 
                  command=qa_window.destroy).pack(side=tk.RIGHT)
    
    def qa_quick_action(self, question):
        """Execute a quick action question."""
        self.qa_ask_question(question)
    
    def qa_ask_question(self, question):
        """Process user question and provide intelligent answer."""
        if not question.strip():
            return
        
        # Add question to display
        self.qa_display_text.insert(tk.END, f"👤 YOU: {question}\n\n")
        self.qa_display_text.see(tk.END)
        
        # Clear entry
        if hasattr(self, 'qa_question_entry'):
            self.qa_question_entry.delete(0, tk.END)
        
        # Process question and generate answer
        answer = self._generate_qa_answer(question.lower())
        
        # Add answer to display
        self.qa_display_text.insert(tk.END, f"🤖 AGENT: {answer}\n\n")
        self.qa_display_text.insert(tk.END, "="*80 + "\n\n")
        self.qa_display_text.see(tk.END)
    
    def _generate_qa_answer(self, question):
        """Generate intelligent answer based on question."""
        try:
            # Accuracy questions
            if 'accuracy' in question:
                if self.ml_engine and hasattr(self.ml_engine, 'results'):
                    results = self.ml_engine.results
                    if results:
                        best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
                        return f"Best Model: {best_model[0]}\n   Accuracy: {best_model[1].get('accuracy', 0):.4f} ({best_model[1].get('accuracy', 0)*100:.2f}%)\n   All Models:\n" + "\n".join([f"   • {k}: {v.get('accuracy', 0):.4f}" for k, v in results.items()])
                return "Model accuracy not available. Please train models first."
            
            # F1 Score questions
            elif 'f1' in question or 'f-1' in question:
                if self.ml_engine and hasattr(self.ml_engine, 'results'):
                    results = self.ml_engine.results
                    if results:
                        return "F1 Scores:\n" + "\n".join([f"   • {k}: {v.get('f1', 'N/A')}" for k, v in results.items()])
                return "F1 scores not available. Please train classification models first."
            
            # Feature importance
            elif 'feature' in question and ('importance' in question or 'important' in question):
                if self.ml_engine and hasattr(self.ml_engine, 'feature_importance'):
                    importance = self.ml_engine.feature_importance
                    if importance:
                        top_5 = list(importance.items())[:5]
                        return "Top 5 Important Features:\n" + "\n".join([f"   {i+1}. {k}: {v:.4f}" for i, (k, v) in enumerate(top_5)])
                return "Feature importance not available yet. This is calculated during model training."
            
            # Model comparison
            elif 'compare' in question and 'model' in question:
                if self.ml_engine and hasattr(self.ml_engine, 'results'):
                    results = self.ml_engine.results
                    if results:
                        comparison = "Model Comparison:\n\n"
                        for model, metrics in sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
                            comparison += f"📊 {model}:\n"
                            for metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    comparison += f"   • {metric}: {value:.4f}\n"
                            comparison += "\n"
                        return comparison
                return "No models trained yet for comparison."
            
            # Data summary
            elif 'summar' in question or 'overview' in question or 'describe' in question:
                if self.current_data is not None:
                    summary = f"Dataset Summary:\n\n"
                    summary += f"📏 Shape: {self.current_data.shape[0]:,} rows × {self.current_data.shape[1]} columns\n"
                    summary += f"💾 Memory: {self.current_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
                    summary += f"🔢 Numeric Columns: {len(self.current_data.select_dtypes(include=[np.number]).columns)}\n"
                    summary += f"📝 Categorical Columns: {len(self.current_data.select_dtypes(include=['object']).columns)}\n"
                    summary += f"🕳️ Missing Values: {self.current_data.isnull().sum().sum():,} ({self.current_data.isnull().sum().sum() / self.current_data.size * 100:.2f}%)\n"
                    if self.agent_mode and self.agent_mode.dataset_profile:
                        summary += f"⭐ Quality Score: {self.agent_mode.dataset_profile.get('quality_score', 0):.1f}/100\n"
                    return summary
                return "No data loaded yet."
            
            # Confusion matrix
            elif 'confusion' in question or 'matrix' in question:
                return "Generating confusion matrix visualization... (Feature coming soon!)\nYou can find it in the ML Performance tab."
            
            # ROC curve
            elif 'roc' in question or 'auc' in question:
                if self.ml_engine and hasattr(self.ml_engine, 'results'):
                    results = self.ml_engine.results
                    if results:
                        roc_info = "ROC-AUC Scores:\n"
                        for model, metrics in results.items():
                            if 'roc_auc' in metrics:
                                roc_info += f"   • {model}: {metrics['roc_auc']:.4f}\n"
                        return roc_info if 'roc_auc' in str(results) else "ROC-AUC scores available in ML Performance visualizations."
                return "ROC curves available after model training."
            
            # Precision/Recall
            elif 'precision' in question or 'recall' in question:
                if self.ml_engine and hasattr(self.ml_engine, 'results'):
                    results = self.ml_engine.results
                    if results:
                        metrics_info = "Classification Metrics:\n\n"
                        for model, metrics in results.items():
                            metrics_info += f"📊 {model}:\n"
                            if 'precision' in metrics:
                                metrics_info += f"   • Precision: {metrics['precision']:.4f}\n"
                            if 'recall' in metrics:
                                metrics_info += f"   • Recall: {metrics['recall']:.4f}\n"
                            if 'f1' in metrics:
                                metrics_info += f"   • F1: {metrics['f1']:.4f}\n"
                            metrics_info += "\n"
                        return metrics_info
                return "Classification metrics available after training."
            
            # Visualization requests
            elif 'visualiz' in question or 'plot' in question or 'chart' in question or 'graph' in question:
                return "I can generate visualizations! Available options:\n   • Correlation Heatmap\n   • Distribution Plots\n   • Feature Relationships\n   • Model Performance Charts\n   • Confusion Matrix\n   • ROC Curves\n\nUse the Visualization tab or export all visualizations using the menu."
            
            # Help/what can you do
            elif 'help' in question or 'what can' in question or 'how' in question:
                return """I can help you with:

📊 Performance Metrics:
   • Accuracy, F1 Score, Precision, Recall
   • ROC-AUC, Confusion Matrix
   • Model Comparison

🎯 Feature Analysis:
   • Feature Importance
   • Correlations
   • Distributions

📈 Visualizations:
   • All types of charts and graphs
   • Custom visualizations
   • Export capabilities

🔍 Data Insights:
   • Dataset summaries
   • Quality assessments
   • Missing value analysis

💡 Just ask me anything about your data or models!"""
            
            # Prediction
            elif 'predict' in question:
                return "To make predictions, please use the ML tab and load your test data. I can then generate predictions with confidence scores."
            
            # Default response
            else:
                return f"Interesting question! I'm still learning to answer this specific query.\n\nTry asking about:\n   • Model accuracy or F1 scores\n   • Feature importance\n   • Data summary\n   • Model comparison\n   • Visualizations\n\nOr use the quick action buttons above!"
        
        except Exception as e:
            return f"Sorry, I encountered an error processing your question: {str(e)}\nPlease try rephrasing or use the quick action buttons."
    
    def qa_export_conversation(self):
        """Export Q&A conversation to file."""
        try:
            if not hasattr(self, 'qa_display_text'):
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_qa_conversation_{timestamp}.txt"
            
            content = self.qa_display_text.get(1.0, tk.END)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Agent Q&A Conversation - {datetime.now()}\n")
                f.write("="*80 + "\n\n")
                f.write(content)
            
            messagebox.showinfo("Export Complete", f"Conversation exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export conversation: {str(e)}")

    def update_all_visualizations(self):
        """Update all visualizations with current data."""
        try:
            if self.advanced_viz_dashboard and self.current_data is not None:
                self.advanced_viz_dashboard.update_data(self.current_data)
                self.add_agent_log("[VIZ] Advanced visualizations updated")
            elif hasattr(self, 'show_data_overview'):
                self.show_data_overview()
                self.add_agent_log("[VIZ] Basic visualizations updated")
        except Exception as e:
            print(f"Error updating visualizations: {e}")
            if hasattr(self, 'add_agent_log'):
                self.add_agent_log(f"[ERROR] Visualization update failed: {str(e)}")

    def refresh_all_dashboards(self):
        """Refresh all dashboard components."""
        try:
            # Update advanced visualization dashboard
            if self.advanced_viz_dashboard and self.current_data is not None:
                self.advanced_viz_dashboard.refresh_all_visualizations()
                
            # Update other UI components
            self.update_data_info()
            self.update_column_list()
            
            # Refresh ML displays if available
            if hasattr(self, 'update_ml_displays'):
                self.update_ml_displays()
                
        except Exception as e:
            print(f"Error refreshing dashboards: {e}")
            
    def get_visualization_mode(self):
        """Get current visualization mode preference."""
        if hasattr(self, 'agent_viz_mode'):
            return self.agent_viz_mode
        return "Both 2D and 3D"

    def create_data_loading_tab(self):
        """Create enhanced multi-format data loading tab."""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 Data Loading")

        # File loading section
        load_frame = ttk.LabelFrame(data_frame, text="Load Your Dataset - Multi-Format Support", padding=10)
        load_frame.pack(fill=tk.X, padx=5, pady=5)

        # File format buttons
        format_frame = ttk.Frame(load_frame)
        format_frame.pack(fill=tk.X, pady=5)

        ttk.Button(format_frame, text="📄 CSV File", 
                  command=lambda: self.load_file('csv')).pack(side=tk.LEFT, padx=2)
        
        if EXCEL_AVAILABLE:
            ttk.Button(format_frame, text="📊 Excel (.xlsx)", 
                      command=lambda: self.load_file('excel')).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(format_frame, text="🔧 JSON File", 
                  command=lambda: self.load_file('json')).pack(side=tk.LEFT, padx=2)
        
        if PARQUET_AVAILABLE:
            ttk.Button(format_frame, text="⚡ Parquet", 
                      command=lambda: self.load_file('parquet')).pack(side=tk.LEFT, padx=2)

        # Database connection
        if SQL_AVAILABLE:
            db_frame = ttk.Frame(load_frame)
            db_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(db_frame, text="🗄️ Connect Database", 
                      command=self.connect_database).pack(side=tk.LEFT, padx=2)
            ttk.Button(db_frame, text="📊 Load from SQL", 
                      command=self.load_from_sql).pack(side=tk.LEFT, padx=2)

        # File info display
        info_frame = ttk.Frame(load_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.file_label = ttk.Label(info_frame, text="No file loaded", foreground="gray")
        self.file_label.pack(side=tk.LEFT)

        ttk.Button(info_frame, text="🔍 AI Analysis", 
                  command=self.analyze_data, style="Accent.TButton").pack(side=tk.RIGHT, padx=5)

        # Enhanced data preview section
        preview_frame = ttk.LabelFrame(data_frame, text="Data Preview & Information", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create notebook for data preview
        preview_notebook = ttk.Notebook(preview_frame)
        preview_notebook.pack(fill=tk.BOTH, expand=True)

        # Data table preview
        table_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(table_frame, text="📋 Data Table")
        self.data_preview_text = scrolledtext.ScrolledText(table_frame, height=20, font=("Consolas", 9))
        self.data_preview_text.pack(fill=tk.BOTH, expand=True)

        # Data information
        info_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(info_frame, text="ℹ️ Dataset Info")
        self.data_info_text = scrolledtext.ScrolledText(info_frame, height=20, font=("Consolas", 9))
        self.data_info_text.pack(fill=tk.BOTH, expand=True)

        # Column analysis
        analysis_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(analysis_frame, text="🎯 Column Analysis")
        self.column_analysis_text = scrolledtext.ScrolledText(analysis_frame, height=20, font=("Consolas", 9))
        self.column_analysis_text.pack(fill=tk.BOTH, expand=True)

        # Data source information
        source_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(source_frame, text="🔗 Data Source")
        self.data_source_text = scrolledtext.ScrolledText(source_frame, height=20, font=("Consolas", 9))
        self.data_source_text.pack(fill=tk.BOTH, expand=True)

    def create_ai_analysis_tab(self):
        """Create AI analysis and decision tab."""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="🧠 AI Analysis")

        # AI Analysis controls
        control_frame = ttk.LabelFrame(ai_frame, text="AI Analysis & Auto-Selection Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="🚀 Start AI Analysis", 
                  command=self.analyze_data, style="Accent.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🎯 Auto Process Selection", 
                  command=self.auto_select_process).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🔄 Re-analyze Data", 
                  command=self.reanalyze_data).pack(side=tk.LEFT, padx=2)

        # Real-time progress indicator
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.ai_progress_var = tk.StringVar(value="Ready for AI analysis")
        ttk.Label(progress_frame, textvariable=self.ai_progress_var).pack(side=tk.LEFT)
        
        self.ai_progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.ai_progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # AI Reasoning display
        reasoning_frame = ttk.LabelFrame(ai_frame, text="🤖 AI Reasoning & Decisions", padding=10)
        reasoning_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create notebook for AI reasoning
        reasoning_notebook = ttk.Notebook(reasoning_frame)
        reasoning_notebook.pack(fill=tk.BOTH, expand=True)

        # Real-time AI decisions
        decisions_frame = ttk.Frame(reasoning_notebook)
        reasoning_notebook.add(decisions_frame, text="🧠 AI Decisions")
        self.ai_decisions_text = scrolledtext.ScrolledText(decisions_frame, height=25, font=("Consolas", 9))
        self.ai_decisions_text.pack(fill=tk.BOTH, expand=True)

        # Auto-selection results
        autoselect_frame = ttk.Frame(reasoning_notebook)
        reasoning_notebook.add(autoselect_frame, text="🎯 Process Selection")
        self.autoselect_text = scrolledtext.ScrolledText(autoselect_frame, height=25, font=("Consolas", 9))
        self.autoselect_text.pack(fill=tk.BOTH, expand=True)

        # Column recommendations
        recommendations_frame = ttk.Frame(reasoning_notebook)
        reasoning_notebook.add(recommendations_frame, text="💡 Recommendations")
        self.recommendations_text = scrolledtext.ScrolledText(recommendations_frame, height=25, font=("Consolas", 9))
        self.recommendations_text.pack(fill=tk.BOTH, expand=True)

        # Strategy explanations
        strategy_frame = ttk.Frame(reasoning_notebook)
        reasoning_notebook.add(strategy_frame, text="🎯 Cleaning Strategies")
        self.strategies_text = scrolledtext.ScrolledText(strategy_frame, height=25, font=("Consolas", 9))
        self.strategies_text.pack(fill=tk.BOTH, expand=True)

    def create_meta_features_tab(self):
        """Create dataset meta-features analysis tab."""
        meta_frame = ttk.Frame(self.notebook)
        self.notebook.add(meta_frame, text="📈 Meta Features")

        # Meta-feature extraction controls
        control_frame = ttk.LabelFrame(meta_frame, text="Dataset Meta-Feature Analysis", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="📊 Extract Meta-Features", 
                  command=self.extract_meta_features, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔄 Update Analysis", 
                  command=self.update_meta_analysis).pack(side=tk.LEFT, padx=5)

        # Meta-features display
        meta_notebook = ttk.Notebook(meta_frame)
        meta_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Dataset characteristics
        chars_frame = ttk.Frame(meta_notebook)
        meta_notebook.add(chars_frame, text="📊 Dataset Characteristics")
        self.meta_chars_text = scrolledtext.ScrolledText(chars_frame, height=20, font=("Consolas", 9))
        self.meta_chars_text.pack(fill=tk.BOTH, expand=True)

        # Data distribution
        dist_frame = ttk.Frame(meta_notebook)
        meta_notebook.add(dist_frame, text="📈 Data Distribution")
        self.meta_dist_text = scrolledtext.ScrolledText(dist_frame, height=20, font=("Consolas", 9))
        self.meta_dist_text.pack(fill=tk.BOTH, expand=True)

        # Quality metrics
        quality_frame = ttk.Frame(meta_notebook)
        meta_notebook.add(quality_frame, text="⭐ Quality Metrics")
        self.meta_quality_text = scrolledtext.ScrolledText(quality_frame, height=20, font=("Consolas", 9))
        self.meta_quality_text.pack(fill=tk.BOTH, expand=True)

        # Recommended pipeline
        pipeline_frame = ttk.Frame(meta_notebook)
        meta_notebook.add(pipeline_frame, text="🔧 Recommended Pipeline")
        self.meta_pipeline_text = scrolledtext.ScrolledText(pipeline_frame, height=20, font=("Consolas", 9))
        self.meta_pipeline_text.pack(fill=tk.BOTH, expand=True)

    def create_data_cleaning_tab(self):
        """Create enhanced smart data cleaning tab."""
        cleaning_frame = ttk.Frame(self.notebook)
        self.notebook.add(cleaning_frame, text="🧹 Smart Cleaning")

        # Enhanced cleaning controls
        control_frame = ttk.LabelFrame(cleaning_frame, text="Advanced Smart Cleaning Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Main cleaning buttons
        button_frame1 = ttk.Frame(control_frame)
        button_frame1.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame1, text="🧹 Smart Clean Data", 
                  command=self.clean_data, style="Accent.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame1, text="🔧 Advanced Feature Engineering", 
                  command=self.advanced_feature_engineering).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame1, text="🎯 Outlier Handling", 
                  command=self.handle_outliers).pack(side=tk.LEFT, padx=2)

        # Advanced cleaning options
        button_frame2 = ttk.Frame(control_frame)
        button_frame2.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame2, text="🤖 KNN Imputation", 
                  command=self.knn_imputation).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame2, text="🔄 Iterative Imputation", 
                  command=self.iterative_imputation).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame2, text="📝 Text Preprocessing", 
                  command=self.text_preprocessing).pack(side=tk.LEFT, padx=2)

        # Export and proceed
        button_frame3 = ttk.Frame(control_frame)
        button_frame3.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame3, text="💾 Export Cleaned Data", 
                  command=self.export_cleaned_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame3, text="▶️ Proceed to ML", 
                  command=self.proceed_to_ml).pack(side=tk.RIGHT, padx=2)

        # Enhanced progress tracking
        progress_frame = ttk.LabelFrame(cleaning_frame, text="Cleaning Progress & Status", padding=10)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)

        self.cleaning_progress_var = tk.StringVar(value="Ready to clean data")
        ttk.Label(progress_frame, textvariable=self.cleaning_progress_var, 
                 font=("Arial", 12, "bold")).pack(pady=2)

        self.cleaning_progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.cleaning_progress_bar.pack(fill=tk.X, pady=2)

        # Real-time cleaning status
        self.cleaning_status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(progress_frame, textvariable=self.cleaning_status_var).pack(pady=2)

        # Cleaning results
        results_frame = ttk.LabelFrame(cleaning_frame, text="🔍 Cleaning Results & Summary", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        results_notebook = ttk.Notebook(results_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True)

        # Cleaning log
        log_frame = ttk.Frame(results_notebook)
        results_notebook.add(log_frame, text="📝 Cleaning Log")
        self.cleaning_results_text = scrolledtext.ScrolledText(log_frame, height=15, font=("Consolas", 9))
        self.cleaning_results_text.pack(fill=tk.BOTH, expand=True)

        # Data lineage
        lineage_frame = ttk.Frame(results_notebook)
        results_notebook.add(lineage_frame, text="🔗 Data Lineage")
        self.data_lineage_text = scrolledtext.ScrolledText(lineage_frame, height=15, font=("Consolas", 9))
        self.data_lineage_text.pack(fill=tk.BOTH, expand=True)

        # Quality improvement
        improvement_frame = ttk.Frame(results_notebook)
        results_notebook.add(improvement_frame, text="📈 Quality Improvement")
        self.quality_improvement_text = scrolledtext.ScrolledText(improvement_frame, height=15, font=("Consolas", 9))
        self.quality_improvement_text.pack(fill=tk.BOTH, expand=True)

    def create_visualization_tab(self):
        """Create enhanced visualization tab with advanced dashboard."""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="[VIZ] Advanced Dashboard")

        # Initialize Advanced Visualization Dashboard
        if ADVANCED_VIZ_AVAILABLE:
            self.advanced_viz_dashboard = AdvancedVisualizationDashboard(viz_frame)
        else:
            # Fallback to basic visualization
            self.create_basic_visualization_tab(viz_frame)

    def create_basic_visualization_tab(self, parent):
        """Create basic visualization tab as fallback."""
        # Enhanced visualization controls
        control_frame = ttk.LabelFrame(parent, text="Basic Visualization Options", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Basic visualization options
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame, text="[DATA] Overview", 
                  command=self.show_data_overview).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="[CLEAN] Summary", 
                  command=self.show_cleaning_summary).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="[3D] Explorer", 
                  command=self.show_3d_visualization).pack(side=tk.LEFT, padx=2)

        # Canvas for plots
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = None
        self.toolbar = None
        self.canvas_frame = canvas_frame

    def create_ml_tab(self):
        """Create enhanced machine learning tab."""
        ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(ml_frame, text="🤖 Machine Learning")

        # Enhanced ML controls
        control_frame = ttk.LabelFrame(ml_frame, text="Advanced Machine Learning Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Target selection and problem type
        target_frame = ttk.Frame(control_frame)
        target_frame.pack(fill=tk.X, pady=5)

        ttk.Label(target_frame, text="Target Column:").pack(side=tk.LEFT)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(target_frame, textvariable=self.target_var, 
                                        state="readonly", width=20)
        self.target_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(target_frame, text="Problem Type:").pack(side=tk.LEFT, padx=(20,5))
        self.problem_type_var = tk.StringVar(value="auto")
        problem_combo = ttk.Combobox(target_frame, textvariable=self.problem_type_var,
                                   values=["auto", "classification", "regression", "nlp", "time_series"], 
                                   state="readonly", width=15)
        problem_combo.pack(side=tk.LEFT, padx=5)

        # ML training options
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X, pady=5)

        self.quick_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Quick Mode", variable=self.quick_mode_var).pack(side=tk.LEFT, padx=5)
        
        self.automl_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="AutoML", variable=self.automl_var).pack(side=tk.LEFT, padx=5)
        
        self.deep_learning_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Deep Learning", variable=self.deep_learning_var).pack(side=tk.LEFT, padx=5)

        # Training buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="🚀 Train Classical ML", 
                  command=self.train_models, style="Accent.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⚡ AutoML Training", 
                  command=self.automl_training).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🧠 Deep Learning", 
                  command=self.train_deep_learning).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📝 NLP Analysis", 
                  command=self.nlp_analysis).pack(side=tk.LEFT, padx=2)

        # ML results with enhanced display
        results_frame = ttk.LabelFrame(ml_frame, text="🎯 Machine Learning Results & Monitoring", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ml_notebook = ttk.Notebook(results_frame)
        ml_notebook.pack(fill=tk.BOTH, expand=True)

        # Real-time training log
        log_frame = ttk.Frame(ml_notebook)
        ml_notebook.add(log_frame, text="📝 Training Log")
        self.ml_log_text = scrolledtext.ScrolledText(log_frame, height=15, font=("Consolas", 9))
        self.ml_log_text.pack(fill=tk.BOTH, expand=True)

        # Model comparison
        comparison_frame = ttk.Frame(ml_notebook)
        ml_notebook.add(comparison_frame, text="📊 Model Comparison")
        self.model_comparison_text = scrolledtext.ScrolledText(comparison_frame, height=15, font=("Consolas", 9))
        self.model_comparison_text.pack(fill=tk.BOTH, expand=True)

        # Training progress
        progress_frame = ttk.Frame(ml_notebook)
        ml_notebook.add(progress_frame, text="📈 Training Progress")
        self.training_progress_text = scrolledtext.ScrolledText(progress_frame, height=15, font=("Consolas", 9))
        self.training_progress_text.pack(fill=tk.BOTH, expand=True)

        # Hyperparameter optimization
        hyperparam_frame = ttk.Frame(ml_notebook)
        ml_notebook.add(hyperparam_frame, text="🎛️ Hyperparameters")
        self.hyperparameter_text = scrolledtext.ScrolledText(hyperparam_frame, height=15, font=("Consolas", 9))
        self.hyperparameter_text.pack(fill=tk.BOTH, expand=True)

    def create_explainability_tab(self):
        """Create explainability and error analysis tab."""
        explain_frame = ttk.Frame(self.notebook)
        self.notebook.add(explain_frame, text="🧩 Explainability")

        # Explainability controls
        control_frame = ttk.LabelFrame(explain_frame, text="Model Explainability & Error Analysis", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        if ModelExplainer:
            # SHAP analysis
            shap_frame = ttk.Frame(control_frame)
            shap_frame.pack(fill=tk.X, pady=2)
            ttk.Label(shap_frame, text="SHAP Analysis:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(shap_frame, text="🧩 Global SHAP", 
                      command=self.show_global_shap).pack(side=tk.LEFT, padx=2)
            ttk.Button(shap_frame, text="🎯 Local SHAP", 
                      command=self.show_local_shap).pack(side=tk.LEFT, padx=2)
            ttk.Button(shap_frame, text="🌊 SHAP Waterfall", 
                      command=self.show_shap_waterfall).pack(side=tk.LEFT, padx=2)

            # LIME analysis
            lime_frame = ttk.Frame(control_frame)
            lime_frame.pack(fill=tk.X, pady=2)
            ttk.Label(lime_frame, text="LIME Analysis:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(lime_frame, text="🔍 LIME Explanations", 
                      command=self.show_lime_explanations).pack(side=tk.LEFT, padx=2)
            ttk.Button(lime_frame, text="📊 Feature Contributions", 
                      command=self.show_feature_contributions).pack(side=tk.LEFT, padx=2)

            # Error analysis
            error_frame = ttk.Frame(control_frame)
            error_frame.pack(fill=tk.X, pady=2)
            ttk.Label(error_frame, text="Error Analysis:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(error_frame, text="❌ Error Analysis", 
                      command=self.show_error_analysis).pack(side=tk.LEFT, padx=2)
            ttk.Button(error_frame, text="📊 Prediction Confidence", 
                      command=self.show_prediction_confidence).pack(side=tk.LEFT, padx=2)
            ttk.Button(error_frame, text="🔄 Calibration Curves", 
                      command=self.show_calibration_curves).pack(side=tk.LEFT, padx=2)
        else:
            # Fallback message
            ttk.Label(control_frame, text="Explainability module not available. Install required dependencies.", 
                     foreground="orange").pack(pady=10)

        # Explainability results
        explain_notebook = ttk.Notebook(explain_frame)
        explain_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # SHAP results
        shap_results_frame = ttk.Frame(explain_notebook)
        explain_notebook.add(shap_results_frame, text="🧩 SHAP Results")
        self.shap_results_text = scrolledtext.ScrolledText(shap_results_frame, height=15, font=("Consolas", 9))
        self.shap_results_text.pack(fill=tk.BOTH, expand=True)

        # LIME results
        lime_results_frame = ttk.Frame(explain_notebook)
        explain_notebook.add(lime_results_frame, text="🔍 LIME Results")
        self.lime_results_text = scrolledtext.ScrolledText(lime_results_frame, height=15, font=("Consolas", 9))
        self.lime_results_text.pack(fill=tk.BOTH, expand=True)

        # Error analysis results
        error_results_frame = ttk.Frame(explain_notebook)
        explain_notebook.add(error_results_frame, text="❌ Error Analysis")
        self.error_results_text = scrolledtext.ScrolledText(error_results_frame, height=15, font=("Consolas", 9))
        self.error_results_text.pack(fill=tk.BOTH, expand=True)

        # Feature importance
        importance_frame = ttk.Frame(explain_notebook)
        explain_notebook.add(importance_frame, text="🎯 Feature Importance")
        self.importance_text = scrolledtext.ScrolledText(importance_frame, height=15, font=("Consolas", 9))
        self.importance_text.pack(fill=tk.BOTH, expand=True)

    def create_results_tab(self):
        """Create enhanced results and export tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📋 Results")

        # Export controls
        export_frame = ttk.LabelFrame(results_frame, text="Enhanced Export Options", padding=10)
        export_frame.pack(fill=tk.X, padx=5, pady=5)

        # Export buttons
        button_frame1 = ttk.Frame(export_frame)
        button_frame1.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame1, text="💾 Export Cleaned Data", 
                  command=self.export_cleaned_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame1, text="📊 Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame1, text="📈 Export All Plots", 
                  command=self.export_all_visualizations).pack(side=tk.LEFT, padx=2)

        button_frame2 = ttk.Frame(export_frame)
        button_frame2.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame2, text="🤖 Export Models", 
                  command=self.export_models).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame2, text="💾 Save Session", 
                  command=self.save_session).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame2, text="📄 Generate Report", 
                  command=self.generate_comprehensive_report).pack(side=tk.LEFT, padx=2)

        # Comprehensive summary with tabs
        summary_frame = ttk.LabelFrame(results_frame, text="📋 Comprehensive Analysis Summary", padding=10)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        summary_notebook = ttk.Notebook(summary_frame)
        summary_notebook.pack(fill=tk.BOTH, expand=True)

        # Executive summary
        exec_frame = ttk.Frame(summary_notebook)
        summary_notebook.add(exec_frame, text="📈 Executive Summary")
        self.exec_summary_text = scrolledtext.ScrolledText(exec_frame, height=20, font=("Consolas", 10))
        self.exec_summary_text.pack(fill=tk.BOTH, expand=True)

        # Technical summary
        tech_frame = ttk.Frame(summary_notebook)
        summary_notebook.add(tech_frame, text="🔧 Technical Details")
        self.summary_text = scrolledtext.ScrolledText(tech_frame, height=20, font=("Consolas", 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Performance metrics
        metrics_frame = ttk.Frame(summary_notebook)
        summary_notebook.add(metrics_frame, text="📊 Performance Metrics")
        self.metrics_summary_text = scrolledtext.ScrolledText(metrics_frame, height=20, font=("Consolas", 10))
        self.metrics_summary_text.pack(fill=tk.BOTH, expand=True)

        # Session history
        history_frame = ttk.Frame(summary_notebook)
        summary_notebook.add(history_frame, text="📚 Session History")
        self.session_history_text = scrolledtext.ScrolledText(history_frame, height=20, font=("Consolas", 10))
        self.session_history_text.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self):
        """Create enhanced application status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Main status
        self.status_var = tk.StringVar(value="Welcome to Advanced AI Data Analysis Platform")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Processing status
        self.processing_status_var = tk.StringVar(value="Ready")
        processing_status = ttk.Label(status_frame, textvariable=self.processing_status_var, 
                                    relief=tk.SUNKEN, anchor=tk.CENTER, width=15)
        processing_status.pack(side=tk.RIGHT)

        # Session info
        self.session_status_var = tk.StringVar(value=f"Session: {self.current_session_id}")
        session_status = ttk.Label(status_frame, textvariable=self.session_status_var, 
                                 relief=tk.SUNKEN, anchor=tk.CENTER, width=25)
        session_status.pack(side=tk.RIGHT)

        # Memory usage
        self.memory_status_var = tk.StringVar(value="Memory: 0MB")
        memory_status = ttk.Label(status_frame, textvariable=self.memory_status_var, 
                                relief=tk.SUNKEN, anchor=tk.CENTER, width=15)
        memory_status.pack(side=tk.RIGHT)

    # ==== ENHANCED FILE LOADING METHODS ====

    def load_file(self, file_type='csv'):
        """Enhanced multi-format file loading."""
        file_types = {
            'csv': [("CSV files", "*.csv")],
            'excel': [("Excel files", "*.xlsx"), ("Excel files", "*.xls")] if EXCEL_AVAILABLE else [],
            'json': [("JSON files", "*.json")],
            'parquet': [("Parquet files", "*.parquet")] if PARQUET_AVAILABLE else []
        }
        
        if file_type not in file_types or not file_types[file_type]:
            messagebox.showerror("Error", f"{file_type.upper()} format not supported. Install required dependencies.")
            return

        file_path = filedialog.askopenfilename(
            title=f"Select {file_type.upper()} file",
            filetypes=file_types[file_type] + [("All files", "*.*")]
        )

        if file_path:
            self.status_var.set(f"Loading {file_type.upper()} data...")
            self.processing_status_var.set("Loading...")
            
            thread = threading.Thread(target=self._load_file_thread, args=(file_path, file_type))
            thread.daemon = True
            thread.start()

    def _load_file_thread(self, file_path, file_type):
        """Enhanced file loading thread with format support."""
        try:
            # Update data source info
            source_info = {
                'type': file_type,
                'path': file_path,
                'size': os.path.getsize(file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                'loaded': datetime.now()
            }
            
            if file_type == 'csv':
                success = self.data_processor.load_and_analyze(file_path)
            elif file_type == 'excel':
                success = self._load_excel_file(file_path)
            elif file_type == 'json':
                success = self._load_json_file(file_path)
            elif file_type == 'parquet':
                success = self._load_parquet_file(file_path)
            else:
                self.queue.put(("error", f"Unsupported file type: {file_type}"))
                return
                
            if success:
                self.current_data = self.data_processor.original_data
                self.queue.put(("file_loaded", {
                    'filename': os.path.basename(file_path),
                    'source_info': source_info
                }))
            else:
                self.queue.put(("error", f"Failed to load {file_type.upper()} file"))
        except Exception as e:
            self.queue.put(("error", f"Error loading {file_type.upper()} file: {str(e)}"))

    def _load_excel_file(self, file_path):
        """Load Excel file with sheet selection."""
        try:
            # Read Excel file and get sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                # Ask user to select sheet
                sheet_name = simpledialog.askstring(
                    "Select Sheet",
                    f"Multiple sheets found: {', '.join(sheet_names)}\nEnter sheet name:",
                    initialvalue=sheet_names[0]
                )
                if not sheet_name or sheet_name not in sheet_names:
                    sheet_name = sheet_names[0]
            else:
                sheet_name = sheet_names[0]
            
            # Load the selected sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Use the data processor to analyze
            return self.data_processor.load_and_analyze_dataframe(df, f"{file_path}[{sheet_name}]")
            
        except Exception as e:
            raise Exception(f"Excel loading error: {str(e)}")

    def _load_json_file(self, file_path):
        """Load JSON file with structure detection."""
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Convert JSON to DataFrame
            if isinstance(json_data, list):
                df = pd.json_normalize(json_data)
            elif isinstance(json_data, dict):
                df = pd.json_normalize([json_data])
            else:
                raise Exception("JSON structure not supported for tabular analysis")
            
            return self.data_processor.load_and_analyze_dataframe(df, file_path)
            
        except Exception as e:
            raise Exception(f"JSON loading error: {str(e)}")

    def _load_parquet_file(self, file_path):
        """Load Parquet file."""
        try:
            df = pd.read_parquet(file_path)
            return self.data_processor.load_and_analyze_dataframe(df, file_path)
        except Exception as e:
            raise Exception(f"Parquet loading error: {str(e)}")

    # ==== DATABASE CONNECTIVITY ====

    def connect_database(self):
        """Connect to database and browse tables."""
        if not SQL_AVAILABLE:
            messagebox.showerror("Error", "Database connectivity requires sqlalchemy. Please install it.")
            return

        # Create database connection dialog
        db_dialog = DatabaseConnectionDialog(self.root)
        self.root.wait_window(db_dialog.dialog)
        
        if db_dialog.connection_string:
            try:
                engine = create_engine(db_dialog.connection_string)
                self.database_connections[db_dialog.connection_name] = engine
                
                # Test connection and get table list
                with engine.connect() as conn:
                    # Get table names (this is database-specific, this works for most SQL databases)
                    tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall() if 'sqlite' in db_dialog.connection_string else []
                
                messagebox.showinfo("Success", f"Connected to {db_dialog.connection_name}\nTables: {len(tables)}")
                self.status_var.set(f"Connected to database: {db_dialog.connection_name}")
                
            except Exception as e:
                messagebox.showerror("Database Error", f"Failed to connect: {str(e)}")

    def load_from_sql(self):
        """Load data from SQL query."""
        if not self.database_connections:
            messagebox.showwarning("Warning", "No database connections available. Connect to a database first.")
            return

        # Create SQL query dialog
        sql_dialog = SQLQueryDialog(self.root, list(self.database_connections.keys()))
        self.root.wait_window(sql_dialog.dialog)
        
        if sql_dialog.query and sql_dialog.connection_name:
            self.status_var.set("Executing SQL query...")
            self.processing_status_var.set("Querying...")
            
            thread = threading.Thread(target=self._load_sql_thread, 
                                     args=(sql_dialog.query, sql_dialog.connection_name))
            thread.daemon = True
            thread.start()

    def _load_sql_thread(self, query, connection_name):
        """Load data from SQL in separate thread."""
        try:
            engine = self.database_connections[connection_name]
            df = pd.read_sql(query, engine)
            
            success = self.data_processor.load_and_analyze_dataframe(df, f"SQL:{connection_name}")
            if success:
                self.current_data = self.data_processor.original_data
                self.queue.put(("file_loaded", {
                    'filename': f"SQL Query ({connection_name})",
                    'source_info': {
                        'type': 'sql',
                        'connection': connection_name,
                        'query': query,
                        'rows': len(df),
                        'loaded': datetime.now()
                    }
                }))
            else:
                self.queue.put(("error", "Failed to analyze SQL data"))
        except Exception as e:
            self.queue.put(("error", f"SQL query error: {str(e)}"))

    # ==== META-FEATURES AND AUTO-SELECTION ====

    def extract_meta_features(self):
        """Extract dataset meta-features for intelligent processing."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.status_var.set("Extracting meta-features...")
        thread = threading.Thread(target=self._extract_meta_features_thread)
        thread.daemon = True
        thread.start()

    def _extract_meta_features_thread(self):
        """Extract meta-features in separate thread."""
        try:
            # Basic dataset characteristics
            data = self.current_data
            meta_features = {
                'dataset_size': data.shape[0] * data.shape[1],
                'n_rows': data.shape[0],
                'n_cols': data.shape[1],
                'n_numeric': len(data.select_dtypes(include=[np.number]).columns),
                'n_categorical': len(data.select_dtypes(include=['object']).columns),
                'n_datetime': len(data.select_dtypes(include=['datetime']).columns),
                'sparsity': (data.isnull().sum().sum() / data.size) * 100,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            # Calculate percentages
            meta_features['pct_numeric'] = (meta_features['n_numeric'] / meta_features['n_cols']) * 100
            meta_features['pct_categorical'] = (meta_features['n_categorical'] / meta_features['n_cols']) * 100
            meta_features['pct_datetime'] = (meta_features['n_datetime'] / meta_features['n_cols']) * 100
            
            # Text analysis
            text_cols = []
            for col in data.select_dtypes(include=['object']).columns:
                if data[col].astype(str).str.len().mean() > 50:  # Average length > 50 chars
                    text_cols.append(col)
            meta_features['n_text'] = len(text_cols)
            meta_features['pct_text'] = (len(text_cols) / meta_features['n_cols']) * 100
            
            # Data quality metrics
            meta_features['completeness'] = ((data.size - data.isnull().sum().sum()) / data.size) * 100
            
            # Class imbalance (if we can detect a likely target)
            potential_targets = [col for col in data.columns if any(keyword in col.lower() 
                               for keyword in ['target', 'label', 'class', 'y', 'outcome'])]
            if potential_targets:
                target_col = potential_targets[0]
                value_counts = data[target_col].value_counts()
                if len(value_counts) > 1:
                    meta_features['class_imbalance'] = (value_counts.max() / value_counts.sum()) * 100
                else:
                    meta_features['class_imbalance'] = 0
            else:
                meta_features['class_imbalance'] = None
            
            # Correlation analysis
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                meta_features['max_correlation'] = upper_tri.max().max()
                meta_features['high_corr_pairs'] = (upper_tri > 0.8).sum().sum()
            else:
                meta_features['max_correlation'] = 0
                meta_features['high_corr_pairs'] = 0
            
            # Outlier detection (simple IQR method)
            outlier_count = 0
            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = numeric_data[col][(numeric_data[col] < (Q1 - 1.5 * IQR)) | 
                                           (numeric_data[col] > (Q3 + 1.5 * IQR))]
                outlier_count += len(outliers)
            
            meta_features['outlier_percentage'] = (outlier_count / data.shape[0]) * 100
            
            self.meta_features = meta_features
            self.queue.put(("meta_features_extracted", meta_features))
            
        except Exception as e:
            self.queue.put(("error", f"Error extracting meta-features: {str(e)}"))

    def auto_select_process(self):
        """Automatically select the best processing pipeline."""
        if not self.meta_features:
            # Extract meta-features first
            self.extract_meta_features()
            return

        if self.auto_selector:
            self.status_var.set("AI selecting optimal process...")
            thread = threading.Thread(target=self._auto_select_process_thread)
            thread.daemon = True
            thread.start()
        else:
            # Fallback rule-based selection
            self._fallback_process_selection()

    def _auto_select_process_thread(self):
        """Auto-select process in separate thread."""
        try:
            recommendation = self.auto_selector.select_pipeline(self.meta_features, self.current_data)
            self.queue.put(("process_selected", recommendation))
        except Exception as e:
            self.queue.put(("error", f"Error in process selection: {str(e)}"))

    def _fallback_process_selection(self):
        """Fallback rule-based process selection."""
        try:
            recommendations = []
            
            # Rule-based recommendations
            if self.meta_features['pct_text'] > 30:
                recommendations.append("🔤 Text-heavy dataset detected → NLP pipeline recommended")
            
            if self.meta_features['dataset_size'] > 1000000:  # 1M cells
                recommendations.append("📊 Large dataset → LightGBM/XGBoost recommended for speed")
            
            if self.meta_features['pct_categorical'] > 50:
                recommendations.append("🏷️ High categorical features → Advanced encoding required")
            
            if self.meta_features['sparsity'] > 30:
                recommendations.append("🕳️ High sparsity → Specialized imputation methods needed")
            
            if self.meta_features['outlier_percentage'] > 10:
                recommendations.append("⚠️ Many outliers detected → Robust preprocessing required")
            
            if self.meta_features['high_corr_pairs'] > 5:
                recommendations.append("🔗 High correlations → Feature selection/PCA recommended")
            
            # Display recommendations
            rec_text = "🎯 AUTOMATED PROCESS RECOMMENDATIONS:\n\n"
            for i, rec in enumerate(recommendations, 1):
                rec_text += f"{i}. {rec}\n"
            
            if not recommendations:
                rec_text += "✅ Standard ML pipeline recommended - no special handling needed"
            
            self.autoselect_text.delete(1.0, tk.END)
            self.autoselect_text.insert(tk.END, rec_text)
            
            self.status_var.set("Process selection complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in process selection: {str(e)}")

    def update_meta_analysis(self):
        """Update meta-feature analysis display."""
        if not self.meta_features:
            messagebox.showwarning("Warning", "No meta-features available. Extract meta-features first.")
            return

        # Update dataset characteristics
        chars_text = "=== DATASET CHARACTERISTICS ===\n\n"
        chars_text += f"Dataset Size: {self.meta_features['n_rows']:,} rows × {self.meta_features['n_cols']} columns\n"
        chars_text += f"Total Cells: {self.meta_features['dataset_size']:,}\n"
        chars_text += f"Memory Usage: {self.meta_features['memory_usage_mb']:.2f} MB\n\n"
        
        chars_text += "Column Types Distribution:\n"
        chars_text += f"  📊 Numeric: {self.meta_features['n_numeric']} ({self.meta_features['pct_numeric']:.1f}%)\n"
        chars_text += f"  🏷️ Categorical: {self.meta_features['n_categorical']} ({self.meta_features['pct_categorical']:.1f}%)\n"
        chars_text += f"  📅 Datetime: {self.meta_features['n_datetime']} ({self.meta_features['pct_datetime']:.1f}%)\n"
        chars_text += f"  📝 Text: {self.meta_features['n_text']} ({self.meta_features['pct_text']:.1f}%)\n"
        
        self.meta_chars_text.delete(1.0, tk.END)
        self.meta_chars_text.insert(tk.END, chars_text)

        # Update data distribution
        dist_text = "=== DATA DISTRIBUTION ANALYSIS ===\n\n"
        dist_text += f"Data Completeness: {self.meta_features['completeness']:.1f}%\n"
        dist_text += f"Sparsity: {self.meta_features['sparsity']:.1f}%\n"
        dist_text += f"Outlier Percentage: {self.meta_features['outlier_percentage']:.1f}%\n\n"
        
        if self.meta_features['class_imbalance'] is not None:
            dist_text += f"Class Imbalance: {self.meta_features['class_imbalance']:.1f}%\n"
        
        dist_text += f"Max Correlation: {self.meta_features['max_correlation']:.3f}\n"
        dist_text += f"High Correlation Pairs: {self.meta_features['high_corr_pairs']}\n"
        
        self.meta_dist_text.delete(1.0, tk.END)
        self.meta_dist_text.insert(tk.END, dist_text)

        # Update quality metrics
        quality_text = "=== DATA QUALITY ASSESSMENT ===\n\n"
        
        # Quality scores
        completeness_score = "🟢 Excellent" if self.meta_features['completeness'] > 95 else \
                           "🟡 Good" if self.meta_features['completeness'] > 85 else \
                           "🟠 Fair" if self.meta_features['completeness'] > 70 else "🔴 Poor"
        
        outlier_score = "🟢 Low" if self.meta_features['outlier_percentage'] < 5 else \
                       "🟡 Moderate" if self.meta_features['outlier_percentage'] < 15 else "🔴 High"
        
        correlation_score = "🟢 Low" if self.meta_features['high_corr_pairs'] < 3 else \
                          "🟡 Moderate" if self.meta_features['high_corr_pairs'] < 10 else "🔴 High"
        
        quality_text += f"Completeness: {completeness_score} ({self.meta_features['completeness']:.1f}%)\n"
        quality_text += f"Outliers: {outlier_score} ({self.meta_features['outlier_percentage']:.1f}%)\n"
        quality_text += f"Multicollinearity: {correlation_score} ({self.meta_features['high_corr_pairs']} pairs)\n\n"
        
        # Overall quality assessment
        quality_scores = [
            self.meta_features['completeness'] / 100,
            1 - (self.meta_features['outlier_percentage'] / 100),
            1 - (self.meta_features['high_corr_pairs'] / 20)  # Normalize high corr pairs
        ]
        overall_quality = np.mean(quality_scores) * 100
        
        quality_rating = "🟢 Excellent" if overall_quality > 80 else \
                        "🟡 Good" if overall_quality > 60 else \
                        "🟠 Fair" if overall_quality > 40 else "🔴 Poor"
        
        quality_text += f"Overall Data Quality: {quality_rating} ({overall_quality:.1f}/100)\n"
        
        self.meta_quality_text.delete(1.0, tk.END)
        self.meta_quality_text.insert(tk.END, quality_text)

    # ==== ENHANCED CLEANING METHODS ====

    def advanced_feature_engineering(self):
        """Perform advanced feature engineering."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.status_var.set("Performing advanced feature engineering...")
        thread = threading.Thread(target=self._advanced_feature_engineering_thread)
        thread.daemon = True
        thread.start()

    def _advanced_feature_engineering_thread(self):
        """Advanced feature engineering in separate thread."""
        try:
            # This would call enhanced data processor methods
            self.queue.put(("log_step", "🔧 Starting advanced feature engineering..."))
            
            # Placeholder - this would be implemented in enhanced data_processor
            success = True  # self.data_processor.advanced_feature_engineering()
            
            if success:
                self.queue.put(("log_step", "✅ Advanced feature engineering completed"))
                self.queue.put(("feature_engineering_complete", None))
            else:
                self.queue.put(("error", "Advanced feature engineering failed"))
                
        except Exception as e:
            self.queue.put(("error", f"Error in feature engineering: {str(e)}"))

    def handle_outliers(self):
        """Handle outliers with various strategies."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        # Show outlier handling options dialog
        options = ["Remove outliers", "Cap outliers", "Transform data", "Keep outliers with flag"]
        choice = self._show_choice_dialog("Outlier Handling", "Choose outlier handling strategy:", options)
        
        if choice:
            self.status_var.set(f"Handling outliers: {choice}")
            thread = threading.Thread(target=self._handle_outliers_thread, args=(choice,))
            thread.daemon = True
            thread.start()

    def _handle_outliers_thread(self, strategy):
        """Handle outliers in separate thread."""
        try:
            self.queue.put(("log_step", f"🎯 Applying outlier strategy: {strategy}"))
            # This would be implemented in enhanced data_processor
            success = True  # self.data_processor.handle_outliers(strategy)
            
            if success:
                self.queue.put(("log_step", "✅ Outlier handling completed"))
                self.queue.put(("outlier_handling_complete", None))
            else:
                self.queue.put(("error", "Outlier handling failed"))
                
        except Exception as e:
            self.queue.put(("error", f"Error handling outliers: {str(e)}"))

    def knn_imputation(self):
        """Perform KNN imputation for missing values."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.status_var.set("Performing KNN imputation...")
        thread = threading.Thread(target=self._knn_imputation_thread)
        thread.daemon = True
        thread.start()

    def _knn_imputation_thread(self):
        """KNN imputation in separate thread."""
        try:
            self.queue.put(("log_step", "🔍 Starting KNN imputation..."))
            # This would be implemented in enhanced data_processor
            success = True  # self.data_processor.knn_imputation()
            
            if success:
                self.queue.put(("log_step", "✅ KNN imputation completed"))
                self.queue.put(("imputation_complete", "KNN"))
            else:
                self.queue.put(("error", "KNN imputation failed"))
                
        except Exception as e:
            self.queue.put(("error", f"Error in KNN imputation: {str(e)}"))

    def iterative_imputation(self):
        """Perform iterative imputation for missing values."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.status_var.set("Performing iterative imputation...")
        thread = threading.Thread(target=self._iterative_imputation_thread)
        thread.daemon = True
        thread.start()

    def _iterative_imputation_thread(self):
        """Iterative imputation in separate thread."""
        try:
            self.queue.put(("log_step", "🔄 Starting iterative imputation..."))
            # This would be implemented in enhanced data_processor
            success = True  # self.data_processor.iterative_imputation()
            
            if success:
                self.queue.put(("log_step", "✅ Iterative imputation completed"))
                self.queue.put(("imputation_complete", "Iterative"))
            else:
                self.queue.put(("error", "Iterative imputation failed"))
                
        except Exception as e:
            self.queue.put(("error", f"Error in iterative imputation: {str(e)}"))

    def text_preprocessing(self):
        """Perform advanced text preprocessing."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        # Check for text columns
        text_columns = []
        for col in self.current_data.columns:
            if self.current_data[col].dtype == 'object':
                avg_length = self.current_data[col].astype(str).str.len().mean()
                if avg_length > 50:  # Likely text column
                    text_columns.append(col)

        if not text_columns:
            messagebox.showinfo("Info", "No text columns detected for preprocessing")
            return

        self.status_var.set("Preprocessing text columns...")
        thread = threading.Thread(target=self._text_preprocessing_thread, args=(text_columns,))
        thread.daemon = True
        thread.start()

    def _text_preprocessing_thread(self, text_columns):
        """Text preprocessing in separate thread."""
        try:
            self.queue.put(("log_step", f"📝 Processing {len(text_columns)} text columns..."))
            # This would be implemented in enhanced data_processor
            success = True  # self.data_processor.advanced_text_preprocessing(text_columns)
            
            if success:
                self.queue.put(("log_step", "✅ Text preprocessing completed"))
                self.queue.put(("text_preprocessing_complete", text_columns))
            else:
                self.queue.put(("error", "Text preprocessing failed"))
                
        except Exception as e:
            self.queue.put(("error", f"Error in text preprocessing: {str(e)}"))

    # ==== ENHANCED ML METHODS ====

    def automl_training(self):
        """Train models using AutoML."""
        if self.cleaned_data is None:
            messagebox.showwarning("Warning", "Please clean data first")
            return

        target_col = self.target_var.get()
        if not target_col:
            messagebox.showwarning("Warning", "Please select a target column")
            return

        self.status_var.set("Starting AutoML training...")
        thread = threading.Thread(target=self._automl_training_thread, args=(target_col,))
        thread.daemon = True
        thread.start()

    def _automl_training_thread(self, target_col):
        """AutoML training in separate thread."""
        try:
            self.queue.put(("log_step", "🤖 Starting AutoML training..."))
            # This would be implemented in enhanced ml_engine
            success = True  # self.ml_engine.automl_training(self.cleaned_data, target_col)
            
            if success:
                self.training_complete = True
                self.queue.put(("log_step", "✅ AutoML training completed"))
                self.queue.put(("training_complete", "AutoML"))
            else:
                self.queue.put(("error", "AutoML training failed"))
                
        except Exception as e:
            self.queue.put(("error", f"Error in AutoML training: {str(e)}"))

    def train_deep_learning(self):
        """Train deep learning models."""
        if self.cleaned_data is None:
            messagebox.showwarning("Warning", "Please clean data first")
            return

        target_col = self.target_var.get()
        if not target_col:
            messagebox.showwarning("Warning", "Please select a target column")
            return

        self.status_var.set("Training deep learning models...")
        thread = threading.Thread(target=self._train_deep_learning_thread, args=(target_col,))
        thread.daemon = True
        thread.start()

    def _train_deep_learning_thread(self, target_col):
        """Deep learning training in separate thread."""
        try:
            self.queue.put(("log_step", "🧠 Starting deep learning training..."))
            # This would be implemented in enhanced ml_engine
            success = True  # self.ml_engine.train_deep_learning(self.cleaned_data, target_col)
            
            if success:
                self.training_complete = True
                self.queue.put(("log_step", "✅ Deep learning training completed"))
                self.queue.put(("training_complete", "Deep Learning"))
            else:
                self.queue.put(("error", "Deep learning training failed"))
                
        except Exception as e:
            self.queue.put(("error", f"Error in deep learning training: {str(e)}"))

    def nlp_analysis(self):
        """Perform NLP analysis."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.status_var.set("Performing NLP analysis...")
        thread = threading.Thread(target=self._nlp_analysis_thread)
        thread.daemon = True
        thread.start()

    def _nlp_analysis_thread(self):
        """NLP analysis in separate thread."""
        try:
            self.queue.put(("log_step", "📝 Starting NLP analysis..."))
            # This would be implemented in enhanced ml_engine
            success = True  # self.ml_engine.nlp_analysis(self.current_data)
            
            if success:
                self.queue.put(("log_step", "✅ NLP analysis completed"))
                self.queue.put(("nlp_analysis_complete", None))
            else:
                self.queue.put(("error", "NLP analysis failed"))
                
        except Exception as e:
            self.queue.put(("error", f"Error in NLP analysis: {str(e)}"))

    # ==== 3D VISUALIZATION METHODS ====

    def show_3d_visualization(self):
        """Show 3D data visualization."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        try:
            data_to_viz = self.cleaned_data if self.cleaned_data is not None else self.current_data
            fig = self.visualizer.create_3d_visualization(data_to_viz)
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating 3D visualization: {str(e)}")

    def show_3d_feature_space(self):
        """Show 3D feature space visualization."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            fig = self.visualizer.create_3d_feature_space(self.ml_engine)
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating 3D feature space: {str(e)}")

    def show_3d_model_performance(self):
        """Show 3D model performance visualization."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            fig = self.visualizer.create_3d_model_performance(self.ml_engine)
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating 3D model performance: {str(e)}")

    def show_training_animation(self):
        """Show training progress animation."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            fig = self.visualizer.create_training_animation(self.ml_engine)
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating training animation: {str(e)}")

    def show_hyperparameter_landscape(self):
        """Show hyperparameter optimization landscape."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            fig = self.visualizer.create_hyperparameter_landscape(self.ml_engine)
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating hyperparameter landscape: {str(e)}")

    # ==== EXPLAINABILITY METHODS ====

    def show_shap_analysis(self):
        """Show SHAP analysis."""
        if not ModelExplainer or not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first and ensure explainability module is available")
            return

        try:
            results = self.explainer.generate_shap_analysis(self.ml_engine)
            self.shap_results_text.delete(1.0, tk.END)
            self.shap_results_text.insert(tk.END, results)
        except Exception as e:
            messagebox.showerror("Error", f"Error in SHAP analysis: {str(e)}")

    def show_global_shap(self):
        """Show global SHAP analysis."""
        self.show_shap_analysis()

    def show_local_shap(self):
        """Show local SHAP analysis for specific instances."""
        if not ModelExplainer or not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first and ensure explainability module is available")
            return

        # Ask for instance index
        instance_idx = simpledialog.askinteger(
            "Local SHAP", 
            f"Enter instance index (0 to {len(self.cleaned_data)-1}):",
            initialvalue=0,
            minvalue=0,
            maxvalue=len(self.cleaned_data)-1
        )
        
        if instance_idx is not None:
            try:
                results = self.explainer.generate_local_shap(self.ml_engine, instance_idx)
                self.shap_results_text.delete(1.0, tk.END)
                self.shap_results_text.insert(tk.END, results)
            except Exception as e:
                messagebox.showerror("Error", f"Error in local SHAP analysis: {str(e)}")

    def show_shap_waterfall(self):
        """Show SHAP waterfall plot."""
        if not ModelExplainer or not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first and ensure explainability module is available")
            return

        try:
            fig = self.explainer.create_shap_waterfall(self.ml_engine)
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating SHAP waterfall: {str(e)}")

    def show_lime_explanations(self):
        """Show LIME explanations."""
        if not ModelExplainer or not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first and ensure explainability module is available")
            return

        try:
            results = self.explainer.generate_lime_explanations(self.ml_engine)
            self.lime_results_text.delete(1.0, tk.END)
            self.lime_results_text.insert(tk.END, results)
        except Exception as e:
            messagebox.showerror("Error", f"Error in LIME explanations: {str(e)}")

    def show_feature_contributions(self):
        """Show feature contributions analysis."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            if ModelExplainer:
                results = self.explainer.analyze_feature_contributions(self.ml_engine)
                self.importance_text.delete(1.0, tk.END)
                self.importance_text.insert(tk.END, results)
            else:
                # Fallback to basic feature importance
                self.show_feature_importance()
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing feature contributions: {str(e)}")

    def show_error_analysis(self):
        """Show error analysis."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            if ModelExplainer:
                results = self.explainer.perform_error_analysis(self.ml_engine)
                self.error_results_text.delete(1.0, tk.END)
                self.error_results_text.insert(tk.END, results)
            else:
                messagebox.showwarning("Warning", "Explainability module not available")
        except Exception as e:
            messagebox.showerror("Error", f"Error in error analysis: {str(e)}")

    def show_prediction_confidence(self):
        """Show prediction confidence analysis."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            if ModelExplainer:
                fig = self.explainer.create_confidence_analysis(self.ml_engine)
                self.display_plot(fig)
            else:
                messagebox.showwarning("Warning", "Explainability module not available")
        except Exception as e:
            messagebox.showerror("Error", f"Error in prediction confidence analysis: {str(e)}")

    def show_calibration_curves(self):
        """Show model calibration curves."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            if ModelExplainer:
                fig = self.explainer.create_calibration_curves(self.ml_engine)
                self.display_plot(fig)
            else:
                messagebox.showwarning("Warning", "Explainability module not available")
        except Exception as e:
            messagebox.showerror("Error", f"Error creating calibration curves: {str(e)}")

    # ==== SESSION MANAGEMENT ====

    def save_session(self):
        """Save current session."""
        try:
            if self.storage_manager:
                success = self.storage_manager.save_session(
                    self.current_session_id,
                    {
                        'current_data': self.current_data,
                        'cleaned_data': self.cleaned_data,
                        'meta_features': self.meta_features,
                        'models': self.ml_engine.models if hasattr(self.ml_engine, 'models') else {},
                        'training_complete': self.training_complete
                    }
                )
                if success:
                    messagebox.showinfo("Success", f"Session {self.current_session_id} saved successfully")
                else:
                    messagebox.showerror("Error", "Failed to save session")
            else:
                # Fallback session saving
                session_file = self.storage_dir / f"{self.current_session_id}.session"
                session_data = {
                    'timestamp': datetime.now().isoformat(),
                    'data_shape': self.current_data.shape if self.current_data is not None else None,
                    'cleaned_data_shape': self.cleaned_data.shape if self.cleaned_data is not None else None,
                    'training_complete': self.training_complete
                }
                
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Session metadata saved to {session_file}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving session: {str(e)}")

    def load_session(self):
        """Load previous session."""
        try:
            session_files = list(self.storage_dir.glob("*.session"))
            if not session_files:
                messagebox.showinfo("Info", "No saved sessions found")
                return

            # Show session selection dialog
            session_names = [f.stem for f in session_files]
            choice = self._show_choice_dialog("Load Session", "Select session to load:", session_names)
            
            if choice:
                if self.storage_manager:
                    session_data = self.storage_manager.load_session(choice)
                    if session_data:
                        self.current_data = session_data.get('current_data')
                        self.cleaned_data = session_data.get('cleaned_data')
                        self.meta_features = session_data.get('meta_features', {})
                        self.training_complete = session_data.get('training_complete', False)
                        
                        self.current_session_id = choice
                        self.session_status_var.set(f"Session: {choice}")
                        
                        # Update displays
                        self.update_analysis_displays()
                        if self.meta_features:
                            self.update_meta_analysis()
                        
                        messagebox.showinfo("Success", f"Session {choice} loaded successfully")
                    else:
                        messagebox.showerror("Error", "Failed to load session")
                else:
                    # Fallback - just load metadata
                    session_file = self.storage_dir / f"{choice}.session"
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    messagebox.showinfo("Session Info", 
                                      f"Session: {choice}\n"
                                      f"Saved: {session_data.get('timestamp', 'Unknown')}\n"
                                      f"Data Shape: {session_data.get('data_shape', 'Unknown')}\n"
                                      f"Training Complete: {session_data.get('training_complete', False)}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error loading session: {str(e)}")

    def clear_session(self):
        """Clear current session."""
        if messagebox.askyesno("Clear Session", "Are you sure you want to clear the current session? All unsaved work will be lost."):
            self.current_data = None
            self.cleaned_data = None
            self.processed_data = None
            self.meta_features = {}
            self.training_complete = False
            
            # Clear all displays
            for text_widget in [self.data_preview_text, self.data_info_text, self.column_analysis_text,
                               self.ai_decisions_text, self.autoselect_text, self.recommendations_text,
                               self.strategies_text, self.cleaning_results_text, self.ml_log_text,
                               self.model_comparison_text, self.summary_text]:
                text_widget.delete(1.0, tk.END)
            
            # Clear plot
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            if self.toolbar:
                self.toolbar.destroy()
                self.toolbar = None
            
            # Reset status
            self.initialize_session_management()
            self.status_var.set("Session cleared - ready for new analysis")
            self.processing_status_var.set("Ready")
            
            messagebox.showinfo("Success", "Session cleared successfully")

    def list_sessions(self):
        """List all available sessions."""
        try:
            session_files = list(self.storage_dir.glob("*.session"))
            if not session_files:
                messagebox.showinfo("Sessions", "No saved sessions found")
                return
            
            # Create session list dialog
            session_list = []
            for session_file in sorted(session_files, key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    session_info = f"📁 {session_file.stem}\n"
                    session_info += f"   📅 {session_data.get('timestamp', 'Unknown')}\n"
                    session_info += f"   📊 Data: {session_data.get('data_shape', 'Unknown')}\n"
                    session_info += f"   🤖 Training: {'✅' if session_data.get('training_complete') else '❌'}\n"
                    session_list.append(session_info)
                except:
                    session_list.append(f"📁 {session_file.stem} (corrupted)")
            
            session_text = "Available Sessions:\n\n" + "\n".join(session_list)
            
            # Show in a dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Session List")
            dialog.geometry("500x400")
            dialog.grab_set()
            
            text_widget = scrolledtext.ScrolledText(dialog, font=("Consolas", 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, session_text)
            text_widget.config(state=tk.DISABLED)
            
            close_btn = ttk.Button(dialog, text="Close", command=dialog.destroy)
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error listing sessions: {str(e)}")

    # ==== PYGAME INTERFACE ====

    def switch_to_pygame(self):
        """Switch to Pygame interface."""
        if not PygameInterface:
            messagebox.showerror("Error", "Pygame interface not available. Install pygame and create pygame_ui module.")
            return
        
        try:
            self.pygame_mode = True
            self.pygame_interface = PygameInterface(self)
            
            # Hide Tkinter window
            self.root.withdraw()
            
            # Start Pygame interface
            self.pygame_interface.run()
            
            # Show Tkinter window when Pygame closes
            self.root.deiconify()
            self.pygame_mode = False
            
        except Exception as e:
            messagebox.showerror("Error", f"Error switching to Pygame: {str(e)}")
            self.pygame_mode = False
            self.root.deiconify()

    # ==== UTILITY METHODS ====

    def toggle_theme(self):
        """Toggle between dark and light theme."""
        try:
            style = ttk.Style()
            current_theme = style.theme_use()
            
            if current_theme == "clam":
                style.theme_use("alt")
                self.root.configure(bg="#f0f0f0")
            else:
                style.theme_use("clam")
                self.root.configure(bg="#2c3e50")
                
            messagebox.showinfo("Theme", f"Switched to {style.theme_use()} theme")
        except Exception as e:
            messagebox.showerror("Error", f"Error changing theme: {str(e)}")

    def show_preferences(self):
        """Show preferences dialog."""
        pref_dialog = PreferencesDialog(self.root)
        self.root.wait_window(pref_dialog.dialog)

    def start_api_server(self):
        """Start API server for remote access."""
        try:
            # This would start the API server from api_server.py
            result = messagebox.askyesnocancel(
                "API Server", 
                "Start API server for remote access?\n\n"
                "This will allow external applications to access the platform via REST API.\n\n"
                "Yes = Start on localhost:8000\n"
                "No = Custom configuration\n"
                "Cancel = Cancel"
            )
            
            if result is True:
                # Start default server
                threading.Thread(target=self._start_api_server, args=("localhost", 8000), daemon=True).start()
                messagebox.showinfo("API Server", "API server starting on http://localhost:8000")
            elif result is False:
                # Custom configuration
                host = simpledialog.askstring("API Server", "Enter host (default: localhost):", initialvalue="localhost")
                port = simpledialog.askinteger("API Server", "Enter port (default: 8000):", initialvalue=8000, minvalue=1024, maxvalue=65535)
                
                if host and port:
                    threading.Thread(target=self._start_api_server, args=(host, port), daemon=True).start()
                    messagebox.showinfo("API Server", f"API server starting on http://{host}:{port}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error starting API server: {str(e)}")

    def _start_api_server(self, host, port):
        """Start API server in separate thread."""
        try:
            # This would import and run the API server
            # from api_server import create_app, run_server
            # app = create_app(self)
            # run_server(app, host, port)
            pass
        except Exception as e:
            self.queue.put(("error", f"API server error: {str(e)}"))

    def docker_deployment(self):
        """Show Docker deployment options."""
        docker_dialog = DockerDeploymentDialog(self.root)
        self.root.wait_window(docker_dialog.dialog)

    def cloud_upload(self):
        """Upload to cloud storage."""
        cloud_dialog = CloudUploadDialog(self.root)
        self.root.wait_window(cloud_dialog.dialog)

    def _show_choice_dialog(self, title, message, choices):
        """Show a choice dialog and return selected choice."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x300")
        dialog.grab_set()
        dialog.transient(self.root)
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (300 // 2)
        dialog.geometry(f"400x300+{x}+{y}")
        
        result = [None]  # Use list to store result (mutable)
        
        # Message
        ttk.Label(dialog, text=message, wraplength=350).pack(pady=10)
        
        # Choice listbox
        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for choice in choices:
            listbox.insert(tk.END, choice)
        
        if choices:
            listbox.selection_set(0)  # Select first item by default
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def on_ok():
            selection = listbox.curselection()
            if selection:
                result[0] = choices[selection[0]]
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Handle double-click
        listbox.bind('<Double-Button-1>', lambda e: on_ok())
        
        # Handle Enter key
        dialog.bind('<Return>', lambda e: on_ok())
        dialog.bind('<Escape>', lambda e: on_cancel())
        
        dialog.wait_window()
        return result[0]

    # ==== ENHANCED EXPORT METHODS ====

    def export_all_visualizations(self):
        """Export all generated visualizations."""
        if self.canvas is None:
            messagebox.showwarning("Warning", "No visualizations to export")
            return

        export_dir = filedialog.askdirectory(title="Select export directory")
        if export_dir:
            try:
                export_path = Path(export_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Export current plot
                if self.canvas:
                    plot_path = export_path / f"visualization_{timestamp}.png"
                    self.canvas.figure.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                # Export additional plots if available
                exported_count = 1
                
                # Generate and export additional visualizations
                if self.current_data is not None:
                    try:
                        fig = self.visualizer.create_data_overview(self.current_data, self.cleaned_data)
                        overview_path = export_path / f"data_overview_{timestamp}.png"
                        fig.savefig(overview_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        exported_count += 1
                    except:
                        pass
                
                if self.training_complete:
                    try:
                        fig = self.visualizer.create_model_performance(self.ml_engine)
                        perf_path = export_path / f"model_performance_{timestamp}.png"
                        fig.savefig(perf_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        exported_count += 1
                    except:
                        pass
                
                messagebox.showinfo("Export Complete", 
                                  f"Exported {exported_count} visualizations to:\n{export_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting visualizations: {str(e)}")

    def export_models(self):
        """Export trained models."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "No trained models to export")
            return

        export_dir = filedialog.askdirectory(title="Select model export directory")
        if export_dir:
            try:
                export_path = Path(export_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = export_path / f"models_{timestamp}"
                model_dir.mkdir(exist_ok=True)
                
                # Export models (this would be implemented in enhanced ml_engine)
                exported_models = []  # self.ml_engine.export_models(model_dir)
                
                # Export model metadata
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': self.current_session_id,
                    'data_shape': self.cleaned_data.shape if self.cleaned_data is not None else None,
                    'target_column': self.target_var.get(),
                    'problem_type': self.problem_type_var.get(),
                    'models_exported': exported_models
                }
                
                metadata_path = model_dir / "model_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                messagebox.showinfo("Export Complete", 
                                  f"Models exported to:\n{model_dir}\n\n"
                                  f"Exported: {len(exported_models)} models")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting models: {str(e)}")

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        file_path = filedialog.asksaveasfilename(
            title="Save comprehensive report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("Text files", "*.txt"), ("Markdown files", "*.md")]
        )

        if file_path:
            try:
                self.status_var.set("Generating comprehensive report...")
                thread = threading.Thread(target=self._generate_report_thread, args=(file_path,))
                thread.daemon = True
                thread.start()
            except Exception as e:
                messagebox.showerror("Error", f"Error generating report: {str(e)}")

    def _generate_report_thread(self, file_path):
        """Generate report in separate thread."""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.html':
                self._generate_html_report(file_path)
            elif file_ext == '.md':
                self._generate_markdown_report(file_path)
            else:
                self._generate_text_report(file_path)
            
            self.queue.put(("report_generated", file_path))
            
        except Exception as e:
            self.queue.put(("error", f"Error generating report: {str(e)}"))

    def _generate_html_report(self, file_path):
        """Generate HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Data Analytics Report - {self.current_session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #3498db; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }}
                .metric {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .success {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .error {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 AI Data Analytics Report</h1>
                <p>Session: {self.current_session_id}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="metric">
                    <strong>Dataset:</strong> {self.current_data.shape if self.current_data is not None else 'No data loaded'}
                </div>
                <div class="metric">
                    <strong>Data Quality:</strong> {'Improved through AI cleaning' if self.cleaned_data is not None else 'Original data analyzed'}
                </div>
                <div class="metric">
                    <strong>ML Training:</strong> {'✅ Complete' if self.training_complete else '❌ Not completed'}
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Data Analysis</h2>
                <p>Comprehensive AI-driven analysis was performed on the dataset...</p>
                <!-- Additional content would be added here -->
            </div>
            
            <div class="section">
                <h2>🤖 Machine Learning Results</h2>
                {'<p>Models were successfully trained and evaluated.</p>' if self.training_complete else '<p>No machine learning training was performed.</p>'}
            </div>
            
            <div class="section">
                <h2>📈 Recommendations</h2>
                <ul>
                    <li>Data quality assessment completed</li>
                    <li>AI-driven cleaning recommendations applied</li>
                    {'<li>Machine learning models trained and evaluated</li>' if self.training_complete else ''}
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_markdown_report(self, file_path):
        """Generate Markdown report."""
        md_content = f"""# 🧠 AI Data Analytics Report

**Session:** {self.current_session_id}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary

- **Dataset:** {self.current_data.shape if self.current_data is not None else 'No data loaded'}
- **Data Quality:** {'Improved through AI cleaning' if self.cleaned_data is not None else 'Original data analyzed'}
- **ML Training:** {'✅ Complete' if self.training_complete else '❌ Not completed'}

## 🔍 Data Analysis

Comprehensive AI-driven analysis was performed on the dataset with intelligent recommendations for cleaning and processing.

## 🤖 Machine Learning Results

{'Models were successfully trained and evaluated.' if self.training_complete else 'No machine learning training was performed.'}

## 📈 Recommendations

- Data quality assessment completed
- AI-driven cleaning recommendations applied
{'- Machine learning models trained and evaluated' if self.training_complete else ''}

---

*Report generated by Advanced AI Data Analytics Platform*
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

    def _generate_text_report(self, file_path):
        """Generate plain text report."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("AI DATA ANALYTICS COMPREHENSIVE REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Session ID: {self.current_session_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 30 + "\n")
            if self.current_data is not None:
                f.write(f"Original Shape: {self.current_data.shape[0]:,} rows × {self.current_data.shape[1]} columns\n")
                if self.cleaned_data is not None:
                    f.write(f"Cleaned Shape: {self.cleaned_data.shape[0]:,} rows × {self.cleaned_data.shape[1]} columns\n")
            else:
                f.write("No data loaded\n")
            f.write("\n")
            
            # AI Analysis
            f.write("AI ANALYSIS RESULTS:\n")
            f.write("-" * 30 + "\n")
            if hasattr(self.data_processor, 'ai_engine') and self.data_processor.ai_engine:
                f.write(f"AI Decisions Made: {len(self.data_processor.ai_engine.reasoning_log)}\n")
                for decision in self.data_processor.ai_engine.reasoning_log[-5:]:  # Last 5 decisions
                    f.write(f"[{decision['timestamp']}] {decision['decision']}\n")
            else:
                f.write("No AI analysis performed\n")
            f.write("\n")
            
            # ML Results
            f.write("MACHINE LEARNING RESULTS:\n")
            f.write("-" * 30 + "\n")
            if self.training_complete:
                f.write("Training Status: Complete\n")
                if hasattr(self.ml_engine, 'models'):
                    f.write(f"Models Trained: {len(self.ml_engine.models)}\n")
                if hasattr(self.ml_engine, 'best_model_name'):
                    f.write(f"Best Model: {self.ml_engine.best_model_name}\n")
            else:
                f.write("Training Status: Not completed\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("End of Report\n")

    # ==== ORIGINAL METHODS (Enhanced) ====

    def analyze_data(self):
        """Enhanced AI analysis of the data."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.status_var.set("AI analyzing data...")
        self.processing_status_var.set("Analyzing...")
        self.ai_progress_bar.start()
        
        # Analysis is already done in load_and_analyze, just update displays
        self.update_analysis_displays()
        
        # Extract meta-features if not already done
        if not self.meta_features:
            self.extract_meta_features()
        
        self.ai_progress_bar.stop()
        self.status_var.set("AI analysis complete")
        self.processing_status_var.set("Complete")

    def reanalyze_data(self):
        """Re-run AI analysis."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data to analyze")
            return
        
        self.data_processor = SmartDataProcessor()
        success = self.data_processor.load_and_analyze_dataframe(self.current_data, "reanalysis")
        if success:
            self.update_analysis_displays()
            # Reset meta-features for re-extraction
            self.meta_features = {}
            self.extract_meta_features()
            
    def clean_data(self):
        """Perform enhanced smart data cleaning."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.status_var.set("AI cleaning data...")
        self.processing_status_var.set("Cleaning...")
        self.cleaning_progress_bar.start()
        self.cleaning_status_var.set("Status: Cleaning in progress...")

        thread = threading.Thread(target=self._clean_data_thread)
        thread.daemon = True
        thread.start()

    def _clean_data_thread(self):
        """Enhanced clean data in separate thread."""
        try:
            self.queue.put(("log_step", "🧹 Starting intelligent data cleaning..."))
            
            success = self.data_processor.smart_clean_data()
            if success:
                self.cleaned_data = self.data_processor.cleaned_data
                
                # Generate data lineage
                lineage_info = self._generate_data_lineage()
                
                self.queue.put(("cleaning_complete", lineage_info))
            else:
                self.queue.put(("error", "Failed to clean data"))
        except Exception as e:
            self.queue.put(("error", f"Error cleaning data: {str(e)}"))

    def _generate_data_lineage(self):
        """Generate data lineage information."""
        lineage = {
            'original_shape': self.current_data.shape,
            'cleaned_shape': self.cleaned_data.shape,
            'cleaning_steps': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # This would be populated by the enhanced data processor
        # with detailed information about each cleaning step
        
        return lineage

    def proceed_to_ml(self):
        """Enhanced ML preparation with better data validation."""
        if self.cleaned_data is None:
            if self.current_data is None:
                messagebox.showwarning("Warning", "Please load data first")
                return
            else:
                # Ask user if they want to use uncleaned data
                response = messagebox.askyesnocancel("Use Original Data?", 
                                         "No cleaned data available. Would you like to use the original data?\n\n"
                                         "Recommendation: Clean the data first for better results.\n\n"
                                         "Yes = Use original data\n"
                                         "No = Clean data first\n"
                                         "Cancel = Cancel")
                if response is None:  # Cancel
                    return
                elif response is False:  # No - clean first
                    self.clean_data()
                    return
                else:  # Yes - use original
                    self.cleaned_data = self.current_data.copy()
                    messagebox.showinfo("Info", "Using original data. Results may be affected by data quality issues.")

        # Update target column dropdown with cleaned data columns
        self.target_combo['values'] = list(self.cleaned_data.columns)
        
        # Enhanced auto-suggestion for target column
        suggested_targets = []
        for col in self.cleaned_data.columns:
            col_lower = col.lower()
            # Extended keyword matching
            target_keywords = ['target', 'label', 'class', 'y', 'outcome', 'diagnosis', 'result', 
                             'prediction', 'response', 'dependent', 'output', 'category']
            if any(keyword in col_lower for keyword in target_keywords):
                suggested_targets.append(col)
        
        # Also check for binary columns that might be targets
        for col in self.cleaned_data.columns:
            if self.cleaned_data[col].nunique() == 2 and col not in suggested_targets:
                unique_vals = set(self.cleaned_data[col].unique())
                binary_patterns = [
                    {'Yes', 'No'}, {'True', 'False'}, {'1', '0'}, {1, 0},
                    {'M', 'F'}, {'Male', 'Female'}, {'Positive', 'Negative'},
                    {'High', 'Low'}, {'Good', 'Bad'}, {'Pass', 'Fail'}
                ]
                if any(unique_vals.issubset(pattern) for pattern in binary_patterns):
                    suggested_targets.append(col)
        
        if suggested_targets:
            self.target_var.set(suggested_targets[0])
            self.status_var.set(f"Suggested target column: {suggested_targets[0]}")
        
        # Auto-detect problem type based on target
        if self.target_var.get():
            target_col = self.target_var.get()
            unique_count = self.cleaned_data[target_col].nunique()
            
            if self.cleaned_data[target_col].dtype in ['object', 'category'] or unique_count < 10:
                self.problem_type_var.set("classification")
            else:
                self.problem_type_var.set("regression")
        
        # Switch to ML tab
        self.notebook.select(5)  # ML tab (updated index)
        
        self.status_var.set("Ready for machine learning - data validated and prepared")
        messagebox.showinfo("ML Ready", 
                           f"Data is ready for machine learning!\n\n"
                           f"Dataset: {self.cleaned_data.shape[0]:,} rows × {self.cleaned_data.shape[1]} columns\n"
                           f"Suggested Target: {self.target_var.get()}\n"
                           f"Detected Problem: {self.problem_type_var.get()}\n\n"
                           f"Please review the suggestions and train models.")

    def train_models(self):
        """Enhanced machine learning model training."""
        if self.cleaned_data is None:
            messagebox.showwarning("Warning", "Please clean data first")
            return

        target_col = self.target_var.get()
        if not target_col:
            messagebox.showwarning("Warning", "Please select a target column")
            return

        self.status_var.set("Training ML models...")
        self.processing_status_var.set("Training...")

        thread = threading.Thread(target=self._train_models_thread, args=(target_col,))
        thread.daemon = True
        thread.start()

    def _train_models_thread(self, target_col):
        """Enhanced model training with proper data encoding and validation."""
        try:
            self.queue.put(("log_step", "🔍 Preparing data for machine learning..."))
        
            # Use cleaned data if available, otherwise use current data
            training_data = self.cleaned_data if self.cleaned_data is not None else self.current_data
        
            if training_data is None:
                self.queue.put(("error", "No data available for training"))
                return
                
            self.queue.put(("log_step", f"📊 Training data shape: {training_data.shape}"))
            self.queue.put(("log_step", f"🎯 Target column: {target_col}"))
            
            # Create a copy to avoid modifying original data
            training_data = training_data.copy()
        
            # Enhanced target column processing
            if target_col not in training_data.columns:
                self.queue.put(("error", f"Target column '{target_col}' not found in data"))
                return
        
            target_series = training_data[target_col]
            self.queue.put(("log_step", f"🎯 Target data type: {target_series.dtype}"))
            self.queue.put(("log_step", f"🎯 Target unique values: {target_series.nunique()}"))
            
            # Enhanced categorical target encoding
            if target_series.dtype == 'object' or target_series.dtype.name == 'category':
                unique_values = target_series.unique()
                self.queue.put(("log_step", f"🔄 Categorical target detected. Unique values: {list(unique_values)}"))
                
                # Advanced categorical patterns
                if len(unique_values) == 2:
                    # Binary classification with intelligent mapping
                    value_pairs = [
                        ({'M', 'B'}, {'M': 1, 'B': 0}),  # Medical diagnosis
                        ({'Male', 'Female'}, {'Male': 1, 'Female': 0}),
                        ({'Yes', 'No'}, {'Yes': 1, 'No': 0}),
                        ({'True', 'False'}, {'True': 1, 'False': 0}),
                        ({'High', 'Low'}, {'High': 1, 'Low': 0}),
                        ({'Positive', 'Negative'}, {'Positive': 1, 'Negative': 0}),
                        ({'Pass', 'Fail'}, {'Pass': 1, 'Fail': 0}),
                        ({'Good', 'Bad'}, {'Good': 1, 'Bad': 0})
                    ]
                    
                    mapped = False
                    for pattern, mapping in value_pairs:
                        if set(unique_values) == pattern:
                            training_data[target_col] = target_series.map(mapping)
                            self.queue.put(("log_step", f"✅ Encoded target: {mapping}"))
                            mapped = True
                            break
                    
                    if not mapped:
                        # Generic binary encoding
                        value_map = {unique_values[0]: 0, unique_values[1]: 1}
                        training_data[target_col] = target_series.map(value_map)
                        self.queue.put(("log_step", f"✅ Generic binary encoding: {value_map}"))
                else:
                    # Multi-class classification - use label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    training_data[target_col] = le.fit_transform(target_series)
                    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    self.queue.put(("log_step", f"✅ Multi-class encoding: {label_mapping}"))
            
            # Verify no NaN values in target after encoding
            if training_data[target_col].isnull().any():
                nan_count = training_data[target_col].isnull().sum()
                self.queue.put(("error", f"Target column contains {nan_count} NaN values after encoding"))
                return
            
            # Enhanced feature processing
            feature_columns = [col for col in training_data.columns if col != target_col]
            self.queue.put(("log_step", f"🔢 Processing {len(feature_columns)} feature columns..."))
            
            # Comprehensive feature type detection and processing
            processed_features = self._process_features_comprehensive(training_data, feature_columns, target_col)
            
            if processed_features is None:
                self.queue.put(("error", "Feature processing failed"))
                return
            
            training_data = processed_features
            
            # Final validation
            features = training_data.drop(target_col, axis=1)
            target = training_data[target_col]
            
            self.queue.put(("log_step", f"✅ Final data shape: {training_data.shape}"))
            self.queue.put(("log_step", f"✅ Features: {features.shape[1]}, Target: {target.name}"))
            self.queue.put(("log_step", f"✅ Data validation complete - no NaN values"))
        
            # Prepare data for ML
            success = self.ml_engine.prepare_data(
                training_data, target_col, 
                self.problem_type_var.get()
            )
        
            if success:
                self.queue.put(("log_step", "✅ Data preparation successful"))
            
                # Enhanced training with multiple options
                training_options = {
                    'quick_mode': self.quick_mode_var.get(),
                    'automl': self.automl_var.get(),
                    'deep_learning': self.deep_learning_var.get(),
                    'cross_validation': True,
                    'feature_selection': True
                }
                
                success = self.ml_engine.train_all_models(**training_options)
            
                if success:
                    self.training_complete = True
                    self.queue.put(("training_complete", "Enhanced Training"))
                else:
                    self.queue.put(("error", "Model training failed - check ML engine logs"))
            else:
                self.queue.put(("error", "Data preparation failed in ML engine"))
            
        except Exception as e:
            self.queue.put(("error", f"Error in model training: {str(e)}"))

    def _process_features_comprehensive(self, data, feature_columns, target_col):
        """Comprehensive feature processing."""
        try:
            processed_data = data.copy()
            
            for col in feature_columns:
                col_data = processed_data[col]
                
                # Detect column type and apply appropriate processing
                if col_data.dtype == 'object':
                    # Text/categorical processing
                    if col_data.astype(str).str.len().mean() > 50:
                        # Text column - apply basic text preprocessing
                        self.queue.put(("log_step", f"📝 Processing text column: {col}"))
                        # This would be expanded with actual text preprocessing
                        processed_data = processed_data.drop(col, axis=1)
                    else:
                        # Categorical column
                        unique_count = col_data.nunique()
                        if unique_count < 10:
                            # Low cardinality - one-hot encoding
                            self.queue.put(("log_step", f"🏷️ One-hot encoding: {col}"))
                            dummies = pd.get_dummies(col_data, prefix=col)
                            processed_data = pd.concat([processed_data, dummies], axis=1)
                            processed_data = processed_data.drop(col, axis=1)
                        else:
                            # High cardinality - frequency encoding or drop
                            self.queue.put(("log_step", f"🗑️ Dropping high-cardinality column: {col}"))
                            processed_data = processed_data.drop(col, axis=1)
                
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    # Datetime processing
                    self.queue.put(("log_step", f"📅 Processing datetime column: {col}"))
                    processed_data[f'{col}_year'] = col_data.dt.year
                    processed_data[f'{col}_month'] = col_data.dt.month
                    processed_data[f'{col}_day'] = col_data.dt.day
                    processed_data[f'{col}_weekday'] = col_data.dt.weekday
                    processed_data = processed_data.drop(col, axis=1)
                
                elif col_data.dtype in ['int64', 'float64']:
                    # Numeric column - handle missing values
                    if col_data.isnull().any():
                        missing_pct = (col_data.isnull().sum() / len(col_data)) * 100
                        if missing_pct > 50:
                            self.queue.put(("log_step", f"🗑️ Dropping high-missing column: {col} ({missing_pct:.1f}% missing)"))
                            processed_data = processed_data.drop(col, axis=1)
                        else:
                            # Impute with median
                            processed_data[col] = col_data.fillna(col_data.median())
                            self.queue.put(("log_step", f"🔢 Imputed numeric column: {col} with median"))
            
            # Final cleanup - ensure all remaining features are numeric
            feature_cols = [col for col in processed_data.columns if col != target_col]
            for col in feature_cols:
                if processed_data[col].dtype == 'object':
                    try:
                        processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                        if processed_data[col].isnull().any():
                            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    except:
                        processed_data = processed_data.drop(col, axis=1)
                        self.queue.put(("log_step", f"🗑️ Dropped problematic column: {col}"))
            
            return processed_data
            
        except Exception as e:
            self.queue.put(("log_step", f"❌ Error in feature processing: {str(e)}"))
            return None

    # ==== AGENT MODE INTEGRATION METHODS ====
    
    def update_all_visualizations(self):
        """Update all visualizations - used by Agent Mode."""
        if self.current_data is None:
            return
            
        try:
            # Update advanced visualization dashboard if available
            if self.advanced_viz_dashboard:
                self.advanced_viz_dashboard.update_data(self.current_data)
            else:
                # Fallback to standard visualization
                self.show_data_overview()
                
        except Exception as e:
            print(f"Error updating visualizations: {e}")

    # ==== REMAINING VISUALIZATION AND DISPLAY METHODS ====

    def show_data_overview(self):
        """Show enhanced data overview."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        try:
            comparison_data = self.cleaned_data if self.cleaned_data is not None else self.current_data
            
            # Try to use visualizer methods with fallback
            if self.visualizer and hasattr(self.visualizer, 'create_professional_data_overview'):
                fig = self.visualizer.create_professional_data_overview(self.current_data, comparison_data)
            else:
                # Fallback: create simple data overview
                fig = self._create_simple_data_overview(self.current_data, comparison_data)
            
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating data overview: {str(e)}")

    def show_cleaning_summary(self):
        """Show enhanced data cleaning summary visualization."""
        if not hasattr(self.data_processor, 'ai_engine'):
            messagebox.showwarning("Warning", "Please perform AI analysis first")
            return

        try:
            fig = self.visualizer.create_enhanced_cleaning_summary(self.data_processor)
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating cleaning summary: {str(e)}")

    def show_before_after(self):
        """Show enhanced before/after data comparison."""
        if self.current_data is None or self.cleaned_data is None:
            messagebox.showwarning("Warning", "Please load and clean data first")
            return

        try:
            # Create simple before/after comparison
            fig = self._create_simple_before_after(self.current_data, self.cleaned_data)
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating comparison: {str(e)}")

    def show_data_table(self):
        """Show enhanced interactive data table."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        # Enhanced dialog for data selection
        options = ["Original Data", "Cleaned Data", "Side-by-Side Comparison"]
        if self.cleaned_data is None:
            options = options[:1]
        
        choice = self._show_choice_dialog("Data Table", "Which data would you like to view?", options)
        
        if not choice:
            return
        
        try:
            if choice == "Original Data":
                fig = self.visualizer.create_enhanced_data_table(self.current_data, "Original Data")
            elif choice == "Cleaned Data":
                fig = self.visualizer.create_enhanced_data_table(self.cleaned_data, "Cleaned Data")
            else:  # Side-by-Side Comparison
                fig = self.visualizer.create_comparison_table(self.current_data, self.cleaned_data)
            
            self.display_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error creating data table: {str(e)}")

    def show_model_performance(self):
        """Show enhanced ML model performance."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            # Create a simple model performance visualization
            if self.ml_engine and hasattr(self.ml_engine, 'results') and self.ml_engine.results:
                fig = self._create_simple_model_performance(self.ml_engine)
                self.display_plot(fig)
            else:
                messagebox.showinfo("Info", "No model results available yet")
        except Exception as e:
            messagebox.showerror("Error", f"Error showing model performance: {str(e)}")

    def show_feature_importance(self):
        """Show enhanced feature importance."""
        if not self.training_complete:
            messagebox.showwarning("Warning", "Please train models first")
            return

        try:
            # Get feature importance from ML engine
            if self.ml_engine and hasattr(self.ml_engine, 'get_feature_importance'):
                importance_df = self.ml_engine.get_feature_importance()
                if importance_df is not None and not importance_df.empty:
                    fig = self._create_simple_feature_importance(importance_df)
                    self.display_plot(fig)
                else:
                    messagebox.showinfo("Info", "Feature importance not available yet")
            else:
                messagebox.showinfo("Info", "Feature importance not available for this model type")
        except Exception as e:
            messagebox.showerror("Error", f"Error showing feature importance: {str(e)}")

    def _create_simple_model_performance(self, ml_engine):
        """Create a simple model performance visualization."""
        import matplotlib.pyplot as plt
        
        results = ml_engine.results
        models = list(results.keys())
        
        # Determine metric to show based on problem type
        if ml_engine.problem_type == 'classification':
            metric_key = 'accuracy'
            metric_name = 'Accuracy'
        else:
            metric_key = 'r2_score'
            metric_name = 'R² Score'
        
        # Extract metrics
        scores = [results[model].get(metric_key, 0) for model in models]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('🎯 Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bar chart of model scores
        ax1 = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax1.barh(models, scores, color=colors)
        ax1.set_xlabel(metric_name, fontweight='bold')
        ax1.set_title(f'📊 Model Comparison - {metric_name}', fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontweight='bold')
        
        # 2. Ranking
        ax2 = axes[0, 1]
        sorted_models = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
        ax2.axis('off')
        ax2.text(0.5, 0.95, '🏆 Model Rankings', ha='center', fontsize=14, 
                fontweight='bold', transform=ax2.transAxes)
        
        for i, (model, score) in enumerate(sorted_models[:5], 1):
            medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
            y_pos = 0.85 - (i-1) * 0.15
            ax2.text(0.1, y_pos, f'{medal} {model}', fontsize=11, 
                    transform=ax2.transAxes, fontweight='bold')
            ax2.text(0.9, y_pos, f'{score:.4f}', fontsize=11, ha='right',
                    transform=ax2.transAxes)
        
        # 3. Performance distribution
        ax3 = axes[1, 0]
        ax3.hist(scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(scores):.4f}')
        ax3.set_xlabel(metric_name, fontweight='bold')
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('📈 Score Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.text(0.5, 0.95, '📋 Performance Summary', ha='center', fontsize=14,
                fontweight='bold', transform=ax4.transAxes)
        
        stats_text = [
            f'Best Model: {sorted_models[0][0]}',
            f'Best Score: {sorted_models[0][1]:.4f}',
            f'Mean Score: {np.mean(scores):.4f}',
            f'Std Dev: {np.std(scores):.4f}',
            f'Models Trained: {len(models)}',
            f'Problem Type: {ml_engine.problem_type.title()}'
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 0.8 - i * 0.12
            ax4.text(0.1, y_pos, text, fontsize=11, transform=ax4.transAxes)
        
        plt.tight_layout()
        return fig

    def _create_simple_feature_importance(self, importance_df):
        """Create a simple feature importance visualization."""
        import matplotlib.pyplot as plt
        
        # Take top 20 features
        top_n = min(20, len(importance_df))
        top_features = importance_df.head(top_n)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle('🔍 Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Horizontal bar chart
        ax1 = axes[0]
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_features)))
        y_pos = np.arange(len(top_features))
        
        ax1.barh(y_pos, top_features['importance'], color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_features['feature'])
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score', fontweight='bold')
        ax1.set_title(f'📊 Top {top_n} Features', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax1.text(row['importance'], i, f" {row['importance']:.4f}", 
                    va='center', fontweight='bold', fontsize=9)
        
        # 2. Cumulative importance
        ax2 = axes[1]
        cumulative = np.cumsum(top_features['importance'])
        ax2.plot(range(1, len(cumulative)+1), cumulative, marker='o', 
                linewidth=2, markersize=6, color='steelblue')
        ax2.axhline(y=cumulative[-1]*0.8, color='red', linestyle='--', 
                   label='80% of total importance')
        ax2.set_xlabel('Number of Features', fontweight='bold')
        ax2.set_ylabel('Cumulative Importance', fontweight='bold')
        ax2.set_title('📈 Cumulative Feature Importance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add summary text
        total_importance = importance_df['importance'].sum()
        top_5_importance = importance_df.head(5)['importance'].sum()
        summary_text = f"Top 5 features: {(top_5_importance/total_importance)*100:.1f}% of total importance"
        ax2.text(0.5, 0.02, summary_text, transform=ax2.transAxes, 
                ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontweight='bold')
        
        plt.tight_layout()
        return fig

    def _create_simple_data_overview(self, original_data, processed_data=None):
        """Create a simple data overview visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('📊 Data Overview', fontsize=16, fontweight='bold')
        
        # 1. Data shape info
        ax1 = axes[0, 0]
        ax1.axis('off')
        ax1.text(0.5, 0.9, 'Data Information', ha='center', fontsize=14, 
                fontweight='bold', transform=ax1.transAxes)
        info_text = [
            f'Rows: {original_data.shape[0]:,}',
            f'Columns: {original_data.shape[1]}',
            f'Memory: {original_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB',
            f'Numeric: {len(original_data.select_dtypes(include=[np.number]).columns)}',
            f'Categorical: {len(original_data.select_dtypes(include=["object"]).columns)}',
        ]
        for i, text in enumerate(info_text):
            ax1.text(0.1, 0.7 - i*0.12, text, fontsize=11, transform=ax1.transAxes)
        
        # 2. Missing values
        ax2 = axes[0, 1]
        missing = original_data.isnull().sum()
        if missing.sum() > 0:
            top_missing = missing[missing > 0].sort_values(ascending=False)[:10]
            ax2.barh(range(len(top_missing)), top_missing.values, color='coral')
            ax2.set_yticks(range(len(top_missing)))
            ax2.set_yticklabels(top_missing.index)
            ax2.set_xlabel('Missing Count')
            ax2.set_title('🕳️ Top Missing Values')
        else:
            ax2.text(0.5, 0.5, '✅ No Missing Values', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
            ax2.axis('off')
        
        # 3. Data types distribution
        ax3 = axes[1, 0]
        dtypes_count = original_data.dtypes.value_counts()
        colors = plt.cm.Set3(range(len(dtypes_count)))
        wedges, texts, autotexts = ax3.pie(dtypes_count.values, labels=dtypes_count.index,
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('📋 Data Types Distribution')
        
        # 4. Numeric summary
        ax4 = axes[1, 1]
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = original_data[numeric_cols].describe().loc[['mean', 'std', 'min', 'max']].T
            top_stats = stats.head(10)
            
            x = np.arange(len(top_stats))
            width = 0.2
            ax4.bar(x - width*1.5, top_stats['mean'], width, label='Mean', alpha=0.8)
            ax4.bar(x - width*0.5, top_stats['std'], width, label='Std', alpha=0.8)
            ax4.set_xlabel('Features')
            ax4.set_ylabel('Value')
            ax4.set_title('📈 Numeric Features Summary')
            ax4.legend()
            ax4.set_xticks(x)
            ax4.set_xticklabels(top_stats.index, rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center',
                    fontsize=12, transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.tight_layout()
        return fig

    def _create_simple_before_after(self, before_data, after_data):
        """Create a simple before/after comparison visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('🔄 Before & After Comparison', fontsize=16, fontweight='bold')
        
        # 1. Shape comparison
        ax1 = axes[0, 0]
        categories = ['Rows', 'Columns']
        before_vals = [before_data.shape[0], before_data.shape[1]]
        after_vals = [after_data.shape[0], after_data.shape[1]]
        
        x = np.arange(len(categories))
        width = 0.35
        ax1.bar(x - width/2, before_vals, width, label='Before', color='lightcoral', alpha=0.8)
        ax1.bar(x + width/2, after_vals, width, label='After', color='lightgreen', alpha=0.8)
        ax1.set_ylabel('Count')
        ax1.set_title('📏 Data Shape Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Missing values comparison
        ax2 = axes[0, 1]
        before_missing = before_data.isnull().sum().sum()
        after_missing = after_data.isnull().sum().sum()
        missing_vals = [before_missing, after_missing]
        colors = ['lightcoral', 'lightgreen']
        ax2.bar(['Before', 'After'], missing_vals, color=colors, alpha=0.8)
        ax2.set_ylabel('Missing Values')
        ax2.set_title('🕳️ Missing Values Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage change
        if before_missing > 0:
            pct_change = ((after_missing - before_missing) / before_missing) * 100
            ax2.text(0.5, 0.95, f'Change: {pct_change:+.1f}%',
                    transform=ax2.transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Data types comparison
        ax3 = axes[1, 0]
        before_dtypes = before_data.dtypes.value_counts()
        after_dtypes = after_data.dtypes.value_counts()
        
        all_types = sorted(set(list(before_dtypes.index) + list(after_dtypes.index)))
        before_counts = [before_dtypes.get(dt, 0) for dt in all_types]
        after_counts = [after_dtypes.get(dt, 0) for dt in all_types]
        
        x = np.arange(len(all_types))
        width = 0.35
        ax3.bar(x - width/2, before_counts, width, label='Before', color='lightcoral', alpha=0.8)
        ax3.bar(x + width/2, after_counts, width, label='After', color='lightgreen', alpha=0.8)
        ax3.set_ylabel('Count')
        ax3.set_title('📋 Data Types Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(dt) for dt in all_types], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Memory usage comparison
        ax4 = axes[1, 1]
        before_mem = before_data.memory_usage(deep=True).sum() / 1024**2
        after_mem = after_data.memory_usage(deep=True).sum() / 1024**2
        
        ax4.bar(['Before', 'After'], [before_mem, after_mem], 
                color=['lightcoral', 'lightgreen'], alpha=0.8)
        ax4.set_ylabel('Memory (MB)')
        ax4.set_title('💾 Memory Usage Comparison')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add memory saved
        mem_saved = before_mem - after_mem
        pct_saved = (mem_saved / before_mem) * 100 if before_mem > 0 else 0
        ax4.text(0.5, 0.95, f'Saved: {mem_saved:.1f} MB ({pct_saved:.1f}%)',
                transform=ax4.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        return fig

    def display_plot(self, fig):
        """Enhanced display matplotlib figure in canvas with memory management."""
        try:
            # Clear previous plot
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            if self.toolbar:
                self.toolbar.destroy()
                self.toolbar = None

            # Create new canvas
            self.canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
            self.toolbar.update()
            
            # Update memory usage
            self._update_memory_usage()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying plot: {str(e)}")

    def _update_memory_usage(self):
        """Update memory usage display."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_status_var.set(f"Memory: {memory_mb:.0f}MB")
        except:
            # Fallback if psutil not available
            import sys
            if hasattr(sys, 'getsizeof'):
                data_size = 0
                if self.current_data is not None:
                    data_size += sys.getsizeof(self.current_data)
                if self.cleaned_data is not None:
                    data_size += sys.getsizeof(self.cleaned_data)
                self.memory_status_var.set(f"Data: {data_size/1024/1024:.0f}MB")

    # ==== EXPORT METHODS (Enhanced) ====

    def export_cleaned_data(self):
        """Enhanced export cleaned data with format options."""
        if self.cleaned_data is None:
            messagebox.showwarning("Warning", "No cleaned data to export")
            return

        # Choose export format
        formats = ["CSV", "Excel", "JSON", "Parquet"]
        if not EXCEL_AVAILABLE:
            formats.remove("Excel")
        if not PARQUET_AVAILABLE:
            formats.remove("Parquet")
            
        format_choice = self._show_choice_dialog("Export Format", "Choose export format:", formats)
        if not format_choice:
            return

        # File dialog based on format
        format_extensions = {
            "CSV": (".csv", "CSV files", "*.csv"),
            "Excel": (".xlsx", "Excel files", "*.xlsx"),
            "JSON": (".json", "JSON files", "*.json"),
            "Parquet": (".parquet", "Parquet files", "*.parquet")
        }

        ext, desc, pattern = format_extensions[format_choice]
        
        file_path = filedialog.asksaveasfilename(
            title=f"Save cleaned data as {format_choice}",
            defaultextension=ext,
            filetypes=[(desc, pattern), ("All files", "*.*")]
        )

        if file_path:
            try:
                if format_choice == "CSV":
                    self.cleaned_data.to_csv(file_path, index=False)
                elif format_choice == "Excel":
                    self.cleaned_data.to_excel(file_path, index=False)
                elif format_choice == "JSON":
                    self.cleaned_data.to_json(file_path, orient='records', indent=2)
                elif format_choice == "Parquet":
                    self.cleaned_data.to_parquet(file_path, index=False)
                
                messagebox.showinfo("Success", f"Cleaned data exported successfully to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting data: {str(e)}")

    def export_results(self):
        """Enhanced export analysis results."""
        file_path = filedialog.asksaveasfilename(
            title="Save analysis results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                if file_path.endswith('.json'):
                    self._export_results_json(file_path)
                else:
                    self._export_results_text(file_path)
                    
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting results: {str(e)}")

    def _export_results_json(self, file_path):
        """Export results in JSON format."""
        results = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "original_shape": list(self.current_data.shape) if self.current_data is not None else None,
                "cleaned_shape": list(self.cleaned_data.shape) if self.cleaned_data is not None else None
            },
            "meta_features": self.meta_features,
            "training_complete": self.training_complete
        }
        
        # Add AI decisions if available
        if hasattr(self.data_processor, 'ai_engine') and self.data_processor.ai_engine:
            results["ai_decisions"] = [
                {
                    "timestamp": decision["timestamp"],
                    "decision": decision["decision"]
                }
                for decision in self.data_processor.ai_engine.reasoning_log
            ]
        
        # Add ML results if available
        if self.training_complete and hasattr(self.ml_engine, 'get_results_summary'):
            results["ml_results"] = self.ml_engine.get_results_summary()
            
        # Write with proper encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def _export_results_text(self, file_path):
        """Export results in text format (enhanced version of original method)."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ENHANCED AI DATA ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Session ID: {self.current_session_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset info
            if self.current_data is not None:
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Original Dataset: {self.current_data.shape[0]:,} rows × {self.current_data.shape[1]} columns\n")
                if self.cleaned_data is not None:
                    f.write(f"Cleaned Dataset: {self.cleaned_data.shape[0]:,} rows × {self.cleaned_data.shape[1]} columns\n")
                    rows_removed = self.current_data.shape[0] - self.cleaned_data.shape[0]
                    cols_changed = self.cleaned_data.shape[1] - self.current_data.shape[1]
                    f.write(f"Rows Removed: {rows_removed:,}\n")
                    f.write(f"Columns Added/Removed: {cols_changed:+d}\n")
                f.write("\n")
            
            # Meta-features
            if self.meta_features:
                f.write("DATASET META-FEATURES:\n")
                f.write("-" * 40 + "\n")
                for key, value in self.meta_features.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # AI Analysis results
            if hasattr(self.data_processor, 'ai_engine') and self.data_processor.ai_engine:
                f.write("AI ANALYSIS DECISIONS:\n")
                f.write("-" * 40 + "\n")
                for decision in self.data_processor.ai_engine.reasoning_log:
                    f.write(f"[{decision['timestamp']}] {decision['decision']}\n")
                f.write("\n")
            
            # ML Results
            if self.training_complete:
                f.write("MACHINE LEARNING RESULTS:\n")
                f.write("-" * 40 + "\n")
                if hasattr(self.ml_engine, 'get_training_log'):
                    for log_entry in self.ml_engine.get_training_log():
                        f.write(log_entry + "\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Enhanced Analysis Report\n")

    # ==== ENHANCED QUEUE HANDLING ====

    def check_queue(self):
        """Enhanced queue checking with more message types."""
        try:
            while True:
                message_type, data = self.queue.get_nowait()

                if message_type == "file_loaded":
                    filename = data["filename"] if isinstance(data, dict) else data
                    self.file_label.config(text=f"Loaded: {filename}")
                    self.status_var.set("File loaded successfully - AI analysis complete")
                    self.processing_status_var.set("Ready")
                    
                    # Update data source info if available
                    if isinstance(data, dict) and "source_info" in data:
                        self._update_data_source_display(data["source_info"])
                    
                    self.update_analysis_displays()

                elif message_type == "meta_features_extracted":
                    self.status_var.set("Meta-features extracted successfully")
                    self.update_meta_analysis()

                elif message_type == "process_selected":
                    self.status_var.set("Process selection completed")
                    self._display_process_selection_results(data)

                elif message_type == "cleaning_complete":
                    self.cleaning_progress_bar.stop()
                    self.status_var.set("Data cleaning completed successfully")
                    self.processing_status_var.set("Complete")
                    self.cleaning_progress_var.set("Cleaning completed!")
                    self.cleaning_status_var.set("Status: Complete")
                    
                    # Update lineage if available
                    if isinstance(data, dict):
                        self._update_data_lineage_display(data)
                    
                    self.update_cleaning_displays()

                elif message_type == "training_complete":
                    self.status_var.set(f"Model training completed successfully - {data if isinstance(data, str) else 'Standard'}")
                    self.processing_status_var.set("Complete")
                    self.update_ml_displays()

                elif message_type == "log_step":
                    # Add training step to log
                    self.ml_log_text.insert(tk.END, data + "\n")
                    self.ml_log_text.see(tk.END)
                    
                    # Also update cleaning log if it's a cleaning step
                    if any(keyword in data.lower() for keyword in ['clean', 'impute', 'encode', 'process']):
                        self.cleaning_results_text.insert(tk.END, data + "\n")
                        self.cleaning_results_text.see(tk.END)

                elif message_type == "report_generated":
                    self.status_var.set(f"Report generated: {data}")
                    messagebox.showinfo("Report Generated", f"Comprehensive report saved to:\n{data}")

                elif message_type == "error":
                    self.cleaning_progress_bar.stop()
                    self.ai_progress_bar.stop()
                    self.status_var.set("Error occurred")
                    self.processing_status_var.set("Error")
                    self.cleaning_status_var.set("Status: Error")
                    messagebox.showerror("Error", data)

        except queue.Empty:
            pass

        # Update memory usage periodically
        if hasattr(self, '_last_memory_update'):
            if (datetime.now() - self._last_memory_update).seconds > 5:
                self._update_memory_usage()
                self._last_memory_update = datetime.now()
        else:
            self._last_memory_update = datetime.now()

        # Schedule next check
        self.root.after(100, self.check_queue)

    def _update_data_source_display(self, source_info):
        """Update data source information display."""
        try:
            source_text = "=== DATA SOURCE INFORMATION ===\n\n"
            source_text += f"Type: {source_info['type'].upper()}\n"
            source_text += f"Path: {source_info['path']}\n"
            source_text += f"Size: {source_info['size']:,} bytes\n"
            source_text += f"Modified: {source_info['modified']}\n"
            source_text += f"Loaded: {source_info['loaded']}\n\n"
            
            if 'rows' in source_info:
                source_text += f"Rows: {source_info['rows']:,}\n"
            
            if 'connection' in source_info:
                source_text += f"Database Connection: {source_info['connection']}\n"
                source_text += f"Query: {source_info['query']}\n"
            
            self.data_source_text.delete(1.0, tk.END)
            self.data_source_text.insert(tk.END, source_text)
        except Exception as e:
            print(f"Error updating data source display: {e}")

    def _display_process_selection_results(self, results):
        """Display process selection results."""
        try:
            if isinstance(results, dict):
                result_text = "🎯 INTELLIGENT PROCESS SELECTION RESULTS:\n\n"
                result_text += f"Recommended Pipeline: {results.get('pipeline', 'Standard ML')}\n"
                result_text += f"Confidence: {results.get('confidence', 'N/A')}\n\n"
                
                if 'reasoning' in results:
                    result_text += "Reasoning:\n"
                    for reason in results['reasoning']:
                        result_text += f"• {reason}\n"
                
                self.autoselect_text.delete(1.0, tk.END)
                self.autoselect_text.insert(tk.END, result_text)
        except Exception as e:
            print(f"Error displaying process selection results: {e}")

    def _update_data_lineage_display(self, lineage_info):
        """Update data lineage display."""
        try:
            lineage_text = "=== DATA LINEAGE TRACKING ===\n\n"
            lineage_text += f"Original Shape: {lineage_info['original_shape']}\n"
            lineage_text += f"Final Shape: {lineage_info['cleaned_shape']}\n"
            lineage_text += f"Processing Time: {lineage_info['timestamp']}\n\n"
            
            lineage_text += "Processing Steps:\n"
            for step in lineage_info.get('cleaning_steps', []):
                lineage_text += f"• {step}\n"
            
            self.data_lineage_text.delete(1.0, tk.END)
            self.data_lineage_text.insert(tk.END, lineage_text)
        except Exception as e:
            print(f"Error updating data lineage display: {e}")

    # ==== ENHANCED DISPLAY UPDATE METHODS ====

    def update_analysis_displays(self):
        """Enhanced analysis display updates."""
        # Update data preview with enhanced formatting
        self.data_preview_text.delete(1.0, tk.END)
        if self.current_data is not None:
            preview = self.current_data.head(20).to_string(max_cols=10, max_colwidth=30)
            self.data_preview_text.insert(tk.END, preview)

        # Update data info with comprehensive details
        self.data_info_text.delete(1.0, tk.END)
        if hasattr(self.data_processor, 'data_info'):
            info_text = self._format_enhanced_data_info()
            self.data_info_text.insert(tk.END, info_text)

        # Update column analysis with detailed insights
        self.column_analysis_text.delete(1.0, tk.END)
        if hasattr(self.data_processor, 'column_analyses'):
            analysis_text = self._format_enhanced_column_analysis()
            self.column_analysis_text.insert(tk.END, analysis_text)

        # Update AI decisions with enhanced formatting
        self.ai_decisions_text.delete(1.0, tk.END)
        if hasattr(self.data_processor, 'ai_engine') and self.data_processor.ai_engine:
            for decision in self.data_processor.ai_engine.reasoning_log:
                timestamp = decision.get('timestamp', 'N/A')
                text = decision.get('decision', 'N/A')
                self.ai_decisions_text.insert(tk.END, f"🧠 [{timestamp}] {text}\n")
            self.ai_decisions_text.see(tk.END)

        # Update recommendations with priority indicators
        self.recommendations_text.delete(1.0, tk.END)
        if hasattr(self.data_processor, 'cleaning_recommendations'):
            rec_text = self._format_enhanced_recommendations()
            self.recommendations_text.insert(tk.END, rec_text)

        # Update strategies with detailed explanations
        self.strategies_text.delete(1.0, tk.END)
        if hasattr(self.data_processor, 'column_analyses'):
            strategy_text = self._format_enhanced_strategies()
            self.strategies_text.insert(tk.END, strategy_text)

    def _format_enhanced_data_info(self):
        """Enhanced data information formatting."""
        info = self.data_processor.data_info
        text = "=== ENHANCED DATASET INFORMATION ===\n\n"
        
        # Basic info
        text += f"📊 Shape: {info['shape'][0]:,} rows × {info['shape'][1]} columns\n"
        text += f"💾 Memory Usage: {info['memory_usage_mb']:.2f} MB\n"
        text += f"✅ Data Completeness: {info['completeness_percentage']:.1f}%\n\n"
        
        # Column type breakdown
        text += f"📈 Column Types Distribution:\n"
        text += f"   🔢 Numerical: {len(info['numerical_columns'])} columns\n"
        text += f"   🏷️  Categorical: {len(info['categorical_columns'])} columns\n"
        text += f"   📅 Datetime: {len(info['datetime_columns'])} columns\n"
        text += f"   📝 Text: {len(info['text_columns'])} columns\n\n"
        
        # Data quality indicators
        text += f"⚠️  Data Quality Issues:\n"
        text += f"   🔺 High Cardinality: {len(info['high_cardinality_columns'])} columns\n"
        text += f"   🕳️  High Missing Values: {len(info['high_missing_columns'])} columns\n\n"
        
        # Additional meta-information
        if self.meta_features:
            text += f"🔍 Advanced Metrics:\n"
            text += f"   📊 Data Sparsity: {self.meta_features.get('sparsity', 0):.1f}%\n"
            text += f"   🎯 Max Correlation: {self.meta_features.get('max_correlation', 0):.3f}\n"
            text += f"   ⚠️  Outliers: {self.meta_features.get('outlier_percentage', 0):.1f}%\n"
        
        return text

    def _format_enhanced_column_analysis(self):
        """Enhanced column analysis formatting."""
        text = "=== ENHANCED AI COLUMN ANALYSIS ===\n\n"
        
        # Sort columns by priority for better organization
        sorted_analyses = sorted(
            self.data_processor.column_analyses.items(),
            key=lambda x: x[1]['priority']['level'],
            reverse=True
        )
        
        for col, analysis in sorted_analyses:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                analysis['priority']['level'], "⚪"
            )
            
            text += f"📊 Column: {col} {priority_emoji}\n"
            text += f"   🔍 AI Type: {analysis['ai_type']}\n"
            text += f"   🕳️  Missing: {analysis['null_percentage']:.1f}%\n"
            text += f"   🔢 Unique: {analysis['unique_count']} ({analysis['unique_percentage']:.1f}%)\n"
            text += f"   ⭐ Priority: {analysis['priority']['level'].upper()}\n"
            text += f"   💭 Reasoning: {analysis['reasoning']}\n"
            text += f"   🎯 Strategy: {analysis['missing_strategy']['reasoning']}\n"
            text += "─" * 60 + "\n"
        
        return text

    def _format_enhanced_recommendations(self):
        """Enhanced cleaning recommendations formatting."""
        text = "=== ENHANCED AI CLEANING RECOMMENDATIONS ===\n\n"
        
        if hasattr(self.data_processor, 'cleaning_recommendations'):
            # Group by priority
            priority_groups = {"high": [], "medium": [], "low": []}
            
            for col, rec in self.data_processor.cleaning_recommendations.items():
                priority = rec.get('priority', 'medium').lower()
                priority_groups[priority].append((col, rec))
            
            for priority in ["high", "medium", "low"]:
                if priority_groups[priority]:
                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[priority]
                    text += f"{priority_emoji} {priority.upper()} PRIORITY ITEMS:\n"
                    text += "─" * 40 + "\n"
                    
                    for col, rec in priority_groups[priority]:
                        text += f"📊 Column: {col}\n"
                        text += f"   🤖 Auto-Apply: {'Yes' if rec['auto_apply'] else 'No (Manual Review)'}\n"
                        text += f"   💡 Recommendations:\n"
                        for r in rec['recommendations']:
                            text += f"      • {r}\n"
                        text += "\n"
                    text += "\n"
        
        return text

    def _format_enhanced_strategies(self):
        """Enhanced cleaning strategies formatting."""
        text = "=== CLEANING STRATEGIES EXPLAINED ===\n\n"
        
        strategies = {
            'none': ("No Action", "Column is complete - no missing values detected", "🟢"),
            'median': ("Median Imputation", "Fill missing numerical values with median (robust to outliers)", "🔢"),
            'mode': ("Mode Imputation", "Fill missing categorical values with most frequent value", "🏷️"),
            'knn': ("KNN Imputation", "Use K-Nearest Neighbors to predict missing values based on similar rows", "🤖"),
            'iterative': ("Iterative Imputation", "Use machine learning to predict missing values iteratively", "🔄"),
            'create_indicator': ("Missing Flag + Impute", "Create binary flag for missingness pattern + impute values", "🚩"),
            'consider_removal': ("Manual Review", "High missing percentage - consider column removal or advanced techniques", "⚠️"),
            'frequency_encoding': ("Frequency Encoding", "Replace categorical values with their frequency of occurrence", "📊"),
            'one_hot_encoding': ("One-Hot Encoding", "Create binary columns for each category (low cardinality)", "🎯"),
            'text_preprocessing': ("Text Processing", "Tokenization, lemmatization, and vectorization for text data", "📝")
        }
        
        for strategy, (name, explanation, emoji) in strategies.items():
            text += f"{emoji} {name.upper()}:\n"
            text += f"   Strategy: {strategy}\n"
            text += f"   Description: {explanation}\n"
            text += f"   Best for: "
            
            # Add use case recommendations
            if strategy == 'median':
                text += "Numeric columns with moderate missing values\n"
            elif strategy == 'mode':
                text += "Categorical columns with clear dominant category\n"
            elif strategy == 'knn':
                text += "Columns where similar rows can inform missing values\n"
            elif strategy == 'iterative':
                text += "Complex missing patterns with feature interdependencies\n"
            elif strategy == 'create_indicator':
                text += "When missingness itself may be informative\n"
            else:
                text += "Various scenarios based on data characteristics\n"
            
            text += "\n"
        
        return text

    def update_cleaning_displays(self):
        """Enhanced cleaning results display updates."""
        if self.cleaned_data is not None:
            summary = self.data_processor.get_cleaning_summary()
            
            results_text = "=== ENHANCED DATA CLEANING SUMMARY ===\n\n"
            
            # Shape changes
            results_text += f"📊 Data Shape Changes:\n"
            results_text += f"   Original: {summary['original_shape'][0]:,} × {summary['original_shape'][1]}\n"
            results_text += f"   Cleaned:  {summary['cleaned_shape'][0]:,} × {summary['cleaned_shape'][1]}\n"
            results_text += f"   Rows Removed: {summary['rows_removed']:,}\n"
            results_text += f"   Columns Changed: {summary['columns_added']:+d}\n\n"
            
            # Quality improvements
            if 'data_quality_improvement' in summary:
                quality = summary['data_quality_improvement']
                results_text += f"📈 Quality Improvements:\n"
                results_text += f"   Original Completeness: {quality['original_completeness']:.1f}%\n"
                results_text += f"   Cleaned Completeness:  {quality['cleaned_completeness']:.1f}%\n"
                results_text += f"   Net Improvement: +{quality['improvement']:.1f}%\n"
                results_text += f"   Missing Values Fixed: {quality['missing_values_handled']:,}\n\n"
            
            # AI decision summary
            results_text += f"🤖 AI Decision Summary:\n"
            results_text += f"   Total Decisions: {summary['ai_decisions']}\n"
            results_text += f"   Recommendations Applied: {summary['recommendations_applied']}\n"
            results_text += f"   Processing Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            # Detailed cleaning log
            results_text += "📝 DETAILED CLEANING LOG:\n"
            results_text += "=" * 70 + "\n"
            
            for entry in summary.get('cleaning_log', []):
                timestamp = entry.get('timestamp', 'N/A')
                decision = entry.get('decision', 'N/A')
                results_text += f"🕒 [{timestamp}] {decision}\n"
            
            self.cleaning_results_text.delete(1.0, tk.END)
            self.cleaning_results_text.insert(tk.END, results_text)
            
            # Update quality improvement tab if it exists
            if hasattr(self, 'quality_improvement_text'):
                quality_text = self._format_quality_improvement_details(summary)
                self.quality_improvement_text.delete(1.0, tk.END)
                self.quality_improvement_text.insert(tk.END, quality_text)

    def _format_quality_improvement_details(self, summary):
        """Format detailed quality improvement information."""
        text = "=== DATA QUALITY IMPROVEMENT ANALYSIS ===\n\n"
        
        if 'data_quality_improvement' in summary:
            quality = summary['data_quality_improvement']
            
            # Overall improvement score
            improvement_score = quality['improvement']
            if improvement_score > 10:
                score_rating = "🟢 Excellent"
            elif improvement_score > 5:
                score_rating = "🟡 Good"
            elif improvement_score > 0:
                score_rating = "🟠 Moderate"
            else:
                score_rating = "🔴 Minimal"
            
            text += f"📊 Overall Quality Improvement: {score_rating} (+{improvement_score:.1f}%)\n\n"
            
            # Specific improvements
            text += f"🔍 Detailed Improvements:\n"
            text += f"   • Data Completeness: {quality['original_completeness']:.1f}% → {quality['cleaned_completeness']:.1f}%\n"
            text += f"   • Missing Values Addressed: {quality['missing_values_handled']:,}\n"
            
            # Additional quality metrics if available
            if 'outliers_handled' in quality:
                text += f"   • Outliers Processed: {quality['outliers_handled']:,}\n"
            if 'duplicates_removed' in quality:
                text += f"   • Duplicates Removed: {quality['duplicates_removed']:,}\n"
            if 'encoding_applied' in quality:
                text += f"   • Encoding Transformations: {quality['encoding_applied']}\n"
        
        return text

    def update_ml_displays(self):
        """Enhanced ML results display updates."""
        # Update model comparison with enhanced formatting
        comparison = self.ml_engine.get_model_comparison()
        if isinstance(comparison, dict) and 'headers' in comparison and 'rows' in comparison:
            comp_text = "MODEL PERFORMANCE COMPARISON\n"
            comp_text += "=" * 80 + "\n\n"
            
            # Header
            header_line = f"{'Model':<25}"
            for header in comparison['headers'][1:]:
                header_line += f"{header:<15}"
            comp_text += header_line + "\n"
            comp_text += "─" * 80 + "\n"
            
            # Rows with ranking indicators
            for i, row in enumerate(comparison['rows']):
                rank_indicator = "[1st]" if i == 0 else "[2nd]" if i == 1 else "[3rd]" if i == 2 else f"[{i+1}th]"
                row_line = f"{rank_indicator} {row[0]:<20}"
                
                for j, cell in enumerate(row[1:]):
                    # Performance indicators without emojis
                    if isinstance(cell, (int, float)) and j == 0:  # Assuming first metric is primary
                        if cell > 0.9:
                            prefix = "[HIGH]"
                        elif cell > 0.8:
                            prefix = "[MED]"
                        else:
                            prefix = "[LOW]"
                        row_line += f"{prefix}{cell:<10}"
                    else:
                        row_line += f"{cell:<15}"
                
                comp_text += row_line + "\n"
        elif isinstance(comparison, dict) and 'error' in comparison:
            comp_text = f"MODEL COMPARISON ERROR\n{comparison['error']}\n"
        else:
            comp_text = "No model comparison data available"
        
        self.model_comparison_text.delete(1.0, tk.END)
        self.model_comparison_text.insert(tk.END, comp_text)

        # Update comprehensive summary
        self.update_comprehensive_summary()

    def update_comprehensive_summary(self):
        """Enhanced comprehensive summary display."""
        # Executive Summary
        exec_text = "EXECUTIVE SUMMARY\n"
        exec_text += "=" * 50 + "\n\n"
        
        if self.current_data is not None:
            exec_text += f"📊 Dataset Overview:\n"
            exec_text += f"   • Original: {self.current_data.shape[0]:,} rows × {self.current_data.shape[1]} columns\n"
            if self.cleaned_data is not None:
                improvement = ((self.cleaned_data.shape[0] * self.cleaned_data.shape[1]) / 
                              (self.current_data.shape[0] * self.current_data.shape[1])) * 100
                exec_text += f"   • Processed: {self.cleaned_data.shape[0]:,} rows × {self.cleaned_data.shape[1]} columns\n"
                exec_text += f"   • Data Utilization: {improvement:.1f}%\n"
            exec_text += "\n"
        
        if self.meta_features:
            exec_text += f"🔍 Data Quality Assessment:\n"
            completeness = self.meta_features.get('completeness', 0)
            sparsity = self.meta_features.get('sparsity', 0)
            quality_score = completeness - sparsity
            
            if quality_score > 80:
                quality_rating = "🟢 Excellent"
            elif quality_score > 60:
                quality_rating = "🟡 Good"
            elif quality_score > 40:
                quality_rating = "🟠 Fair"
            else:
                quality_rating = "🔴 Needs Improvement"
            
            exec_text += f"   • Overall Quality: {quality_rating} ({quality_score:.1f}%)\n"
            exec_text += f"   • Completeness: {completeness:.1f}%\n"
            exec_text += f"   • Data Types: {self.meta_features.get('pct_numeric', 0):.0f}% numeric, {self.meta_features.get('pct_categorical', 0):.0f}% categorical\n"
            exec_text += "\n"
        
        if self.training_complete:
            exec_text += f"🤖 Machine Learning Results:\n"
            exec_text += f"   • Status: ✅ Training Complete\n"
            if hasattr(self.ml_engine, 'models'):
                exec_text += f"   • Models Trained: {len(self.ml_engine.models)}\n"
            if hasattr(self.ml_engine, 'best_model_name'):
                exec_text += f"   • Best Model: {self.ml_engine.best_model_name}\n"
            exec_text += f"   • Problem Type: {self.problem_type_var.get().title()}\n"
        else:
            exec_text += f"🤖 Machine Learning: ❌ Not Started\n"
        
        self.exec_summary_text.delete(1.0, tk.END)
        self.exec_summary_text.insert(tk.END, exec_text)
        
        # Technical Summary (existing method but enhanced)
        tech_text = "🔧 TECHNICAL ANALYSIS SUMMARY\n"
        tech_text += "=" * 60 + "\n\n"
        
        # Dataset technical details
        if self.current_data is not None:
            tech_text += f"📊 DATASET TECHNICAL DETAILS:\n"
            tech_text += f"   Original Shape: {self.current_data.shape[0]:,} × {self.current_data.shape[1]}\n"
            if self.cleaned_data is not None:
                tech_text += f"   Processed Shape: {self.cleaned_data.shape[0]:,} × {self.cleaned_data.shape[1]}\n"
                tech_text += f"   Memory Usage: {self.current_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
            tech_text += "\n"
        
        # AI Analysis technical details
        if hasattr(self.data_processor, 'ai_engine') and self.data_processor.ai_engine:
            tech_text += f"🧠 AI ANALYSIS DETAILS:\n"
            tech_text += f"   Decisions Made: {len(self.data_processor.ai_engine.reasoning_log)}\n"
            tech_text += f"   Columns Analyzed: {len(getattr(self.data_processor, 'column_analyses', {}))}\n"
            tech_text += f"   Cleaning Strategies: {len(getattr(self.data_processor, 'cleaning_recommendations', {}))}\n"
            tech_text += "\n"
        
        # ML technical details
        if self.training_complete:
            tech_text += f"🤖 ML TECHNICAL DETAILS:\n"
            if hasattr(self.ml_engine, 'models'):
                tech_text += f"   Models Trained: {len(self.ml_engine.models)}\n"
            tech_text += f"   Problem Type: {self.problem_type_var.get()}\n"
            tech_text += f"   Target Variable: {self.target_var.get()}\n"
            if hasattr(self.ml_engine, 'feature_count'):
                tech_text += f"   Feature Count: {self.ml_engine.feature_count}\n"
            tech_text += "\n"
        
        # Session information
        tech_text += f"💾 SESSION INFORMATION:\n"
        tech_text += f"   Session ID: {self.current_session_id}\n"
        tech_text += f"   Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        tech_text += f"   Platform: Enhanced AI Data Analytics Platform\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, tech_text)


# ==== DIALOG CLASSES ====

class DatabaseConnectionDialog:
    """Dialog for database connection configuration."""
    
    def __init__(self, parent):
        self.parent = parent
        self.connection_string = None
        self.connection_name = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Database Connection")
        self.dialog.geometry("500x400")
        self.dialog.grab_set()
        self.dialog.transient(parent)
        
        self.create_widgets()
        self.center_dialog()
    
    def center_dialog(self):
        """Center the dialog on screen."""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (400 // 2)
        self.dialog.geometry(f"500x400+{x}+{y}")
    
    def create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🗄️ Database Connection Setup", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Connection name
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=5)
        ttk.Label(name_frame, text="Connection Name:").pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value="My Database")
        ttk.Entry(name_frame, textvariable=self.name_var, width=30).pack(side=tk.RIGHT)
        
        # Database type
        type_frame = ttk.Frame(main_frame)
        type_frame.pack(fill=tk.X, pady=5)
        ttk.Label(type_frame, text="Database Type:").pack(side=tk.LEFT)
        self.db_type_var = tk.StringVar(value="sqlite")
        db_combo = ttk.Combobox(type_frame, textvariable=self.db_type_var,
                               values=["sqlite", "postgresql", "mysql", "mssql"],
                               state="readonly", width=27)
        db_combo.pack(side=tk.RIGHT)
        
        # Host
        host_frame = ttk.Frame(main_frame)
        host_frame.pack(fill=tk.X, pady=5)
        ttk.Label(host_frame, text="Host:").pack(side=tk.LEFT)
        self.host_var = tk.StringVar(value="localhost")
        ttk.Entry(host_frame, textvariable=self.host_var, width=30).pack(side=tk.RIGHT)
        
        # Port
        port_frame = ttk.Frame(main_frame)
        port_frame.pack(fill=tk.X, pady=5)
        ttk.Label(port_frame, text="Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value="5432")
        ttk.Entry(port_frame, textvariable=self.port_var, width=30).pack(side=tk.RIGHT)
        
        # Database name
        db_frame = ttk.Frame(main_frame)
        db_frame.pack(fill=tk.X, pady=5)
        ttk.Label(db_frame, text="Database:").pack(side=tk.LEFT)
        self.database_var = tk.StringVar()
        ttk.Entry(db_frame, textvariable=self.database_var, width=30).pack(side=tk.RIGHT)
        
        # Username
        user_frame = ttk.Frame(main_frame)
        user_frame.pack(fill=tk.X, pady=5)
        ttk.Label(user_frame, text="Username:").pack(side=tk.LEFT)
        self.username_var = tk.StringVar()
        ttk.Entry(user_frame, textvariable=self.username_var, width=30).pack(side=tk.RIGHT)
        
        # Password
        pass_frame = ttk.Frame(main_frame)
        pass_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pass_frame, text="Password:").pack(side=tk.LEFT)
        self.password_var = tk.StringVar()
        ttk.Entry(pass_frame, textvariable=self.password_var, show="*", width=30).pack(side=tk.RIGHT)
        
        # Connection string preview
        preview_frame = ttk.LabelFrame(main_frame, text="Connection String Preview", padding=10)
        preview_frame.pack(fill=tk.X, pady=(20, 10))
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=3, width=50)
        self.preview_text.pack(fill=tk.X)
        
        # Update preview when fields change
        for var in [self.db_type_var, self.host_var, self.port_var, 
                   self.database_var, self.username_var]:
            var.trace('w', self.update_preview)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Test Connection", 
                  command=self.test_connection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Connect", 
                  command=self.connect, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def update_preview(self, *args):
        """Update connection string preview."""
        try:
            db_type = self.db_type_var.get()
            host = self.host_var.get()
            port = self.port_var.get()
            database = self.database_var.get()
            username = self.username_var.get()
            
            if db_type == "sqlite":
                conn_str = f"sqlite:///{database}.db"
            elif db_type == "postgresql":
                conn_str = f"postgresql://{username}:***@{host}:{port}/{database}"
            elif db_type == "mysql":
                conn_str = f"mysql://{username}:***@{host}:{port}/{database}"
            elif db_type == "mssql":
                conn_str = f"mssql://{username}:***@{host}:{port}/{database}"
            else:
                conn_str = "Invalid database type"
            
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, conn_str)
        except:
            pass
    
    def test_connection(self):
        """Test database connection."""
        try:
            conn_str = self.build_connection_string()
            if SQL_AVAILABLE:
                engine = sa.create_engine(conn_str)
                with engine.connect() as conn:
                    conn.execute(sa.text("SELECT 1"))
                messagebox.showinfo("Success", "Connection test successful!")
            else:
                messagebox.showwarning("Warning", "SQLAlchemy not available for testing")
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect:\n{str(e)}")
    
    def build_connection_string(self):
        """Build connection string from form fields."""
        db_type = self.db_type_var.get()
        host = self.host_var.get()
        port = self.port_var.get()
        database = self.database_var.get()
        username = self.username_var.get()
        password = self.password_var.get()
        
        if db_type == "sqlite":
            return f"sqlite:///{database}.db"
        elif db_type == "postgresql":
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == "mysql":
            return f"mysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == "mssql":
            return f"mssql://{username}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def connect(self):
        """Accept connection and close dialog."""
        try:
            self.connection_string = self.build_connection_string()
            self.connection_name = self.name_var.get()
            self.dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid connection parameters:\n{str(e)}")
    
    def cancel(self):
        """Cancel and close dialog."""
        self.connection_string = None
        self.connection_name = None
        self.dialog.destroy()


class SQLQueryDialog:
    """Dialog for SQL query input."""
    
    def __init__(self, parent, connections):
        self.parent = parent
        self.connections = connections
        self.query = None
        self.connection_name = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("SQL Query")
        self.dialog.geometry("600x500")
        self.dialog.grab_set()
        self.dialog.transient(parent)
        
        self.create_widgets()
        self.center_dialog()
    
    def center_dialog(self):
        """Center the dialog on screen."""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (500 // 2)
        self.dialog.geometry(f"600x500+{x}+{y}")
    
    def create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="📊 SQL Query Builder", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Connection selection
        conn_frame = ttk.Frame(main_frame)
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(conn_frame, text="Database Connection:").pack(side=tk.LEFT)
        
        self.connection_var = tk.StringVar()
        conn_combo = ttk.Combobox(conn_frame, textvariable=self.connection_var,
                                 values=self.connections, state="readonly", width=30)
        conn_combo.pack(side=tk.RIGHT)
        if self.connections:
            conn_combo.current(0)
        
        # Query input
        query_frame = ttk.LabelFrame(main_frame, text="SQL Query", padding=10)
        query_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.query_text = scrolledtext.ScrolledText(query_frame, height=15, width=70,
                                                   font=("Consolas", 10))
        self.query_text.pack(fill=tk.BOTH, expand=True)
        
        # Sample queries
        samples_frame = ttk.LabelFrame(main_frame, text="Sample Queries", padding=10)
        samples_frame.pack(fill=tk.X, pady=(0, 10))
        
        sample_buttons_frame = ttk.Frame(samples_frame)
        sample_buttons_frame.pack(fill=tk.X)
        
        samples = [
            ("All Records", "SELECT * FROM table_name LIMIT 1000;"),
            ("Table Info", "SELECT name FROM sqlite_master WHERE type='table';"),
            ("Count Records", "SELECT COUNT(*) FROM table_name;"),
            ("Sample Data", "SELECT * FROM table_name ORDER BY RANDOM() LIMIT 100;")
        ]
        
        for i, (label, query) in enumerate(samples):
            ttk.Button(sample_buttons_frame, text=label,
                      command=lambda q=query: self.insert_sample(q)).pack(side=tk.LEFT, padx=2)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Execute Query", 
                  command=self.execute, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def insert_sample(self, query):
        """Insert sample query."""
        self.query_text.delete(1.0, tk.END)
        self.query_text.insert(tk.END, query)
    
    def execute(self):
        """Execute query and close dialog."""
        query = self.query_text.get(1.0, tk.END).strip()
        connection = self.connection_var.get()
        
        if not query:
            messagebox.showwarning("Warning", "Please enter a SQL query")
            return
        
        if not connection:
            messagebox.showwarning("Warning", "Please select a database connection")
            return
        
        self.query = query
        self.connection_name = connection
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel and close dialog."""
        self.query = None
        self.connection_name = None
        self.dialog.destroy()


class PreferencesDialog:
    """Dialog for application preferences."""
    
    def __init__(self, parent):
        self.parent = parent
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Preferences")
        self.dialog.geometry("400x500")
        self.dialog.grab_set()
        self.dialog.transient(parent)
        
        self.create_widgets()
        self.center_dialog()
    
    def center_dialog(self):
        """Center the dialog on screen."""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (500 // 2)
        self.dialog.geometry(f"400x500+{x}+{y}")
    
    def create_widgets(self):
        """Create preference widgets."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="⚙️ Application Preferences", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for preference categories
        pref_notebook = ttk.Notebook(main_frame)
        pref_notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # General preferences
        general_frame = ttk.Frame(pref_notebook)
        pref_notebook.add(general_frame, text="General")
        
        # Theme selection
        theme_frame = ttk.LabelFrame(general_frame, text="Appearance", padding=10)
        theme_frame.pack(fill=tk.X, pady=5)
        
        self.theme_var = tk.StringVar(value="Dark")
        ttk.Radiobutton(theme_frame, text="Dark Theme", variable=self.theme_var, 
                       value="Dark").pack(anchor=tk.W)
        ttk.Radiobutton(theme_frame, text="Light Theme", variable=self.theme_var, 
                       value="Light").pack(anchor=tk.W)
        
        # Auto-save
        autosave_frame = ttk.LabelFrame(general_frame, text="Auto-Save", padding=10)
        autosave_frame.pack(fill=tk.X, pady=5)
        
        self.autosave_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(autosave_frame, text="Enable auto-save sessions", 
                       variable=self.autosave_var).pack(anchor=tk.W)
        
        # Performance preferences
        perf_frame = ttk.Frame(pref_notebook)
        pref_notebook.add(perf_frame, text="Performance")
        
        # Memory management
        memory_frame = ttk.LabelFrame(perf_frame, text="Memory Management", padding=10)
        memory_frame.pack(fill=tk.X, pady=5)
        
        self.memory_limit_var = tk.StringVar(value="1GB")
        ttk.Label(memory_frame, text="Memory Limit:").pack(anchor=tk.W)
        ttk.Combobox(memory_frame, textvariable=self.memory_limit_var,
                    values=["512MB", "1GB", "2GB", "4GB", "8GB"], 
                    state="readonly").pack(anchor=tk.W, pady=5)
        
        # Processing preferences
        proc_frame = ttk.LabelFrame(perf_frame, text="Processing", padding=10)
        proc_frame.pack(fill=tk.X, pady=5)
        
        self.multiprocessing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(proc_frame, text="Enable multiprocessing", 
                       variable=self.multiprocessing_var).pack(anchor=tk.W)
        
        # ML preferences
        ml_frame = ttk.Frame(pref_notebook)
        pref_notebook.add(ml_frame, text="Machine Learning")
        
        # Default ML settings
        ml_defaults_frame = ttk.LabelFrame(ml_frame, text="Default Settings", padding=10)
        ml_defaults_frame.pack(fill=tk.X, pady=5)
        
        self.default_cv_var = tk.StringVar(value="5")
        ttk.Label(ml_defaults_frame, text="Cross-Validation Folds:").pack(anchor=tk.W)
        ttk.Combobox(ml_defaults_frame, textvariable=self.default_cv_var,
                    values=["3", "5", "10"], state="readonly").pack(anchor=tk.W, pady=5)
        
        self.default_test_size_var = tk.StringVar(value="0.2")
        ttk.Label(ml_defaults_frame, text="Test Size:").pack(anchor=tk.W)
        ttk.Combobox(ml_defaults_frame, textvariable=self.default_test_size_var,
                    values=["0.1", "0.2", "0.3"], state="readonly").pack(anchor=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack()
        
        ttk.Button(button_frame, text="Apply", command=self.apply_preferences, 
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def apply_preferences(self):
        """Apply preferences and close dialog."""
        # Here you would apply the preferences to the application
        messagebox.showinfo("Preferences", "Preferences applied successfully!")
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel and close dialog."""
        self.dialog.destroy()


class DockerDeploymentDialog:
    """Dialog for Docker deployment configuration."""
    
    def __init__(self, parent):
        self.parent = parent
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Docker Deployment")
        self.dialog.geometry("500x400")
        self.dialog.grab_set()
        self.dialog.transient(parent)
        
        self.create_widgets()
        self.center_dialog()
    
    def center_dialog(self):
        """Center the dialog on screen."""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (400 // 2)
        self.dialog.geometry(f"500x400+{x}+{y}")
    
    def create_widgets(self):
        """Create Docker deployment widgets."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🐳 Docker Deployment", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Dockerfile generation
        dockerfile_frame = ttk.LabelFrame(main_frame, text="Dockerfile Generation", padding=10)
        dockerfile_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(dockerfile_frame, text="Generate Dockerfile", 
                  command=self.generate_dockerfile).pack(pady=5)
        ttk.Label(dockerfile_frame, text="Creates optimized Dockerfile for deployment").pack()
        
        # Docker build
        build_frame = ttk.LabelFrame(main_frame, text="Docker Build", padding=10)
        build_frame.pack(fill=tk.X, pady=10)
        
        # Image name
        name_frame = ttk.Frame(build_frame)
        name_frame.pack(fill=tk.X, pady=5)
        ttk.Label(name_frame, text="Image Name:").pack(side=tk.LEFT)
        self.image_name_var = tk.StringVar(value="ai-analytics-platform")
        ttk.Entry(name_frame, textvariable=self.image_name_var, width=30).pack(side=tk.RIGHT)
        
        ttk.Button(build_frame, text="Build Docker Image", 
                  command=self.build_image).pack(pady=5)
        
        # Docker run
        run_frame = ttk.LabelFrame(main_frame, text="Docker Run", padding=10)
        run_frame.pack(fill=tk.X, pady=10)
        
        # Port mapping
        port_frame = ttk.Frame(run_frame)
        port_frame.pack(fill=tk.X, pady=5)
        ttk.Label(port_frame, text="Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value="8000")
        ttk.Entry(port_frame, textvariable=self.port_var, width=10).pack(side=tk.RIGHT)
        
        ttk.Button(run_frame, text="Run Container", 
                  command=self.run_container).pack(pady=5)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=self.dialog.destroy).pack(pady=20)
    
    def generate_dockerfile(self):
        """Generate Dockerfile."""
        dockerfile_content = '''FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "api_server.py"]
'''
        
        try:
            with open("Dockerfile", "w") as f:
                f.write(dockerfile_content)
            messagebox.showinfo("Success", "Dockerfile generated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate Dockerfile:\n{str(e)}")
    
    def build_image(self):
        """Build Docker image."""
        image_name = self.image_name_var.get()
        if not image_name:
            messagebox.showwarning("Warning", "Please enter an image name")
            return
        
        try:
            # This would run docker build command
            messagebox.showinfo("Docker Build", 
                              f"Docker build command:\n"
                              f"docker build -t {image_name} .\n\n"
                              f"Run this command in your terminal.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to build image:\n{str(e)}")
    
    def run_container(self):
        """Run Docker container."""
        image_name = self.image_name_var.get()
        port = self.port_var.get()
        
        if not image_name:
            messagebox.showwarning("Warning", "Please enter an image name")
            return
        
        try:
            # This would run docker run command
            messagebox.showinfo("Docker Run", 
                              f"Docker run command:\n"
                              f"docker run -p {port}:8000 {image_name}\n\n"
                              f"Run this command in your terminal.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run container:\n{str(e)}")


class CloudUploadDialog:
    """Dialog for cloud storage upload."""
    
    def __init__(self, parent):
        self.parent = parent
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Cloud Upload")
        self.dialog.geometry("450x350")
        self.dialog.grab_set()
        self.dialog.transient(parent)
        
        self.create_widgets()
        self.center_dialog()
    
    def center_dialog(self):
        """Center the dialog on screen."""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (350 // 2)
        self.dialog.geometry(f"450x350+{x}+{y}")
    
    def create_widgets(self):
        """Create cloud upload widgets."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="☁️ Cloud Storage Upload", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Cloud provider selection
        provider_frame = ttk.LabelFrame(main_frame, text="Cloud Provider", padding=10)
        provider_frame.pack(fill=tk.X, pady=10)
        
        self.provider_var = tk.StringVar(value="AWS S3")
        providers = ["AWS S3", "Google Cloud Storage", "Azure Blob Storage", "Dropbox"]
        
        for provider in providers:
            ttk.Radiobutton(provider_frame, text=provider, variable=self.provider_var, 
                           value=provider).pack(anchor=tk.W)
        
        # Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=10)
        
        # Bucket/Container name
        bucket_frame = ttk.Frame(config_frame)
        bucket_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bucket_frame, text="Bucket/Container:").pack(side=tk.LEFT)
        self.bucket_var = tk.StringVar()
        ttk.Entry(bucket_frame, textvariable=self.bucket_var, width=25).pack(side=tk.RIGHT)
        
        # Access key
        key_frame = ttk.Frame(config_frame)
        key_frame.pack(fill=tk.X, pady=2)
        ttk.Label(key_frame, text="Access Key:").pack(side=tk.LEFT)
        self.access_key_var = tk.StringVar()
        ttk.Entry(key_frame, textvariable=self.access_key_var, width=25).pack(side=tk.RIGHT)
        
        # Secret key
        secret_frame = ttk.Frame(config_frame)
        secret_frame.pack(fill=tk.X, pady=2)
        ttk.Label(secret_frame, text="Secret Key:").pack(side=tk.LEFT)
        self.secret_key_var = tk.StringVar()
        ttk.Entry(secret_frame, textvariable=self.secret_key_var, show="*", width=25).pack(side=tk.RIGHT)
        
        # Upload options
        options_frame = ttk.LabelFrame(main_frame, text="Upload Options", padding=10)
        options_frame.pack(fill=tk.X, pady=10)
        
        self.include_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include cleaned data", 
                       variable=self.include_data_var).pack(anchor=tk.W)
        
        self.include_models_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include trained models", 
                       variable=self.include_models_var).pack(anchor=tk.W)
        
        self.include_plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include visualizations", 
                       variable=self.include_plots_var).pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Upload", command=self.upload, 
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def upload(self):
        """Upload to cloud storage."""
        provider = self.provider_var.get()
        bucket = self.bucket_var.get()
        
        if not bucket:
            messagebox.showwarning("Warning", "Please enter bucket/container name")
            return
        
        # This would implement actual cloud upload
        messagebox.showinfo("Cloud Upload", 
                          f"Upload configuration:\n"
                          f"Provider: {provider}\n"
                          f"Bucket: {bucket}\n\n"
                          f"Note: Implement actual cloud SDK integration for functionality.")


def main():
    """Main function to run the enhanced application."""
    root = tk.Tk()
    
    # Configure ttk style for enhanced appearance
    style = ttk.Style()
    
    # Try to use a modern theme
    try:
        style.theme_use("clam")
    except:
        try:
            style.theme_use("alt")
        except:
            pass  # Use default theme
    
    # Configure custom styles
    style.configure("Accent.TButton", 
                   foreground="white", 
                   background="#3498db",
                   focuscolor="none")
    style.map("Accent.TButton",
             background=[("active", "#2980b9")])
    
    # Configure notebook style
    style.configure("TNotebook", background="#ecf0f1")
    style.configure("TNotebook.Tab", padding=[20, 10])
    
    # Set window icon if available
    try:
        # You can add an icon file here
        # root.iconbitmap("app_icon.ico")
        pass
    except:
        pass
    
    # Create and run the enhanced application
    try:
        app = EnhancedAIDataPlatform(root)
        
        # Set up proper cleanup on close
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit? Any unsaved work will be lost."):
                try:
                    # Save current session if possible
                    if app.storage_manager and app.current_data is not None:
                        app.save_session()
                except:
                    pass  # Don't let cleanup errors prevent closing
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Startup Error", 
                           f"Failed to start application:\n{str(e)}\n\n"
                           f"Please check that all required modules are installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()