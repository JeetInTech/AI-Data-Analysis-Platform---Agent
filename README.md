# 🤖 AI Data Analysis Platform

> **Transform any CSV dataset into intelligent insights with automated ML, beautiful visualizations, natural language instructions, and autonomous agent-based processing**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Agent Mode with Instructions](#-agent-mode-with-instructions)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🎯 Overview

The **AI Data Analysis Platform** is a comprehensive, GUI-based machine learning system that provides:

- 🎯 **Natural Language Instructions** - Guide the AI with simple commands
- 🤖 **Autonomous Agent Mode** - Fully automated ML pipeline
- 📊 **Smart Data Processing** - Intelligent cleaning and preprocessing
- 🧠 **Advanced ML Models** - XGBoost, LightGBM, CatBoost, Neural Networks
- 🎨 **Beautiful Visualizations** - Interactive charts and plots
- 💬 **Q&A System** - Ask questions about your data and models
- 📚 **Self-Learning** - Agent improves from experience

Perfect for data scientists, ML engineers, and analysts who want an intelligent assistant for end-to-end machine learning workflows.

---

## 🌟 Key Features

### 🎯 **NEW: Natural Language Instructions (ReAct Pattern)**

Give the agent instructions before it processes your data:

```
"I only want house prices"
"Just clean the data, no training"
"Predict sales using only the top features"
"Focus on correlation analysis"
```

**How it works:**
- **Thought**: Agent reasons about your instruction
- **Action**: Agent creates a custom execution plan
- **Observation**: Agent reflects on results

**Benefits:**
- ✅ Control what the agent does
- ✅ Skip unnecessary steps
- ✅ Focus on specific analyses
- ✅ Filter data automatically
- ✅ Select target columns easily

### 🔄 **Smart Data Processing**

- **Automatic CSV loading** with intelligent encoding detection
- **Intelligent data cleaning**: Missing values, outliers, duplicates
- **Smart preprocessing**: Feature engineering, encoding, scaling
- **Data type detection**: Numeric, categorical, datetime
- **Feature selection**: Automatic selection of best features
- **Quality scoring**: 0-100 data quality metrics

### 🤖 **Advanced Machine Learning**

- **Multiple algorithms**: 
  - Random Forest, XGBoost, LightGBM, CatBoost
  - Gradient Boosting, Neural Networks
  - SVM, Ridge, Lasso, Linear models
- **Ensemble methods**: Voting and Stacking
- **Automatic hyperparameter tuning**
- **Cross-validation** with statistical significance
- **Problem detection**: Auto-detect classification vs regression

### 📊 **Beautiful Visualizations**

- **Data overview**: Shape, missing values, distributions, correlations
- **Target analysis**: Class distributions, statistical summaries
- **Model performance**: Comparison charts, confusion matrices, ROC curves
- **Feature importance**: Interactive importance plots
- **Neural network visualization**: 3D architecture rendering
- **Custom plots**: Distribution, correlation, box plots, scatter plots

### 🔍 **Natural Language Querying**

Ask questions in plain English:
- *"What is the model accuracy?"*
- *"Show feature importance"*
- *"Which model performed best?"*
- *"What is the R² score?"*
- *"Compare all models"*

### 🤖 **Autonomous Agent Mode**

The agent handles everything automatically:
- **Intelligent Planning**: Analyzes dataset and creates optimal strategy
- **Adaptive Execution**: Adjusts based on data characteristics
- **Error Recovery**: Retries with fallback strategies
- **Quality Control**: Validates each step
- **Self-Learning**: Improves from experience
- **Real-time Monitoring**: Live progress updates

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd "AI-Data-Analysis-Platform"

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

### 3. Basic Workflow

1. **Load Data**: Click "📁 Load CSV File" and select your dataset
2. **Launch Agent Mode**: Click "🚀 [LAUNCH] AGENT MODE"
3. **Give Instructions** (Optional): 
   - Type instructions like "I only want house prices"
   - Or click "Skip" for fully autonomous mode
4. **Watch the Magic**: Agent analyzes, cleans, trains models, and visualizes
5. **Ask Questions**: Use the Q&A interface after completion

---

## 🎯 Agent Mode with Instructions

### How to Use

1. **Load your dataset** first
2. Click **"🚀 [LAUNCH] AGENT MODE"**
3. **Instruction dialog appears** with examples
4. **Type your instructions** or click "Skip"
5. **Watch the agent work** with your guidance

### Instruction Examples

#### 🔹 Data Filtering
```
"I only want house prices"
"Focus on California real estate data"
"Filter for houses with price > 200000"
"Only include records where bedrooms >= 3"
```

#### 🔹 Data Cleaning Only
```
"Just clean the data, no training"
"Only preprocessing, no models"
"I just want cleaned data"
```

#### 🔹 Prediction Tasks
```
"Predict house prices"
"I want to forecast sales"
"Target variable: price"
"Build a model to predict customer churn"
```

#### 🔹 Analysis Focus
```
"Focus on correlation analysis"
"Analyze feature importance"
"Show me the relationships between variables"
```

#### 🔹 Column Selection
```
"Only use price, bedrooms, and location columns"
"Remove ID and timestamp columns"
"Focus on numerical features only"
```

#### 🔹 Complex Instructions
```
"Clean the data, focus on houses > 200k, then predict price"
"Filter to California, remove outliers, and run classification"
"Preprocess everything, create visualizations, but skip training"
```

### ReAct Reasoning Pattern

The agent uses **ReAct (Reasoning + Acting)** for intelligent decision-making:

```
📝 User instruction: "Predict house prices"

🧠 THOUGHT:
   "User wants to predict the 'price' column.
    This is a regression task.
    Should focus on price-related features."

⚡ ACTION:
   "Filter to price-related columns.
    Apply standard data cleaning.
    Train regression models (XGBoost, Random Forest)."

👁️ OBSERVATION:
   "Dataset filtered to 4 relevant columns.
    500 rows after cleaning.
    Best model: XGBoost (R² = 0.89)"
```

### Skip for Auto Mode

Click **"Skip"** to let the agent work fully autonomously:
- ✅ Full intelligent pipeline runs
- ✅ Data analysis and cleaning
- ✅ Feature engineering
- ✅ ML model training (XGBoost, LightGBM, CatBoost, etc.)
- ✅ Model evaluation and comparison
- ✅ Visualizations and exports

---

## 💡 Usage Examples

### Example 1: Quick Analysis
```python
# Just run the app and load data
python main.py
# Load CSV → Launch Agent → Skip (auto mode)
# Agent does everything automatically!
```

### Example 2: With Instructions
```python
# Load data → Launch Agent
# Type: "I only want house prices above 200k"
# Agent filters and focuses on expensive houses
```

### Example 3: Clean Data Only
```python
# Load data → Launch Agent  
# Type: "Just clean the data, no training"
# Agent cleans and exports, skips ML training
```

### Example 4: Custom Prediction
```python
# Load data → Launch Agent
# Type: "Predict diagnosis using only mean features"
# Agent selects target and specific features
```

---

## 📁 Project Structure

```
AI-Data-Analysis-Platform/
│
├── 📱 CORE APPLICATION
│   ├── main.py                      # Main GUI application
│   ├── launcher.py                  # Alternative launcher
│   └── pygame_ui.py                 # Alternative UI
│
├── 🤖 AGENT & ML CORE
│   ├── agent_mode.py                # Autonomous agent (1500+ lines)
│   ├── instruction_handler.py       # NEW: Natural language parser (520+ lines)
│   ├── ml_engine.py                 # Machine learning engine
│   ├── auto_selector.py             # Automatic model selection
│   └── api_server.py                # REST API server
│
├── 📊 DATA PROCESSING
│   ├── data_processor.py            # Data cleaning and preprocessing
│   └── enhanced_data_processor.py   # Advanced data processing
│
├── 🎨 VISUALIZATION & EXPLAINABILITY
│   ├── neural_visualizer.py         # Neural network 3D visualization
│   ├── advanced_visualization.py    # Advanced plotting functions
│   └── explainability.py            # SHAP, LIME, feature importance
│
├── 💾 STORAGE & MANAGEMENT
│   └── storage_manager.py           # Session and data persistence
│
├── 📋 CONFIGURATION
│   ├── requirements.txt             # Full dependencies
│   ├── requirements_minimal.txt     # Minimal dependencies
│   ├── .gitignore                   # Git ignore rules
│   └── README.md                    # This file
│
└── 🗄️ STORAGE (Auto-generated)
    └── ai_analytics_storage/
        ├── agent_knowledge_base.json   # Agent's learned knowledge
        ├── sessions/                    # Saved sessions
        ├── models/                      # Trained models
        ├── visualizations/              # Generated plots
        ├── exports/                     # Exported results
        ├── logs/                        # Application logs
        └── cache/                       # Cached data
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

### Quick Install

```bash
# 1. Clone the repository
git clone <repository-url>
cd "AI-Data-Analysis-Platform"

# 2. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

**Core Libraries:**
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning algorithms
- matplotlib, seaborn, plotly - Visualizations
- tkinter - GUI framework

**Advanced ML:**
- xgboost, lightgbm, catboost - Gradient boosting
- shap, lime - Model explainability
- tensorflow/torch - Neural networks (optional)

**Agent Features:**
- langchain - Context awareness (optional)
- networkx - Graph visualizations

---

## 🚨 Troubleshooting

### Common Issues

#### **1. Import Errors**
```bash
# Solution: Install missing packages
pip install -r requirements.txt
```

#### **2. Agent Mode Only Cleans Data (Skip Button)**
**Fixed!** The latest version properly checks if instructions are provided:
- With instructions: Custom pipeline
- Skip button (no instructions): Full ML pipeline with training

#### **3. Memory Errors with Large Datasets**
```bash
# Solution: Use sampling or Quick Mode
# In the app, enable "Quick Mode" for faster processing
```

#### **4. Visualization Not Showing**
```bash
# Solution: Check matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
```

#### **5. Training Fails**
```bash
# Check logs in: ai_analytics_storage/logs/
# Ensure target column is properly selected
# Try Quick Mode first for debugging
```

### Getting Help

- **Issues**: Open an issue on GitHub
- **Logs**: Check `ai_analytics_storage/logs/` for detailed error logs
- **Documentation**: Review inline code documentation
- **Tests**: Run `python test_instructions.py` to verify functionality

---

## 📊 Performance Tips

### For Large Datasets (>100K rows)
- Use sampling for initial exploration
- Enable parallel processing (`n_jobs=-1`)
- Consider feature selection to reduce dimensions
- Use Quick Mode for faster training

### For Fast Training
- Use simpler models first (Logistic Regression, Linear)
- Reduce cross-validation folds
- Skip hyperparameter tuning initially

### For Better Accuracy
- Enable feature engineering
- Use ensemble methods (Voting, Stacking)
- Perform hyperparameter tuning
- Try neural networks for complex patterns

---

## 🧪 Testing

### Run Instruction Feature Tests

```bash
# Test the new instruction parsing feature
python test_instructions.py

# Expected output:
# Total Tests: 24
# ✅ Passed: 23 (95.8%)
```

### Test Cases Covered

- ✅ Data filtering instructions
- ✅ Clean-only mode
- ✅ Prediction tasks
- ✅ Analysis focus
- ✅ Target column extraction
- ✅ Complex filters
- ✅ ReAct reasoning generation

---

## 🗺️ Roadmap

### Current Status (v1.0)
- ✅ Natural language instructions
- ✅ ReAct-style reasoning
- ✅ Autonomous agent mode
- ✅ Advanced ML models
- ✅ Interactive visualizations
- ✅ Q&A system
- ✅ Self-learning knowledge base

### Upcoming Features (v1.1+)
- [ ] Enhanced LangChain integration
- [ ] Instruction templates and presets
- [ ] Conversation memory
- [ ] Time series forecasting
- [ ] NLP text analysis
- [ ] Computer vision support
- [ ] Web-based dashboard (Streamlit)
- [ ] Cloud deployment (AWS, Azure, GCP)
- [ ] Real-time streaming data
- [ ] Collaborative features

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests**: Update `test_instructions.py` for new features
5. **Commit**: `git commit -m "Add amazing feature"`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation in README
- Test with different datasets

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **scikit-learn**: Core ML algorithms
- **XGBoost/LightGBM/CatBoost**: Gradient boosting implementations
- **SHAP/LIME**: Model explainability
- **Plotly**: Interactive visualizations
- **Tkinter**: GUI framework
- **pandas/numpy**: Data manipulation
- **LangChain**: Context awareness and reasoning patterns

---

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the usage examples
- Read the inline documentation

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**Made with ❤️ by Data Scientists, for Data Scientists**

**Transform your data into insights with the power of AI!** 🚀

---

© 2025 AI Data Analysis Platform. All rights reserved.

</div>
