#!/usr/bin/env python3
"""
AI Data Analysis Platform Launcher
This script will install minimal requirements and launch the application
"""
import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def install_minimal_requirements():
    """Install minimal requirements to get the app running"""
    requirements_file = Path("requirements_minimal.txt")
    
    if requirements_file.exists():
        print("Installing minimal requirements...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements_minimal.txt"
            ])
            print("✅ Minimal requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
            return False
    else:
        print("requirements_minimal.txt not found")
        return False

def launch_app():
    """Launch the AI Data Analysis Platform"""
    try:
        print("🚀 Launching AI Data Analysis Platform...")
        print("Note: Some advanced features may be disabled due to missing dependencies")
        print("To enable all features, install the full requirements.txt later")
        print("-" * 60)
        
        # Import and run the main application
        import main
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install the missing dependencies")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

def main():
    print("🧠 AI Data Analysis Platform Launcher")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Install minimal requirements
    if not install_minimal_requirements():
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install -r requirements_minimal.txt")
        sys.exit(1)
    
    # Launch the application
    launch_app()

if __name__ == "__main__":
    main()