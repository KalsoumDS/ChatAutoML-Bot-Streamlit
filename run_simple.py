#!/usr/bin/env python3
"""
Script de lancement pour l'interface simple de TabularAI
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "app" / "streamlit_app_simple.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
