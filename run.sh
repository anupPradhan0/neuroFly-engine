#!/bin/bash
# BioFly-Arcade Engine Launcher
# Automatically uses the correct Python environment where dependencies are installed

PYTHON_BIN="/home/mors/.local/share/mise/installs/python/3.10.19/bin/python3"

echo "Launching BioFly Dashboard..."
$PYTHON_BIN run_dashboard.py
