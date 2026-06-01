#!/bin/bash
set -e

echo "=== Creating venv for EPUB Translator Pipeline ==="

# Create virtual environment
python -m venv venv

# Activate and upgrade pip
source venv/bin/activate
pip install --upgrade pip

# Install core dependencies (no torch needed for validation)
pip install beautifulsoup4 lxml tomli

# Install pytest for testing framework
pip install pytest

echo ""
echo "[OK] Base dependencies installed!"
echo "=== Installation Summary ==="
pip list

