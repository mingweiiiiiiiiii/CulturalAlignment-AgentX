#!/bin/bash

echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing requirements..."
pip install --upgrade pip
pip install transformers langgraph scikit-learn matplotlib

echo "🚀 Running culturally-aware dialogue system..."
python main.py