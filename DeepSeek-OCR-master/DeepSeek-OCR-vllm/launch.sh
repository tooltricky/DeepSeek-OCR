#!/bin/bash

# DeepSeek-OCR Web Interface Launcher

echo "========================================"
echo "  DeepSeek-OCR Web Interface Launcher  "
echo "========================================"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "deepseek-ocr" ]]; then
    echo "Warning: deepseek-ocr conda environment is not activated."
    echo "Attempting to activate..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate deepseek-ocr

    if [[ "$CONDA_DEFAULT_ENV" != "deepseek-ocr" ]]; then
        echo "Error: Failed to activate deepseek-ocr environment."
        echo "Please manually activate it with: conda activate deepseek-ocr"
        exit 1
    fi
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting DeepSeek-OCR Web Interface..."
echo "The interface will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch streamlit
streamlit run app.py
