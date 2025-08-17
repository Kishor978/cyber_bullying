# Cyberbullying Detection System - Deployment Guide

This guide explains how to deploy the Cyberbullying Detection System using Streamlit.

## Prerequisites

Before deploying the application, make sure you have:

- Python 3.9 or higher
- pip package manager
- Required Python packages (install using `pip install -r requirements.txt`)
- Trained model files (or run the experiments first)

## Local Deployment

### Step 1: Set Up the Environment

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-username/cyberbullying-detection.git
cd cyberbullying-detection

# Install required packages
pip install -r requirements.txt
```


### Step 2: Run the Streamlit App

```bash
# Or run directly with streamlit
streamlit run deployment/app.py
```

The app will be available at `http://localhost:8501`


## Usage

### Single Text Analysis

1. Select a model from the sidebar
2. Enter text in the input field
3. Click "Analyze Text"
4. View the results and explanation

### Batch Analysis

1. Prepare a CSV file with a column containing texts to analyze
2. Upload the file through the "Batch Analysis" section
3. Click "Run Batch Analysis"
4. Download the results as a CSV file

