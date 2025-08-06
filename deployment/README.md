# Cyberbullying Detection System - Deployment Guide

This guide explains how to deploy the Cyberbullying Detection System using Streamlit.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Usage](#usage)
5. [Advanced Configuration](#advanced-configuration)
6. [Troubleshooting](#troubleshooting)

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

### Step 2: Deploy the Models

Before running the Streamlit app, you need to deploy the trained models:

```bash
# Deploy all models
python deploy_models.py --all

# Or deploy specific models
python deploy_models.py --bert    # Deploy only BERT model
python deploy_models.py --bilstm  # Deploy only BiLSTM model
python deploy_models.py --emotion # Deploy only Emotion Fusion model
```

### Step 3: Run the Streamlit App

```bash
# Run using the convenience script
python run_streamlit_app.py

# Or run directly with streamlit
streamlit run deployment/app.py
```

The app will be available at `http://localhost:8501`

## Docker Deployment

### Step 1: Build the Docker Image

```bash
docker build -t cyberbullying-detection:latest .
```

### Step 2: Run the Docker Container

```bash
docker run -p 8501:8501 cyberbullying-detection:latest
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

## Advanced Configuration

### Changing the Port

```bash
# Run on a different port (e.g., 8502)
python run_streamlit_app.py --port 8502
```

### Using a Specific Model

```bash
# Deploy and use only the Emotion Fusion model
python run_streamlit_app.py --emotion
```

### Skipping Model Deployment

```bash
# Skip model deployment (useful if models are already deployed)
python run_streamlit_app.py --skip-deploy
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install the required package using `pip install <package_name>`

2. **CUDA out of memory**: If using a GPU, try reducing batch size or switch to CPU by setting `device = torch.device('cpu')` in the code

3. **Model not found**: Ensure you've deployed the models using `deploy_models.py` before running the app

4. **Port already in use**: Change the port using `--port` parameter or kill the process using the current port

### Getting Help

If you encounter issues not covered here, please:

1. Check the project issues page on GitHub
2. Consult the detailed documentation in the `docs` folder
3. Contact the project maintainers

## License

This project is licensed under the terms of the MIT license.
