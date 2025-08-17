# Cyberbullying Detection System

![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive machine learning system for detecting cyberbullying in social media texts using multiple models including traditional machine learning and state-of-the-art deep learning approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Web Application](#web-application)
- [Monitoring with Grafana](#monitoring-with-grafana)
- [License](#license)

## 📝 Overview

This project implements various machine learning and deep learning approaches to detect cyberbullying content in text messages. The system includes:

- Traditional machine learning models (Logistic Regression)
- Neural network models (BiLSTM with GloVe embeddings)
- Transformer-based models (BERT)
- Novel emotion-aware fusion model (RoBERTa + CNN + emotion features)

The system has been trained and evaluated on multiple cyberbullying datasets including Davidson and Vidgen datasets, and offers a web-based interface for testing and deployment.

## 🧠 Models

The project includes four main model types:

1. **Logistic Regression**: A baseline model using TF-IDF features for text classification.
2. **BiLSTM**: A bidirectional LSTM neural network using GloVe word embeddings.
3. **BERT**: A fine-tuned BERT model leveraging contextual embeddings.
4. **Emotion Fusion Model**: Our advanced model combining RoBERTa embeddings with CNN layers and emotion features for improved performance.

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Kishor978/cyber_bullying.git
cd cyber_bullying
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download GloVe embeddings (if not already included):
```bash
# Download GloVe embeddings and place them in the glove/ directory
# The project uses glove.6B.100d.txt
```

## 💻 Usage

### Running Experiments

You can run different experiments using the `run_experiments.py` script:

```bash
# Run a specific model experiment
python run_experiments.py --baseline  # Logistic Regression
python run_experiments.py --bert      # BERT model
python run_experiments.py --bilstm    # BiLSTM model
python run_experiments.py --emotion   # Emotion Fusion model

# Run all experiments
python run_experiments.py --all

# Save output to a log file
python run_experiments.py --all > all_experiments_output.log 2>&1
```

### Web Application

To run the Streamlit web application:

```bash
streamlit run deployment/app.py
```

The application will be available at `http://localhost:8501`.

## 📊 Dataset

This project uses multiple cyberbullying datasets:

1. **Davidson Dataset**: A Twitter dataset with hate speech, offensive language, and neutral content.
2. **Vidgen Dataset**: A dynamically generated hate speech dataset.
3. **Combined Dataset**: A merged dataset for more comprehensive training.

The datasets can be accessed from Mendeley Data: [https://data.mendeley.com/datasets/wmx9jj2htd/2](https://data.mendeley.com/datasets/wmx9jj2htd/2)

## 📈 Evaluation

The models are evaluated using multiple metrics:

- Accuracy
- Precision, Recall, and F1-score
- Confusion matrices

Evaluation results and visualizations are saved in the `results/` directory.

## 🌐 Web Application

The project includes a user-friendly Streamlit web application that allows:

- Real-time text classification with any of the trained models
- Batch processing of texts from CSV files
- Visualization of classification results
- Detailed explanation of model decisions

### Using the Web App

1. Select a model from the sidebar
2. Enter text in the input field or upload a CSV file
3. View the classification results and confidence scores

## � Monitoring with Grafana

This project utilizes Grafana for real-time monitoring and visualization of model performance metrics and usage analytics. Grafana provides powerful dashboards to track:

- Model prediction accuracy over time
- System load and performance metrics
- Detection rates and false positives
- User interaction patterns

### Setup Grafana

1. Install Grafana:
```bash
# For Docker installation
docker run -d -p 3000:3000 --name=grafana grafana/grafana-oss

# For other installation methods, visit: https://grafana.com/docs/grafana/latest/setup-grafana/installation/
```

2. Configure data sources:
```bash
# Install InfluxDB or Prometheus for metrics storage
docker run -d -p 8086:8086 --name=influxdb influxdb

# Add Python logging to send metrics
pip install influxdb-client
```

3. Import the Cyberbullying Detection dashboard:
```bash
# Available in the monitoring/ directory
```

### Integration with the Application

The model predictor has been instrumented to send performance metrics to InfluxDB:
- Prediction latency
- Confidence scores
- Classification results
- System resource utilization

Access the Grafana dashboard at `http://localhost:3000` (default credentials: admin/admin)

## �📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Kishor978 (Project Lead)

## 🙏 Acknowledgements

- The creators of the Davidson and Vidgen datasets
- PyTorch, Hugging Face, and scikit-learn communities
- GloVe embeddings team
```