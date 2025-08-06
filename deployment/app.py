import os
import sys
# Add project root to path so we can import modules - MUST BE BEFORE OTHER IMPORTS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from model_predictor import ModelPredictor
from healthcheck import start_healthcheck_server

# Start health check server for Docker deployment

# App title and description
st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS if it exists
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
if os.path.exists(css_path):
    local_css(css_path)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.current_model = None

# Sidebar for model selection
st.sidebar.title("🛡️ Cyberbullying Detection")
st.sidebar.markdown("### Model Selection")

model_type = st.sidebar.selectbox(
    "Choose a model:",
    ["Emotion Fusion Model", "BERT Model", "BiLSTM Model", "Logistic Regression Model"],
    index=0
)

# Map display names to model types
model_type_map = {
    "Emotion Fusion Model": "emotion_fusion",
    "BERT Model": "bert",
    "BiLSTM Model": "bilstm",
    "Logistic Regression Model": "logistic"
}

selected_model = model_type_map[model_type]

# Load model if needed
if not st.session_state.model_loaded or st.session_state.current_model != selected_model:
    with st.spinner(f"Loading {model_type}..."):
        try:
            st.session_state.predictor = ModelPredictor(model_type=selected_model)
            st.session_state.model_loaded = True
            st.session_state.current_model = selected_model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.session_state.model_loaded = False

# Sidebar model information
with st.sidebar.expander("📖 Model Information"):
    if selected_model == "emotion_fusion":
        st.markdown("""
        ### Emotion Fusion Model
        This is our most advanced model that combines RoBERTa embeddings with CNN layers and emotion features.
        
        **Features:**
        - Emotion-aware classification
        - Deep learning architecture
        - Best accuracy among all models
        """)
    elif selected_model == "bert":
        st.markdown("""
        ### BERT Model
        A fine-tuned BERT model for cyberbullying detection.
        
        **Features:**
        - Contextual word embeddings
        - Transformer architecture
        - Good performance on various text styles
        """)
    elif selected_model == "bilstm":
        st.markdown("""
        ### BiLSTM Model
        A bidirectional LSTM model using GloVe embeddings.
        
        **Features:**
        - Sequential text processing
        - Word embeddings from GloVe
        - Captures text dependencies
        """)
    elif selected_model == "logistic":
        st.markdown("""
        ### Logistic Regression Model
        A baseline model using TF-IDF features.
        
        **Features:**
        - Simple and interpretable
        - Fast inference
        - Good baseline performance
        """)

# Sidebar history section
with st.sidebar.expander("📊 Analysis History"):
    if st.button("Clear History"):
        st.session_state.history = []
    
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            st.markdown(f"**{i+1}. {entry['text'][:50]}...**")
            st.markdown(f"Prediction: {entry['result']['prediction_text']} ({entry['result']['confidence']:.2%})")
            st.markdown("---")

# Main content
st.title("🛡️ Cyberbullying Detection System")
st.markdown("### Detect and analyze potential cyberbullying in text")

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    text_input = st.text_area("Enter text to analyze:", height=150, key="text_input")
    
with col2:
    st.markdown("### Options")
    show_confidence = st.checkbox("Show confidence score", value=True)
    show_explanation = st.checkbox("Show explanation", value=True)

# Analysis button
if st.button("Analyze Text"):
    if not text_input:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Predict
            result = st.session_state.predictor.predict(text_input)
            
            # Add to history
            st.session_state.history.append({
                'text': text_input,
                'result': result
            })
            
            # Display result
            st.markdown("## Results")
            
            # Display with styled box
            if result['prediction'] == 1:  # Bullying
                st.error(f"🚨 **{result['prediction_text']}**")
            else:  # Non-bullying
                st.success(f"✅ **{result['prediction_text']}**")
                
            # Show confidence if selected
            if show_confidence:
                confidence = result['confidence']
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Confidence visualization
                fig, ax = plt.subplots(figsize=(10, 2))
                colors = ['#d4edda', '#f8d7da'] if result['prediction'] == 0 else ['#f8d7da', '#d4edda']
                sns.barplot(x=[confidence, 1-confidence], palette=colors, ax=ax)
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                for i, p in enumerate(ax.patches):
                    txt = "Non-bullying" if i == 0 and result['prediction'] == 0 or i == 1 and result['prediction'] == 1 else "Bullying"
                    ax.annotate(f"{txt}: {p.get_width():.2%}",
                                (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.),
                                ha='center', va='center', fontsize=11, fontweight='bold', color='black')
                st.pyplot(fig)
            
            # Show explanation if selected
            if show_explanation:
                st.markdown("### Explanation")
                if result['prediction'] == 1:  # Bullying
                    st.markdown("""
                    The model has identified language patterns that are commonly associated with cyberbullying:
                    - Potentially offensive or derogatory terms
                    - Aggressive tone or threatening language
                    - Personal attacks or insults
                    """)
                else:  # Non-bullying
                    st.markdown("""
                    The text appears to use neutral or positive language patterns:
                    - No significant aggressive or offensive terms detected
                    - Communication appears to be constructive or neutral
                    - No identified patterns of personal attacks
                    """)

# Batch analysis section
st.markdown("---")
st.markdown("## 📁 Batch Analysis")

uploaded_file = st.file_uploader("Upload a CSV or TXT file for batch analysis", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Check for a column named 'text' or let user select
            if 'text' in df.columns:
                text_column = 'text'
            else:
                text_column = st.selectbox("Select the column containing text:", df.columns)
        else:  # txt file
            content = uploaded_file.read().decode('utf-8')
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            df = pd.DataFrame({'text': texts})
            text_column = 'text'
            
        if st.button("Run Batch Analysis"):
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze each text entry
            for i, text in enumerate(df[text_column]):
                status_text.text(f"Processing item {i+1} of {len(df)}...")
                progress_bar.progress((i + 1) / len(df))
                
                try:
                    result = st.session_state.predictor.predict(text)
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'prediction': result['prediction_text'],
                        'confidence': result['confidence']
                    })
                except Exception as e:
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'prediction': 'Error',
                        'confidence': 0,
                        'error': str(e)
                    })
            
            # Display results
            results_df = pd.DataFrame(results)
            st.markdown("### Batch Analysis Results")
            st.dataframe(results_df)
            
            # Summary statistics
            st.markdown("### Summary Statistics")
            bullying_count = sum(1 for result in results if result['prediction'] == 'Bullying')
            non_bullying_count = sum(1 for result in results if result['prediction'] == 'Non-bullying')
            error_count = sum(1 for result in results if result['prediction'] == 'Error')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bullying", bullying_count, f"{bullying_count/len(results):.1%}")
            with col2:
                st.metric("Non-bullying", non_bullying_count, f"{non_bullying_count/len(results):.1%}")
            with col3:
                if error_count > 0:
                    st.metric("Errors", error_count, f"{error_count/len(results):.1%}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            counts = [bullying_count, non_bullying_count]
            bars = ax.bar(['Bullying', 'Non-bullying'], counts, color=['#f8d7da', '#d4edda'])
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Predictions')
            for i, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3, counts[i],
                        ha='center', va='bottom')
            st.pyplot(fig)
            
            # Option to download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="cyberbullying_analysis_results.csv",
                mime="text/csv",
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This cyberbullying detection system uses machine learning to identify potentially harmful content.
The system should be used as a tool to assist human judgment, not replace it.

**Note:** The model makes predictions based on patterns learned from training data and may not be accurate in all cases.
""")

# CSS file
css_content = """
body {
    font-family: 'Roboto', sans-serif;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    border-radius: 4px;
}

.stButton>button:hover {
    background-color: #45a049;
}

.css-1aumxhk {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 10px;
}

.error {
    background-color: #f8d7da;
    padding: 15px;
    border-radius: 5px;
    font-weight: bold;
}

.success {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 5px;
    font-weight: bold;
}
"""

# Save CSS file
css_file_path = os.path.join(os.path.dirname(__file__), 'style.css')
if not os.path.exists(css_file_path):
    with open(css_file_path, 'w') as f:
        f.write(css_content)
