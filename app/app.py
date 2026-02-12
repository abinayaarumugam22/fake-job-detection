from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ============================================================================
# LOAD MODEL AND TOKENIZER AT STARTUP
# ============================================================================
print("=" * 80)
print("LOADING FAKE JOB DETECTOR MODEL...")
print("=" * 80)

# Get absolute project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_bilstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.pkl")

print("Model path:", MODEL_PATH)
print("Tokenizer path:", TOKENIZER_PATH)
print("Config path:", CONFIG_PATH)

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("✅ Tokenizer loaded")

# Load config
with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)
print("✅ Configuration loaded")

MAX_LEN = config['max_len']
print(f"✅ Configuration loaded (MAX_LEN: {MAX_LEN})")

print("=" * 80)
print("✅ FAKE JOB DETECTOR API READY!")
print("=" * 80)

# ============================================================================
# TEXT PREPROCESSING FUNCTION
# ============================================================================
def clean_text(text):
    """
    Clean and preprocess job description text
    Same preprocessing as training data
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove very short words
    text = ' '.join([word for word in text.split() if len(word) > 1])
    
    return text

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_job(job_text):
    """
    Predict if a job posting is fake or real
    
    Args:
        job_text (str): Job description text
        
    Returns:
        dict: Prediction results with probability and label
    """
    # Clean the text
    cleaned = clean_text(job_text)
    
    # Check if text is too short after cleaning
    if len(cleaned) < 50:
        return {
            'error': True,
            'message': 'Job description too short. Please provide more details.',
            'min_length_required': 50
        }
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned])
    
    # Pad sequence
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Make prediction
    prediction_prob = float(model.predict(padded, verbose=0)[0][0])
    
    # Determine label
    is_fake = prediction_prob > 0.5
    label = 'FAKE' if is_fake else 'REAL'
    confidence = prediction_prob if is_fake else (1 - prediction_prob)
    
    # Risk level
    if prediction_prob >= 0.75:
        risk_level = 'HIGH'
        risk_color = '#e74c3c'
    elif prediction_prob >= 0.5:
        risk_level = 'MEDIUM'
        risk_color = '#f39c12'
    elif prediction_prob >= 0.25:
        risk_level = 'LOW'
        risk_color = '#f1c40f'
    else:
        risk_level = 'VERY LOW'
        risk_color = '#2ecc71'
    
    # Warning message
    if is_fake:
        if prediction_prob >= 0.75:
            warning = '🚨 HIGH RISK! This job posting shows strong signs of being fraudulent. Avoid applying!'
        else:
            warning = '⚠️ CAUTION! This job posting may be fraudulent. Verify before applying.'
    else:
        warning = '✅ This job posting appears legitimate. However, always verify company details before applying.'
    
    return {
        'error': False,
        'prediction': label,
        'is_fake': is_fake,
        'probability': round(prediction_prob, 4),
        'confidence': round(confidence * 100, 2),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'warning': warning,
        'text_length': len(cleaned)
    }

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Render the web interface"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for job fraud prediction
    
    Expected JSON:
    {
        "job_description": "text of job posting..."
    }
    
    Returns:
    {
        "prediction": "FAKE" or "REAL",
        "probability": 0.XX,
        "confidence": XX%,
        "warning": "message"
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'job_description' not in data:
            return jsonify({
                'error': True,
                'message': 'Missing job_description in request body'
            }), 400
        
        job_text = data['job_description']
        
        # Validate input
        if not job_text or len(job_text.strip()) == 0:
            return jsonify({
                'error': True,
                'message': 'Job description cannot be empty'
            }), 400
        
        # Make prediction
        result = predict_job(job_text)
        
        if result.get('error'):
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    }), 200

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
