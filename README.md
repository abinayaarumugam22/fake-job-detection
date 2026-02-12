# 🔍 Fake Job Posting Detection using Deep Learning

An AI-powered web application that detects fraudulent job postings using Bidirectional LSTM (BiLSTM) deep learning model.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![Accuracy](https://img.shields.io/badge/Accuracy-96.98%25-success)
![Recall](https://img.shields.io/badge/Recall-87.79%25-success)

## 🎯 Project Overview

This project helps job seekers identify fraudulent job postings using Natural Language Processing and Deep Learning. The model analyzes job descriptions and predicts whether they are legitimate or fake with **87.79% recall** and **96.98% accuracy**.

### Key Features
- ✅ **High Detection Rate**: Catches 87.8% of fake job postings
- ✅ **Real-time Predictions**: Instant analysis through web interface
- ✅ **Deep Learning Model**: BiLSTM architecture with 1.4M parameters
- ✅ **User-Friendly Interface**: Clean, responsive web design
- ✅ **REST API**: Easy integration with other applications

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 96.98% |
| Recall (Fake Detection) | 87.79% |
| Precision | 63.45% |
| F1-Score | 0.7366 |
| ROC-AUC | 0.9875 |

**Confusion Matrix:**
- True Positives: 151 (Fake jobs caught)
- False Negatives: 21 (Fake jobs missed)
- True Negatives: 3,316 (Real jobs identified)
- False Positives: 87 (Real jobs flagged)

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.15, Keras
- **NLP**: NLTK, Tokenization, Word Embeddings
- **Backend**: Flask, Flask-CORS
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure

```
fake_job_detection/
├── data/                          # Dataset files
│   ├── fake_job_postings.csv     # Original dataset
│   └── preprocessed_data.csv     # Cleaned dataset
├── models/                        # Trained models
│   ├── best_bilstm_model.h5      # Best model (highest recall)
│   ├── tokenizer.pkl             # Text tokenizer
│   └── config.pkl                # Model configuration
├── output/                        # Visualizations & results
│   ├── data_exploration.png      # EDA plots
│   ├── training_history.png      # Training curves
│   └── model_evaluation.png      # Evaluation metrics
├── app/                           # Web application
│   ├── app.py                    # Flask backend
│   └── templates/
│       └── index.html            # Web interface
├── preprocess.py                 # Data preprocessing
├── tokenize.py                   # Text tokenization
├── build_model.py                # Model training
├── evaluate_model.py             # Model evaluation
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 - 3.11
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/abinayaarumugam22/fake-job-detection.git
cd fake-job-detection
```

2. **Create virtual environment**
```bash
python -m venv fake_job_env
fake_job_env\Scripts\activate  # Windows
source fake_job_env/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Download from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Place `fake_job_postings.csv` in `data/` folder

5. **Train model (Optional - pre-trained model included)**
```bash
python preprocess.py
python tokenize.py
python build_model.py
```

6. **Run web application**
```bash
cd app
python app.py
```

7. **Open browser**
```
http://127.0.0.1:5000
```

## 🧠 Model Architecture

```
Input (200 tokens)
    ↓
Embedding Layer (10,000 vocab → 128 dimensions)
    ↓
BiLSTM Layer 1 (64 units)
    ↓
BiLSTM Layer 2 (32 units)
    ↓
Dropout (50%)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (30%)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Total Parameters:** 1,424,257

## 📈 Training Details

- **Dataset Size**: 17,875 job postings (95.2% real, 4.8% fake)
- **Train/Test Split**: 80/20
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Binary Crossentropy
- **Class Imbalance Handling**: Class weights (19.8x for minority class)

## 🌐 API Documentation

### Predict Endpoint

**POST** `/api/predict`

**Request Body:**
```json
{
    "job_description": "Software Engineer position at leading tech company..."
}
```

**Response:**
```json
{
    "prediction": "REAL",
    "is_fake": false,
    "probability": 0.2345,
    "confidence": 76.55,
    "risk_level": "LOW",
    "warning": "✅ This job posting appears legitimate..."
}
```

### Health Check

**GET** `/api/health`

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "tokenizer_loaded": true
}
```

## 📊 Dataset

**Source**: [Kaggle - Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

**Features**:
- Job title
- Location
- Department
- Salary range
- Company profile
- Description
- Requirements
- Benefits
- Employment type
- Required experience
- Required education
- Industry
- Function
- Fraudulent (target variable)

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end deep learning pipeline
- ✅ NLP text preprocessing and tokenization
- ✅ Handling imbalanced datasets
- ✅ BiLSTM architecture for sequence modeling
- ✅ Model evaluation and interpretation
- ✅ Flask API development
- ✅ Web application deployment
- ✅ Git version control

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/abinayaarumugam22)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/abinaya-arumugam-187ab325b/)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Inspired by the need to protect job seekers from fraud
- Built as a portfolio project to demonstrate ML/DL skills

## ⚠️ Disclaimer

This tool provides AI-based predictions and should not be the sole factor in determining job legitimacy. Always verify job postings through official company channels and trusted sources.

---

**If you found this project helpful, please consider giving it a ⭐!**
