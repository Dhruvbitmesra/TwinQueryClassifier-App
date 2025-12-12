<p align="center">
  <img src="assets/logo-small.png" width="100" height="100" style="border-radius:16px" alt="Logo"/>
</p>

<h1 align="center">TwinQueryClassifier</h1>

<p align="center">
  <strong>🔍 Duplicate Question Finder</strong><br>
  Machine-learning powered detection of semantically similar questions — fast, interpretable, and beautifully designed.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img alt="Stars" src="https://img.shields.io/github/stars/Dhruvbitmesra/TwinQueryClassifier?style=for-the-badge" />
</p>

---

## 🎥 Demo (GIF)

<p align="center">
  <img src="assets/demo.gif" width="850" alt="TwinQueryClassifier Demo"
       style="border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.45)">
</p>

---

## ✨ Overview

TwinQueryClassifier predicts whether two natural-language questions are **duplicates** using:

- NLP preprocessing  
- Token-based features  
- Fuzzy string similarity  
- Length & semantic ratios  
- Bag-of-words vectors  
- A trained ML classifier  

Built with **Streamlit**, the UI is fast, interactive, and includes confidence scoring, history tracking, and a modern animated layout.

---

## 🧠 Key Features

### 🔹 NLP + Feature Engineering  
- Stopword ratios  
- Common token counts  
- Longest common substring  
- Fuzzywuzzy similarity metrics  
- Vectorized text inputs (CountVectorizer)

### 🔹 ML Model  
- Trained using XGBoost / RandomForest  
- Probability output + threshold slider  
- Automatically handles all text processing

### 🔹 UI Enhancements  
- Clean card layout  
- Tips sidebar  
- Animated interactions  
- Example question presets  
- Downloadable history (CSV)

---
## 📂 Project Structure


TwinQueryClassifier/
│── app.py
│── helper.py
│── model.pkl
│── cv.pkl
│── stopwords.pkl
│── stopwords_gen.py
│── test_load.py
│── train_model.py
│── setup.sh
│── requirements.txt
│── render.yaml
│
└── assets/
├── demo.gif
└── logo-small.png


---

## 🧭 Run Locally

```bash
# Clone repo
git clone https://github.com/Dhruvbitmesra/TwinQueryClassifier.git
cd TwinQueryClassifier

# Create venv
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate stopwords (if missing)
bash setup.sh
# OR:
python stopwords_gen.py

# Run Streamlit app
streamlit run app.py
🧬 Model Details
Features Used

Common tokens

Stopword ratios

Fuzzy string match (QRatio, Partial, Token Sort, Token Set)

Length differences

LCS ratio

BoW vectors

Classifier

XGBoost / RandomForest

Probability output

Threshold-adjustable

🔮 Future Enhancements

Transformer embeddings (MiniLM, Sentence-BERT)

SHAP/LIME explanations

FastAPI backend

Lottie animations

Model comparison dashboard

❤️ Author

 Dhruv (BIT Mesra)


📬 Contact
GitHub: https://github.com/Dhruvbitmesra

LinkedIn: https://www.linkedin.com/in/dhruv610/
## 📂 Project Structure

