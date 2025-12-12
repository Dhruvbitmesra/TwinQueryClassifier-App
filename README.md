<p align="center">
  <img src="assets/Logo.png" width="120" style="border-radius:18px;margin-bottom:10px">
</p>

<h1 align="center">TwinQueryClassifier</h1>

<p align="center">
  <strong>🔍 Smart Duplicate Question Detector</strong><br>
  Machine-learning powered semantic similarity detection — fast, clean, and beautifully designed.
</p>

---

## 🎥 Demo
<p align="center">
  <img src="assets/demo.gif" width="850" style="border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,0.35)">
</p>

---

## ✨ Overview
TwinQueryClassifier predicts whether two questions are duplicates using a blend of:

- NLP preprocessing  
- Token & semantic similarity  
- Fuzzy string matching  
- Length-based ratios  
- Bag-of-Words vectors  
- A trained ML classifier  

Designed with **Streamlit**, offering a smooth, modern, and interactive UI.

---

## 🧠 Key Features

### 🔹 NLP Feature Engineering  
- Stopword ratios  
- Common token intersection  
- Longest Common Substring  
- FuzzyWuzzy similarity metrics  
- CountVectorizer BoW vectors  

### 🔹 ML Model  
- XGBoost / RandomForest  
- Probability output  
- Adjustable threshold slider  

### 🔹 UI Enhancements  
- Clean, minimal layout  
- Sidebar tips  
- Example question presets  
- Exportable history (CSV)  

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

## 🧬 Model Details

### **Feature Set**
- Common tokens  
- Stopword ratios  
- Multiple fuzzy similarity scores  
- Length & ratio differences  
- LCS (Longest Common Substring)  
- Bag-of-Words vectors  

### **Classifier**
- XGBoost / RandomForest  
- Probability output  
- Adjustable threshold  


---
🔮 Roadmap & Future Enhancements

🌟 SBERT / MiniLM Transformer embeddings

📊 SHAP / LIME explainability

⚡ REST API with FastAPI

🎨 Lottie animations for UI polish

📈 Model comparison dashboard


---
❤️ Author

 Dhruv (BIT Mesra)
---
📬 Connect

GitHub: https://github.com/Dhruvbitmesra

LinkedIn: https://www.linkedin.com/in/dhruv610/
