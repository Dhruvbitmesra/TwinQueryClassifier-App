<h1 align="center">TwinQueryClassifier</h1>
<p align="center">
  <strong>Duplicate Question Finder</strong> — an ML-powered system that identifies whether two natural-language questions express the same meaning.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge" />
  <img alt="Streamlit" src="https://img.shields.io/badge/streamlit-app-red?style=for-the-badge" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />
  <img alt="Stars" src="https://img.shields.io/github/stars/Dhruvbitmesra/TwinQueryClassifier?style=for-the-badge" />
</p>

---

## 🎥 Demo  
<p align="center">
  <!-- Replace this with your real demo.gif -->
  <img src="assets/demo.gif" alt="TwinQueryClassifier Demo" width="850" style="border-radius:12px; box-shadow: 0 12px 40px rgba(2,6,23,0.4)" />
</p>

---

## ✨ Overview  
**TwinQueryClassifier** detects duplicate questions using a hybrid NLP approach combining:

- Custom preprocessing  
- Stopword-aware token analysis  
- Fuzzy string similarity metrics  
- Bag-of-Words embeddings  
- Feature engineering  
- ML classification (RandomForest / XGBoost)  
- A clean, modern Streamlit UI  

This project is built to be:  
✨ **Deployable** • ⚡ **Fast** • 🎯 **Accurate** • 🎨 **Aesthetic**

---

## 📁 Project Structure  

```
TwinQueryClassifier/
├─ app.py
├─ helper.py
├─ model.pkl
├─ cv.pkl
├─ stopwords.pkl
├─ stopwords_gen.py
├─ setup.sh
├─ requirements.txt
├─ render.yaml
└─ assets/
   └─ demo.gif   <-- add your GIF here
```

---

## 🧭 Run Locally  

```bash
git clone https://github.com/Dhruvbitmesra/TwinQueryClassifier.git
cd TwinQueryClassifier

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

# Generate stopwords.pkl (optional)
bash setup.sh  
# or
python stopwords_gen.py

streamlit run app.py
```

Now open:  
👉 http://localhost:8501

---

## ☁ Deploy on Render  

Render will automatically detect `render.yaml`.

### Example config:
```yaml
services:
  - type: web
    name: twinqueryclassifier
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.headless true
    envVars:
      - key: PYTHON_VERSION
        value: 3.10
```

### If your model file is large (>50MB):

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes model.pkl cv.pkl
git commit -m "Add large model files"
git push
```

---

## 🧠 Model Details  

### Features used  
- Token overlap ratios  
- Stopword-based ratios  
- Length differences  
- Longest common substring  
- FuzzyWuzzy metrics  
- Bag-of-Words vectors for each question (3k features × 2)  
- 15+ engineered features  

### Algorithms  
- RandomForestClassifier  
- XGBoost Classifier (optional, more accurate)  

---

## 🚀 Future Enhancements  
- Transformer embeddings (SBERT / BERT)  
- LIME/SHAP explainability  
- FastAPI backend + Streamlit frontend split  
- Animated UI with Lottie  
- Dataset exploration dashboard  

---

## ❤️ Author  
Built with passion by **Dhruv (BIT Mesra)**.  

---

## 📬 Contact  
GitHub: https://github.com/Dhruvbitmesra  

