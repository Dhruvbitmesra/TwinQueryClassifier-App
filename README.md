# TwinQueryClassifier

Duplicate Question Finder using machine learning. Detects semantically similar questions with NLP features, fuzzy matching, and a trained classifier. Clean Streamlit UI included.

## Demo
![Demo](assets/demo.gif)

## Overview
TwinQueryClassifier predicts whether two questions are duplicates using:
- NLP preprocessing
- Token-based features
- Fuzzy string similarity
- Length & semantic ratios
- Bag-of-words vectors
- A trained ML classifier

## Key Features
NLP + Feature Engineering:
- Stopword ratios
- Common token counts
- Longest common substring
- FuzzyWuzzy similarity metrics
- CountVectorizer BoW vectors

ML Model:
- XGBoost / RandomForest
- Probability output
- Threshold slider

UI Enhancements:
- Clean layout
- Tips sidebar
- Example presets
- Downloadable history

## Project Structure
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



Features:

Common tokens

Stopword ratios

QRatio, PartialRatio, TokenSort, TokenSet

Length differences

LCS ratio

BoW vectors

Classifier:

XGBoost / RandomForest

Probability output

Adjustable threshold

Future Enhancements

Transformer embeddings (MiniLM, SBERT)

SHAP/LIME explanations

FastAPI backend

Lottie animations

Model comparison dashboard

Author

Built by Dhruv (BIT Mesra)

Contact

GitHub: https://github.com/Dhruvbitmesra

LinkedIn: https://www.linkedin.com/in/dhruv610/

## Run Locally
```bash
git clone https://github.com/Dhruvbitmesra/TwinQueryClassifier.git
cd TwinQueryClassifier

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

bash setup.sh
# OR:
python stopwords_gen.py


Model Details






streamlit run app.py
