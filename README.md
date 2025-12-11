# Duplicate Question Finder

This repository is a ready-to-deploy Streamlit app that predicts whether two questions are duplicates
using engineered features and a trained model.

## Contents
- `app.py` - Streamlit app
- `helper.py` - helper functions (Option B: smart helper that auto-handles stopwords)
- `setup.sh` - build script that creates Streamlit config and prepares `stopwords.pkl`
- `requirements.txt` - pinned package versions
- `render.yaml` - Render deployment configuration (optional)
- `model.pkl`, `cv.pkl` - your trained model and CountVectorizer (add these to repo or use Git LFS)

## Local testing
1. Create and activate a virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare NLTK stopwords (setup.sh does this; run manually locally if needed):
   ```bash
   bash setup.sh
   ```
4. Place `model.pkl` and `cv.pkl` in project root.
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploy to Render
1. Push repository to GitHub (use Git LFS for large files >100MB).
2. Create a new Web Service in Render and connect to the repo.
3. Render will run `buildCommand` and then `startCommand` from `render.yaml`.

