#!/usr/bin/env bash

# Create Streamlit config
mkdir -p ~/.streamlit/

echo "    [server]\n    port = $PORT\n    enableCORS = false\n    headless = true\n    " > ~/.streamlit/config.toml

# Prepare NLTK stopwords and save as stopwords.pkl
echo "Preparing NLTK stopwords..."
python3 - << 'PY'
import nltk, pickle
from nltk.corpus import stopwords
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
sw = set(stopwords.words('english'))
with open('stopwords.pkl', 'wb') as f:
    pickle.dump(sw, f)
print("stopwords.pkl created successfully!")
PY
