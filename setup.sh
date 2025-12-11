#!/usr/bin/env bash
set -euo pipefail

# Create Streamlit config
mkdir -p ~/.streamlit

cat > ~/.streamlit/config.toml <<'CFG'
[server]
port = $PORT
enableCORS = false
headless = true
CFG

# Prepare NLTK stopwords and save as stopwords.pkl
echo "Preparing NLTK stopwords..."

# Prefer python3, fallback to python
PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=python
else
  echo "ERROR: python not found"
  exit 1
fi

$PYTHON_CMD - <<'PY'
import nltk, pickle, sys
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    sw = set(stopwords.words('english'))
    with open('stopwords.pkl', 'wb') as f:
        pickle.dump(sw, f)
    print("stopwords.pkl created successfully!")
except Exception as e:
    print("Failed to create stopwords.pkl:", e)
    sys.exit(1)
PY

echo "setup.sh finished."
