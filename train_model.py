# train_model.py
# Retrains XGBoost (and RandomForest as fallback) on your engineered features,
# saves model.pkl and cv.pkl compatible with the current environment.

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd

# data & text processing
import re
from bs4 import BeautifulSoup

# feature helpers
from nltk.corpus import stopwords

# vectorizer & ML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# adjust paths
HERE = os.path.dirname(__file__) or "."
DATA_PATH = os.path.join(HERE, "train.csv")   # ensure train.csv is in project folder

# -------------------------
# helper functions (same as app/helper preprocessing)
# -------------------------
def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\\1m', q)
    q = re.sub(r'([0-9]+)000', r'\\1k', q)

    # contractions (shortened here but adequate)
    contractions = {
        "don't": "do not", "i'm": "i am", "you're": "you are",
        "it's": "it is", "that's": "that is", "won't": "will not",
        "can't": "can not", "i've": "i have", "i'll": "i will"
    }
    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have").replace("n't", " not").replace("'re", " are").replace("'ll", " will")

    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q).strip()
    return q

def common_words(q1, q2):
    w1 = set(map(lambda w: w.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda w: w.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def total_words(q1, q2):
    w1 = set(map(lambda w: w.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda w: w.lower().strip(), q2.split(" ")))
    return len(w1) + len(w2)

def token_features(q1, q2, stop_words):
    SAFE_DIV = 1e-4
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return [0.0]*8
    q1_words = set([w for w in q1_tokens if w not in stop_words])
    q2_words = set([w for w in q2_tokens if w not in stop_words])
    q1_stops = set([w for w in q1_tokens if w in stop_words])
    q2_stops = set([w for w in q2_tokens if w in stop_words])
    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    feats = [0.0]*8
    feats[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    feats[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    feats[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    feats[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    feats[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    feats[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    feats[6] = int(q1_tokens[-1] == q2_tokens[-1])
    feats[7] = int(q1_tokens[0] == q2_tokens[0])
    return feats

def length_features(q1, q2):
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return [0.0]*3
    abs_len = abs(len(q1_tokens) - len(q2_tokens))
    mean_len = (len(q1_tokens) + len(q2_tokens)) / 2.0
    # simple longest common substring ratio (naive)
    common = 0
    for i in range(min(len(q1), len(q2))):
        if q1[i] == q2[i]:
            common += 1
    longest_sub = common / (min(len(q1), len(q2)) + 1)
    return [abs_len, mean_len, longest_sub]

# -------------------------
# Load data
# -------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
if 'question1' not in df.columns or 'question2' not in df.columns:
    raise RuntimeError("train.csv must contain question1 and question2 columns")

# sample to speed up (optional)
try:
    new_df = df.sample(30000, random_state=2)
except Exception:
    new_df = df.copy()

# Preprocess
print("Preprocessing text...")
new_df['question1'] = new_df['question1'].fillna("").apply(preprocess)
new_df['question2'] = new_df['question2'].fillna("").apply(preprocess)

# Basic features
print("Building features...")
new_df['q1_len'] = new_df['question1'].str.len()
new_df['q2_len'] = new_df['question2'].str.len()
new_df['q1_num_words'] = new_df['question1'].apply(lambda x: len(x.split()))
new_df['q2_num_words'] = new_df['question2'].apply(lambda x: len(x.split()))
new_df['word_common'] = new_df.apply(lambda r: common_words(r['question1'], r['question2']), axis=1)
new_df['word_total'] = new_df.apply(lambda r: total_words(r['question1'], r['question2']), axis=1)
new_df['word_share'] = round(new_df['word_common'] / (new_df['word_total'] + 1e-9), 2)

# stopwords (from the pickle you generated)
stop_words = set()
stop_pkl = os.path.join(HERE, "stopwords.pkl")
if os.path.exists(stop_pkl):
    stop_words = pickle.load(open(stop_pkl, "rb"))
else:
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        import nltk
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))

token_feats = new_df.apply(lambda r: token_features(r['question1'], r['question2'], stop_words), axis=1)
new_df["cwc_min"] = list(map(lambda x: x[0], token_feats))
new_df["cwc_max"] = list(map(lambda x: x[1], token_feats))
new_df["csc_min"] = list(map(lambda x: x[2], token_feats))
new_df["csc_max"] = list(map(lambda x: x[3], token_feats))
new_df["ctc_min"] = list(map(lambda x: x[4], token_feats))
new_df["ctc_max"] = list(map(lambda x: x[5], token_feats))
new_df["last_word_eq"] = list(map(lambda x: x[6], token_feats))
new_df["first_word_eq"] = list(map(lambda x: x[7], token_feats))

length_feats = new_df.apply(lambda r: length_features(r['question1'], r['question2']), axis=1)
new_df['abs_len_diff'] = list(map(lambda x: x[0], length_feats))
new_df['mean_len'] = list(map(lambda x: x[1], length_feats))
new_df['longest_substr_ratio'] = list(map(lambda x: x[2], length_feats))

# Fuzzy features using fuzzywuzzy if installed
try:
    from fuzzywuzzy import fuzz
    fuzzy_feats = new_df.apply(lambda r: [fuzz.QRatio(r['question1'], r['question2']),
                                          fuzz.partial_ratio(r['question1'], r['question2']),
                                          fuzz.token_sort_ratio(r['question1'], r['question2']),
                                          fuzz.token_set_ratio(r['question1'], r['question2'])], axis=1)
    new_df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_feats))
    new_df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_feats))
    new_df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_feats))
    new_df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_feats))
except Exception:
    new_df['fuzz_ratio'] = 0
    new_df['fuzz_partial_ratio'] = 0
    new_df['token_sort_ratio'] = 0
    new_df['token_set_ratio'] = 0

# Build final dataframe for modeling
final_df = new_df.drop(columns=['id','qid1','qid2','question1','question2'], errors='ignore')
print("Final feature dataframe shape:", final_df.shape)

# Create BOW CountVectorizer on combined questions (same as GUI expects)
questions = list(new_df['question1']) + list(new_df['question2'])
cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)
temp_df1 = pd.DataFrame(q1_arr, index=new_df.index)
temp_df2 = pd.DataFrame(q2_arr, index=new_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
final_df = pd.concat([final_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
print("After adding BOW shape:", final_df.shape)

# Train / test split
X = final_df.iloc[:,1:].values
y = final_df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train RandomForest (fallback)
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("RF accuracy:", acc_rf)

# Train XGBoost (primary)
print("Training XGBoost (this may take a little while)...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGB accuracy:", acc_xgb)

# Choose best model (prefer XGB if better)
best_model = xgb if acc_xgb >= acc_rf else rf
print("Selected model:", type(best_model).__name__)

# Save model and CountVectorizer
model_path = os.path.join(HERE, "model.pkl")
cv_path = os.path.join(HERE, "cv.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
with open(cv_path, "wb") as f:
    pickle.dump(cv, f)

print("Saved model to", model_path)
print("Saved CountVectorizer to", cv_path)
