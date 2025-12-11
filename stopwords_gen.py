# stopwords_gen.py
import nltk, pickle
from nltk.corpus import stopwords

# ensure corpus
nltk.download('stopwords', quiet=True)

sw = set(stopwords.words('english'))
with open('stopwords.pkl', 'wb') as f:
    pickle.dump(sw, f)

print("stopwords.pkl created")
