import pickle
m = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))
print("Loaded model:", type(m), "Loaded cv:", type(cv))
