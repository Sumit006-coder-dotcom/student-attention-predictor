import pickle
import pandas as pd

model = pickle.load(open("model/model.pkl", "rb"))
le = pickle.load(open("model/encoder.pkl", "rb"))

def predict_attention(data):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return le.inverse_transform([pred])[0]