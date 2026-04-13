import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import shap

from model.predict import predict_attention
from model.explain import get_shap_values

# ================= UI =================
st.set_page_config(page_title="Student Attention Predictor", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:28px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🎓 Student Attention Predictor Dashboard</p>', unsafe_allow_html=True)

# ================= INPUT =================
st.sidebar.header("📝 Enter Student Details")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A","group B","group C","group D","group E"])
education = st.sidebar.selectbox("Parental Education",
    ["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"])
lunch = st.sidebar.selectbox("Lunch", ["standard","free/reduced"])
prep = st.sidebar.selectbox("Test Preparation", ["none","completed"])

math = st.sidebar.slider("Math Score", 0, 100, 70)
reading = st.sidebar.slider("Reading Score", 0, 100, 70)
writing = st.sidebar.slider("Writing Score", 0, 100, 70)

input_data = {
    "gender": gender,
    "race/ethnicity": race,
    "parental level of education": education,
    "lunch": lunch,
    "test preparation course": prep,
    "math score": math,
    "reading score": reading,
    "writing score": writing
}

# ================= PREDICTION =================
st.subheader("🧠 Prediction")

try:
    result = predict_attention(input_data)
    st.success(f"🎯 Predicted Attention Level: {result}")
except Exception as e:
    st.error(f"Error: {e}")

# ================= MODEL COMPARISON =================
st.subheader("📊 Model Comparison")

if os.path.exists("model/model_results.txt"):
    with open("model/model_results.txt", "r") as f:
        st.text(f.read())

# ================= BEST MODEL =================
st.subheader("🏆 Best Model")

if os.path.exists("model/best_model.txt"):
    with open("model/best_model.txt", "r") as f:
        st.success(f"Best Model: {f.read()}")

# ================= ACCURACY =================
st.subheader("📈 Best Model Accuracy")

if os.path.exists("model/metrics.txt"):
    with open("model/metrics.txt", "r") as f:
        st.success(f.read())

# ================= CONFUSION MATRIX =================
st.subheader("📊 Confusion Matrix")

if os.path.exists("model/confusion_matrix.png"):
    st.image("model/confusion_matrix.png")

# ================= ROC =================
st.subheader("📈 ROC Curve")

if os.path.exists("model/roc_curve.png"):
    st.image("model/roc_curve.png")

# ================= SHAP =================
st.subheader("🔍 Explain Prediction (SHAP)")

try:
    model = pickle.load(open("model/model.pkl", "rb"))
    df_input = pd.DataFrame([input_data])

    transformed = model.named_steps['preprocessing'].transform(df_input)
    clf = model.named_steps['classifier']

    if "RandomForest" in str(type(clf)):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(transformed)
        shap.summary_plot(shap_values, transformed, show=False)

    else:
        explainer = shap.LinearExplainer(clf, transformed)
        shap_values = explainer.shap_values(transformed)
        shap.summary_plot(shap_values, transformed, show=False)

    st.pyplot(plt.gcf())

except Exception as e:
    st.warning(f"SHAP not available: {e}")
# ================= FEATURE IMPORTANCE =================
st.subheader("📊 Feature Importance")

try:
    model = pickle.load(open("model/model.pkl", "rb"))
    clf = model.named_steps['classifier']
    feature_names = model.named_steps['preprocessing'].get_feature_names_out()

    fig, ax = plt.subplots(figsize=(8,5))

    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    else:
        importance = abs(clf.coef_[0])

    sns.barplot(x=importance, y=feature_names, ax=ax)

    st.pyplot(fig)

except Exception as e:
    st.warning(f"Feature importance not available: {e}")

# ================= DATA VIS =================
st.subheader("📊 Dataset Visualization")

df = pd.read_csv("data/StudentsPerformance.csv")

# Create Attention
df['avg_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3

def get_attention(score):
    if score > 80:
        return "High"
    elif score > 50:
        return "Medium"
    else:
        return "Low"

df['Attention'] = df['avg_score'].apply(get_attention)

feature = st.selectbox("Select Feature", [
    'math score',
    'reading score',
    'writing score'
])

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x="Attention", y=feature, data=df, hue="Attention", legend=False, ax=ax)

plt.xlabel("Attention Level")
plt.ylabel(feature)
plt.title(f"{feature} vs Attention Level")

st.pyplot(fig)

# ================= HEATMAP =================
st.subheader("📍 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(8,6))
sns.heatmap(df[['math score','reading score','writing score']].corr(),
            annot=True, cmap="coolwarm", ax=ax2)

st.pyplot(fig2)