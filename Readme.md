# 🎓 Student Attention Predictor (Industry-Level ML Project)

An end-to-end Machine Learning project that predicts student attention levels using academic and behavioral features. This project includes model training, evaluation, explainability, and deployment using Streamlit.

---

## 🚀 Features

- 🔮 Real-time Prediction using user input
- 📊 Model Comparison (Logistic Regression, Random Forest, XGBoost)
- 🏆 Best Model Selection based on accuracy
- 📈 Evaluation Metrics:
  - Accuracy
  - Confusion Matrix
  - ROC Curve (Multiclass)
- 📊 Data Visualization:
  - Score distribution
  - Boxplots
  - Correlation heatmap
- 🔍 Explainable AI:
  - SHAP (model interpretability)
  - Feature Importance
- 🌐 Streamlit Web App UI

---

## 🧠 Machine Learning Workflow

1. Data Preprocessing  
2. Feature Engineering  
3. Model Training  
4. Model Comparison  
5. Evaluation (ROC, Confusion Matrix)  
6. Model Selection  
7. Explainability (SHAP, Feature Importance)  
8. Deployment using Streamlit  

---

## 📂 Project Structure
student-performance-app/
│
├── app.py
├── data/
│ └── StudentsPerformance.csv
│
├── model/
│ ├── train.py
│ ├── predict.py
│ ├── explain.py
│ ├── model.pkl
│ ├── encoder.pkl
│ ├── metrics.txt
│ ├── model_results.txt
│ ├── best_model.txt
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│
├── utils/
│ └── preprocess.py
│
├── requirements.txt
└── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/student-attention-predictor.git
cd student-attention-predictor
pip install -r requirements.txt
▶️ Run the Project
1️⃣ Train Model
python model/train.py
2️⃣ Run Streamlit App
streamlit run app.py
📊 Model Performance
Logistic Regression: ~97%
Random Forest: ~96%
XGBoost: ~96%

✅ Best Model selected LogisticRegression

📸 Screenshots
🔮 Prediction
Real-time student attention prediction
📊 Model Evaluation
Confusion Matrix
ROC Curve
📈 Visualization
Score vs Attention analysis
🧠 Explainability
SHAP used to explain model predictions
Feature Importance graph shows key influencing features
💡 Key Learnings
End-to-end ML pipeline development
Model comparison and evaluation
Handling categorical + numerical data
Explainable AI (SHAP)
Deployment using Streamlit
⚠️ Note
High accuracy due to relatively small dataset
In real-world scenarios, performance may vary
🚀 Future Improvements
Deploy on cloud (Streamlit Cloud / AWS)
Add database integration
Improve UI/UX
Add more real-world features
👨‍💻 Author

Sumit Kumar Karn

🎓 BCA (Hons. with Research)
💡 Interested in AI, ML & Data Science
