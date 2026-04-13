🎓 Student Attention Predictor
An end-to-end Machine Learning project that predicts student attention levels using multiple models, explainable AI (SHAP), and interactive data visualization — deployed with Streamlit.

🚀 Overview
This project builds a complete ML pipeline from data preprocessing to deployment, enabling real-time prediction of student attention levels. It also integrates Explainable AI (XAI) techniques to interpret model decisions.

✨ Key Features
🔮 Real-Time Prediction via user input (Streamlit UI)
🤖 Multiple ML Models
Logistic Regression
Random Forest
XGBoost

🏆 Automatic Best Model Selection
📊 Model Evaluation Metrics
Accuracy Score
Confusion Matrix
ROC Curve (Multiclass)
📈 Data Visualization
Score Distribution
Boxplots
Correlation Heatmap

🧠 Explainable AI
SHAP values for interpretability
Feature importance analysis
🌐 Interactive Web App (Streamlit)
🧠 Machine Learning Workflow
Plain text
1. Data Preprocessing
2. Feature Engineering
3. Model Training
4. Model Comparison
5. Evaluation (Accuracy, ROC, Confusion Matrix)
6. Best Model Selection
7. Explainability (SHAP)
8. Deployment (Streamlit)
📂 Project Structure
Bash
student-attention-predictor/
│
├── app.py                         # Streamlit app
├── data/
│   └── StudentsPerformance.csv   # Dataset
│
├── model/
│   ├── train.py                  # Model training
│   ├── predict.py                # Prediction logic
│   ├── explain.py                # SHAP explainability
│   ├── model.pkl                 # Saved model
│   ├── encoder.pkl               # Encoders
│   ├── metrics.txt               # Evaluation metrics
│   ├── model_results.txt         # Model comparison
│   ├── best_model.txt            # Selected best model
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│
├── utils/
│   └── preprocess.py             # Data preprocessing
│
├── requirements.txt
└── README.md

⚙️ Installation & Setup
1️⃣ Clone Repository
Bash
git clone https://github.com/your-username/student-attention-predictor.git
cd student-attention-predictor
2️⃣ Install Dependencies
Bash
pip install -r requirements.txt
▶️ Run the Project
🔹 Train the Model
Bash
python model/train.py
🔹 Launch Streamlit App
Bash
streamlit run app.py

📊 Model Performance
Model
Accuracy
Logistic Regression
~97% ✅
Random Forest
~96%
XGBoost
~96%
🏆 Best Model: Logistic Regression

📸 Application Highlights
🔮 Prediction
Real-time student attention prediction based on user input

📊 Model Evaluation
Confusion Matrix visualization
ROC Curve for multiclass classification

📈 Data Analysis
Score vs Attention insights
Distribution & correlation analysis

🧠 Explainable AI
SHAP visualization for feature impact
Feature importance ranking

💡 Key Learnings
End-to-end ML pipeline development
Model comparison & performance evaluation
Handling categorical + numerical data
Explainable AI (SHAP)
Deployment using Streamlit

⚠️ Limitations
High accuracy due to small dataset
Performance may vary in real-world scenarios

🚀 Future Improvements
🗄️ Add database integration
🎨 Improve UI/UX design
📊 Use larger & real-world datasets
⚡ Optimize model performance

👨‍💻 Author
Sumit Kumar Karn
🎓 BCA (Hons. with Research)
💡 Passionate about AI, ML & Data Science
📧 Email: sumitkarn2005@gmail.com
🔗 GitHub: https://github.com/Sumit006-coder-dotcom⁠�
🔗 LinkedIn: https://www.linkedin.com/in/sumit-karn-86606524a/⁠�
