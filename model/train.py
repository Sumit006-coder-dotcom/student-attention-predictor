import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from utils.preprocess import create_target

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")
df = create_target(df)

# Features
X = df.drop(columns=["Attention", "avg_score"])
y = df["Attention"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Columns
cat_cols = ['gender', 'race/ethnicity', 'parental level of education',
            'lunch', 'test preparation course']
num_cols = ['math score', 'reading score', 'writing score']

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])



# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(eval_metric='mlogloss')
}

results = {}

for name, clf in models.items():
    
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", clf)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict (IMPORTANT: use pipeline, not clf)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"{name}: {acc:.3f}")

# Save results
with open("model/model_results.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k}: {v:.3f}\n")

# Best model
best_model_name = max(results, key=results.get)

best_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", models[best_model_name])
])

# ✅ VERY IMPORTANT
best_pipeline.fit(X_train, y_train)

# Now safe to predict
y_pred = best_pipeline.predict(X_test)

# Save
pickle.dump(best_pipeline, open("model/model.pkl", "wb"))
pickle.dump(le, open("model/encoder.pkl", "wb"))

# ================= Accuracy =================
# ================= Accuracy =================
y_pred = best_pipeline.predict(X_test)   # ✅ FIX

acc = accuracy_score(y_test, y_pred)

with open("model/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.2f}")

# ================= Confusion Matrix =================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("model/confusion_matrix.png")

# ================= ROC Curve (Multiclass) =================
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = best_pipeline.predict_proba(X_test)   

fpr, tpr, roc_auc = {}, {}, {}

for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in roc_auc:
    plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC={roc_auc[i]:.2f}")

plt.legend()
plt.title("ROC Curve (Multiclass)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("model/roc_curve.png")

print("🔥 Training Complete + Metrics Saved")