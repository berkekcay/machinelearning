
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

# Streamlit UI
st.title("Machine Learning Task with Hyperparameter Tuning")

# Dataset Loading and Preprocessing
st.header("1. Load and Preprocess Dataset")

@st.cache
def load_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    return df

df = load_data()
st.write("Dataset Overview:")
st.dataframe(df.head())

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

st.write(f"Training samples: {X_train.shape[0]}")
st.write(f"Testing samples: {X_test.shape[0]}")

# Models and Hyperparameter Tuning
st.header("2. Train Models with Hyperparameter Tuning")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

params = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "Decision Tree": {"max_depth": [3, 5, 10], "min_samples_split": [2, 5, 10]},
    "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
}

# Hyperparameter tuning and evaluation
results = {}
st.write("Training Models...")

for name, model in models.items():
    st.write(f"### {name}")
    grid = GridSearchCV(model, params[name], cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    results[name] = {
        "Best Params": grid.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }

    st.write("Best Parameters:", grid.best_params_)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Results Summary
st.header("3. Model Performance Summary")
results_df = pd.DataFrame(results).T
st.dataframe(results_df)

# Visualization
st.header("4. Performance Visualization")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
results_df["Accuracy"].plot(kind="bar", ax=ax, color="skyblue")
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
st.pyplot(fig)

st.write("### Thank you for using the application!")
