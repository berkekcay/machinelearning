import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit UI
st.set_page_config(page_title="Machine Learning App", layout="wide")
st.title("Machine Learning Task with Enhanced UI and Hyperparameter Tuning")

# Sidebar for navigation
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Select a Section", ["Dataset Overview", "Model Training", "Performance Summary"])

# Dataset Loading and Preprocessing
@st.cache_data
def load_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    return df

df = load_data()

if menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("### Raw Dataset")
    st.dataframe(df.head())

    # Show dataset summary
    st.write("### Dataset Summary")
    st.write(df.describe())

    # Correlation heatmap
    st.write("### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(df.corr(), cmap="coolwarm")
    fig.colorbar(cax)
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.columns)), df.columns)
    st.pyplot(fig)

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

if menu == "Model Training":
    st.header("Model Training with Hyperparameter Tuning")

    # Models and Hyperparameter Tuning
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

    # Sidebar for model selection
    selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
    st.write(f"### Training {selected_model}")

    # Hyperparameter tuning
    grid = GridSearchCV(models[selected_model], params[selected_model], cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    st.write("#### Best Parameters")
    st.json(grid.best_params_)

    st.write("#### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Feature Importance for Tree-based models
    if selected_model in ["Decision Tree", "Gradient Boosting"]:
        st.write("#### Feature Importance")
        feature_importance = pd.Series(best_model.feature_importances_, index=df.columns[:-1])
        feature_importance = feature_importance.sort_values(ascending=False)
        fig, ax = plt.subplots()
        feature_importance.plot(kind="bar", ax=ax, color="skyblue")
        st.pyplot(fig)

if menu == "Performance Summary":
    st.header("Performance Summary")

    results = {}
    for name, model in models.items():
        grid = GridSearchCV(model, params[name], cv=5, scoring="accuracy")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        results[name] = {
            "Best Params": grid.best_params_,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0)
        }

    results_df = pd.DataFrame(results).T
    st.write("### Model Performance Metrics")
    st.dataframe(results_df)

    # Visualization
    st.write("### Accuracy Comparison")
    fig, ax = plt.subplots()
    results_df["Accuracy"].plot(kind="bar", ax=ax, color="green")
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)
