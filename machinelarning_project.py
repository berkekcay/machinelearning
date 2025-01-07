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

# Define models and hyperparameters globally
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

if menu == "Model Training":
    st.header("Model Training with Hyperparameter Tuning")

    # Sidebar for model selection
    selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
    st.write(f"### Training {selected_model}")

    # Train model without tuning
    base_model = models[selected_model]
    base_model.fit(X_train, y_train)
    base_pred = base_model.predict(X_test)

    base_metrics = {
        "Accuracy": accuracy_score(y_test, base_pred),
        "Precision": precision_score(y_test, base_pred, zero_division=0),
        "Recall": recall_score(y_test, base_pred, zero_division=0),
        "F1-Score": f1_score(y_test, base_pred, zero_division=0)
    }

    st.write("#### Performance Before Hyperparameter Tuning")
    st.json(base_metrics)

    # Hyperparameter tuning
    grid = GridSearchCV(models[selected_model], params[selected_model], cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    tuned_pred = best_model.predict(X_test)

    tuned_metrics = {
        "Accuracy": accuracy_score(y_test, tuned_pred),
        "Precision": precision_score(y_test, tuned_pred, zero_division=0),
        "Recall": recall_score(y_test, tuned_pred, zero_division=0),
        "F1-Score": f1_score(y_test, tuned_pred, zero_division=0)
    }

    st.write("#### Performance After Hyperparameter Tuning")
    st.json(tuned_metrics)

    # Performance Comparison Visualization
    st.write("### Performance Comparison")
    comparison_df = pd.DataFrame({"Before Tuning": base_metrics, "After Tuning": tuned_metrics}).T
    st.dataframe(comparison_df)

    # Bar Chart for Performance Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind="bar", ax=ax)
    ax.set_title("Performance Metrics Comparison")
    ax.set_ylabel("Scores")
    ax.set_xticklabels(comparison_df.index, rotation=0)
    st.pyplot(fig)

    # Classification Report
    st.write("#### Classification Report")
    report = classification_report(y_test, tuned_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

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
        # Performance before tuning
        model.fit(X_train, y_train)
        base_pred = model.predict(X_test)

        base_metrics = {
            "Accuracy": accuracy_score(y_test, base_pred),
            "Precision": precision_score(y_test, base_pred, zero_division=0),
            "Recall": recall_score(y_test, base_pred, zero_division=0),
            "F1-Score": f1_score(y_test, base_pred, zero_division=0)
        }

        # Performance after tuning
        grid = GridSearchCV(model, params[name], cv=5, scoring="accuracy")
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        tuned_pred = best_model.predict(X_test)

        tuned_metrics = {
            "Accuracy": accuracy_score(y_test, tuned_pred),
            "Precision": precision_score(y_test, tuned_pred, zero_division=0),
            "Recall": recall_score(y_test, tuned_pred, zero_division=0),
            "F1-Score": f1_score(y_test, tuned_pred, zero_division=0)
        }

        results[name] = {
            "Before Tuning": base_metrics,
            "After Tuning": tuned_metrics
        }

    st.write("### Model Performance Comparison")
    for model_name, performance in results.items():
        st.write(f"#### {model_name}")
        st.write("Before Tuning:", performance["Before Tuning"])
        st.write("After Tuning:", performance["After Tuning"])

    # Visualization
    st.write("### Accuracy Comparison")
    before_tuning = {name: perf["Before Tuning"]["Accuracy"] for name, perf in results.items()}
    after_tuning = {name: perf["After Tuning"]["Accuracy"] for name, perf in results.items()}

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(before_tuning))
    ax.bar(x - width/2, before_tuning.values(), width, label="Before Tuning")
    ax.bar(x + width/2, after_tuning.values(), width, label="After Tuning")
    ax.set_xticks(x)
    ax.set_xticklabels(before_tuning.keys(), rotation=45)
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    st.write("### Metric Comparison Across Models")
    metrics_comparison = pd.DataFrame({
        "Before Tuning": before_tuning,
        "After Tuning": after_tuning
    })
    st.dataframe(metrics_comparison)

    # Line Chart for Performance Trends
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_comparison.plot(kind="line", marker="o", ax=ax)
    ax.set_title("Performance Trends Across Models")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(len(metrics_comparison.index)))
    ax.set_xticklabels(metrics_comparison.index, rotation=0)
    st.pyplot(fig)
