import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Apply global styles for charts (dark mode aesthetic)
plt.style.use("dark_background")

# Load the dataset
data = pd.read_csv("data_synthetic.csv")

# Streamlit App Configuration
st.title("IoT Risk Insights and Analysis")
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose an application:", 
                              ["EDA", "Risk Assessment", "Predictive Modeling", "Fraud Detection"])

# Data Cleaning
data_cleaned = data.drop_duplicates().dropna()

# Convert categorical columns to numeric
categorical_columns = data_cleaned.select_dtypes(include=["object"]).columns
data_cleaned = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

# EDA Section
if option == "EDA":
    st.header("Exploratory Data Analysis")

    # Display dataset overview
    st.subheader("Dataset Overview")
    st.write(data_cleaned.describe())

    # Distribution of Risk Profiles
    st.subheader("Distribution of Risk Profiles")
    fig, ax = plt.subplots()
    sns.countplot(data=data_cleaned, x="Risk Profile", palette="Reds", ax=ax)
    ax.set_title("Risk Profile Distribution", fontsize=16, weight='bold')
    ax.set_xlabel("Risk Profile", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Analysis")
    numeric_data = data_cleaned.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, weight='bold')
    st.pyplot(fig)

# Risk Assessment Section
elif option == "Risk Assessment":
    st.header("Risk Assessment")

    # Show high-risk customers
    st.subheader("High-Risk Customers")
    high_risk_customers = data_cleaned[data_cleaned["Risk Profile"] == 3]
    st.write(f"Number of high-risk customers: {len(high_risk_customers)}")
    st.write(high_risk_customers.head(10))

    # Claims History and Risk Profile Analysis
    st.subheader("Claims History vs Risk Profile")
    fig, ax = plt.subplots()
    sns.boxplot(x="Risk Profile", y="Previous Claims History", data=high_risk_customers, palette="Reds", ax=ax)
    ax.set_title("Claims History by Risk Profile", fontsize=16, weight='bold')
    st.pyplot(fig)

# Predictive Modeling Section
elif option == "Predictive Modeling":
    st.header("Predictive Modeling")

    # Prepare data for modeling
    features = data_cleaned.drop(columns=["Customer ID", "Risk Profile"])
    target = data_cleaned["Risk Profile"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature Importance
    st.subheader("Feature Importance")
    feature_importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importances.values[:10], y=feature_importances.index[:10], palette="Reds", ax=ax)
    ax.set_title("Top 10 Feature Importances", fontsize=16, weight='bold')
    st.pyplot(fig)

    # Model Performance
    st.subheader("Model Performance")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Reds", ax=ax)
    ax.set_title("Confusion Matrix", fontsize=16, weight='bold')
    st.pyplot(fig)

# Fraud Detection Section
elif option == "Fraud Detection":
    st.header("Fraud Detection")

    # Basic anomaly detection based on claims history
    st.subheader("Anomaly Detection")
    anomalies = data_cleaned[
        data_cleaned["Previous Claims History"] > data_cleaned["Previous Claims History"].quantile(0.95)]
    st.write(f"Detected {len(anomalies)} anomalies with high claims history:")
    st.write(anomalies.head(10))

    # Visualizing Anomalies
    st.subheader("Claims History Distribution with Anomalies")
    fig, ax = plt.subplots()
    sns.histplot(data_cleaned["Previous Claims History"], kde=True, color="gray", ax=ax)
    sns.histplot(anomalies["Previous Claims History"], kde=True, color="red", ax=ax)
    ax.set_title("Claims History Distribution", fontsize=16, weight='bold')
    st.pyplot(fig)

st.success("Application is running!")
