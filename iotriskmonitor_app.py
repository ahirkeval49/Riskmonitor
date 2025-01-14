import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("data_synthetic.csv")

# Streamlit App Configuration
st.title("IoT Risk Monitoring and Insights")
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose an application:",
                              ["EDA", "Risk Assessment", "Predictive Modeling", "Fraud Detection"])

# Clean the dataset
data_cleaned = data.drop_duplicates().dropna()

# Convert categorical columns to numeric
categorical_columns = data_cleaned.select_dtypes(include=["object"]).columns
data_cleaned = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

# EDA Section
if option == "EDA":
    st.header("Exploratory Data Analysis")

    # Display dataset overview
    st.subheader("Dataset Overview")
    st.write(data_cleaned.head())

    # Distribution of Risk Profiles
    st.subheader("Distribution of Risk Profiles")
    fig, ax = plt.subplots()
    sns.countplot(data=data_cleaned, x="Risk Profile", palette="viridis", ax=ax)
    ax.set_title("Risk Profile Distribution")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_data = data_cleaned.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# Risk Assessment Section
elif option == "Risk Assessment":
    st.header("Risk Assessment")

    # Show high-risk customers
    high_risk_customers = data_cleaned[data_cleaned["Risk Profile"] == 3]
    st.subheader("High-Risk Customers")
    st.write(high_risk_customers)

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

    # Display results
    st.subheader("Model Performance")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Fraud Detection Section
elif option == "Fraud Detection":
    st.header("Fraud Detection")

    # Basic anomaly detection based on claims history
    st.subheader("Anomaly Detection")
    anomalies = data_cleaned[
        data_cleaned["Previous Claims History"] > data_cleaned["Previous Claims History"].quantile(0.95)]
    st.write(f"Detected {len(anomalies)} anomalies with high claims history:")
    st.write(anomalies)

st.success("Application is running!")
