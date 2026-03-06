```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analysis and Retention Strategy Recommendation System",
    layout="wide"
)

st.title("📊 Customer Churn Analysis and Retention Strategy Recommendation System")

# ------------------------------------------------
# LOAD MODEL FILES
# ------------------------------------------------

model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

features = scaler.feature_names_in_

# ------------------------------------------------
# LOAD DATASET (FROM GITHUB REPO)
# ------------------------------------------------

df = pd.read_csv("customer_data.csv")

st.subheader("Customer Dataset Preview")
st.dataframe(df.head())

# ------------------------------------------------
# CONVERT CATEGORICAL DATA
# ------------------------------------------------

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = pd.factorize(df[col])[0]

# ------------------------------------------------
# ALIGN DATA WITH MODEL FEATURES
# ------------------------------------------------

for col in features:
    if col not in df.columns:
        df[col] = 0

X = df[features]

# ------------------------------------------------
# SCALE DATA
# ------------------------------------------------

X_scaled = scaler.transform(X)

# ------------------------------------------------
# PREDICTIONS
# ------------------------------------------------

pred = model.predict(X_scaled)
prob = model.predict_proba(X_scaled)

df["Churn Prediction"] = pred
df["Churn Probability"] = prob[:,1] * 100

# ------------------------------------------------
# CUSTOMER LIFETIME VALUE
# ------------------------------------------------

if "MonthlyCharges" in df.columns and "tenure" in df.columns:
    df["CLV"] = df["MonthlyCharges"] * df["tenure"] * (1 - prob[:,1])
else:
    df["CLV"] = 0

# ------------------------------------------------
# RETENTION STRATEGY FUNCTION
# ------------------------------------------------

def retention_strategy(row):

    strategies = []

    if row["Churn Probability"] > 70:
        strategies.append("Offer loyalty discount")

    if "tenure" in row and row["tenure"] < 12:
        strategies.append("Provide onboarding support")

    if "MonthlyCharges" in row and row["MonthlyCharges"] > 80:
        strategies.append("Suggest lower price plan")

    if len(strategies) == 0:
        strategies.append("Maintain service quality")

    return ", ".join(strategies)

df["Retention Strategy"] = df.apply(retention_strategy, axis=1)

# ------------------------------------------------
# CUSTOMER SEGMENTATION
# ------------------------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ------------------------------------------------
# KPI METRICS
# ------------------------------------------------

churn_rate = df["Churn Prediction"].mean() * 100
avg_clv = df["CLV"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Average CLV", f"${avg_clv:.2f}")

# ------------------------------------------------
# CHURN RISK GAUGE
# ------------------------------------------------

st.subheader("Overall Churn Risk")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=churn_rate,
    title={'text': "Churn Risk Level"},
    gauge={
        'axis': {'range': [0,100]},
        'steps':[
            {'range':[0,40],'color':"green"},
            {'range':[40,70],'color':"yellow"},
            {'range':[70,100],'color':"red"}
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# TABS
# ------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Dashboard",
    "Churn Predictions",
    "Retention Strategies",
    "Customer Segmentation"
])

# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------

with tab1:

    st.subheader("Churn Distribution")

    fig = px.pie(df, names="Churn Prediction")
    st.plotly_chart(fig)

    if "MonthlyCharges" in df.columns:

        st.subheader("Monthly Charges vs Churn")

        fig = px.box(df, x="Churn Prediction", y="MonthlyCharges")
        st.plotly_chart(fig)

    if "tenure" in df.columns:

        st.subheader("Tenure vs Churn")

        fig = px.histogram(df, x="tenure", color="Churn Prediction")
        st.plotly_chart(fig)

# ------------------------------------------------
# PREDICTIONS TABLE
# ------------------------------------------------

with tab2:

    st.subheader("Customer Churn Predictions")

    st.dataframe(df[
        ["Churn Prediction","Churn Probability","CLV"]
    ])

# ------------------------------------------------
# RETENTION STRATEGIES
# ------------------------------------------------

with tab3:

    st.subheader("Recommended Retention Strategies")

    st.dataframe(df[
        ["Churn Probability","Retention Strategy"]
    ])

# ------------------------------------------------
# SEGMENTATION
# ------------------------------------------------

with tab4:

    if "MonthlyCharges" in df.columns and "tenure" in df.columns:

        fig = px.scatter(
            df,
            x="MonthlyCharges",
            y="tenure",
            color="Cluster",
            title="Customer Segments"
        )

        st.plotly_chart(fig)

# ------------------------------------------------
# HIGH RISK CUSTOMERS
# ------------------------------------------------

st.subheader("High Risk Customers")

high_risk = df[df["Churn Probability"] > 70]

st.dataframe(high_risk)

# ------------------------------------------------
# DOWNLOAD RESULTS
# ------------------------------------------------

csv = df.to_csv(index=False)

st.download_button(
    label="Download Prediction Results",
    data=csv,
    file_name="churn_predictions.csv"
)
```
