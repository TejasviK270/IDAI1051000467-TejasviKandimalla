import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="InsightMart Analytics", layout="wide")

# --- Stage 2: Data Cleaning ---
@st.cache_data
def load_and_clean():
    df = pd.read_csv('BlackFriday.csv')
    # Fill missing values
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    # Numerical Encoding
    df['Gender_Num'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Num'] = df['Age'].map(age_map)
    return df

try:
    df = load_and_clean()
except Exception as e:
    st.error(f"Make sure BlackFriday.csv is in your GitHub folder! Error: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Project Stages")
page = st.sidebar.selectbox("Select Page", ["Cleaned Data", "EDA", "Clustering", "Anomalies", "Final Insights"])

# --- Pages ---
if page == "Cleaned Data":
    st.header("Stage 2: Final Cleaned Dataset")
    st.write("This table shows the preprocessed data ready for mining.")
    st.dataframe(df.head(100))

elif page == "EDA":
    st.header("Stage 3: Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spending by Age")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='Age', y='Purchase', ax=ax, palette='viridis')
        st.pyplot(fig) # This uses Matplotlib, NOT Altair
    with col2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots()
        df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

elif page == "Clustering":
    st.header("Stage 4: Customer Segmentation")
    # Scaling for K-Means
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Age_Num', 'Purchase']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df.sample(1000), x='Age', y='Purchase', hue='Cluster', ax=ax)
    st.pyplot(fig)

elif page == "Anomalies":
    st.header("Stage 6: Anomaly Detection")
    # IQR Method
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    limit = Q3 + (1.5 * IQR)
    anomalies = df[df['Purchase'] > limit]
    st.error(f"Detected {len(anomalies)} transactions above the threshold of ${limit:.2f}")
    st.dataframe(anomalies.head(20))

elif page == "Final Insights":
    st.header("Stage 7: Insights & Reporting")
    st.success("### Key Business Findings")
    st.write("- **Primary Target:** Male shoppers aged 26-35 contribute most to revenue.")
    st.write("- **Product Focus:** Product Category 1 is a clear bestseller across all segments.")
    st.write("- **Action:** Use 'Premium' clusters for high-end electronic promotions.")
