import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="Black Friday Insights", layout="wide")

st.title("🛒 Beyond Discounts: Black Friday Sales Insights")
st.markdown("Exploring customer behavior, segments, and anomalies.")

# --- Stage 2: Load & Clean Data ---
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('BlackFriday.csv')
    
    # Handle missing values
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    
    # Encoding for Analysis
    df['Gender_Numeric'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Numeric'] = df['Age'].map(age_map)
    
    return df

df = load_data()

# --- Sidebar Navigation ---
menu = st.sidebar.selectbox("Select Stage", ["EDA", "Clustering", "Anomaly Detection"])

# --- Stage 3: Exploratory Data Analysis (EDA) ---
if menu == "EDA":
    st.header("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Purchase Distribution by Gender")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Gender', y='Purchase', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Top Product Categories")
        fig, ax = plt.subplots()
        df['Product_Category_1'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

# --- Stage 4: Clustering Analysis ---
elif menu == "Clustering":
    st.header("🤖 Customer Segmentation (K-Means)")
    
    # Prepare features for clustering
    features = df[['Age_Numeric', 'Gender_Numeric', 'Purchase']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    num_clusters = st.slider("Select Number of Clusters", 2, 5, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df.sample(1000), x='Age', y='Purchase', hue='Cluster', palette='viridis', ax=ax)
    st.write(f"Visualizing {num_clusters} Customer Segments")
    st.pyplot(fig)

# --- Stage 6: Anomaly Detection ---
elif menu == "Anomaly Detection":
    st.header("⚠️ Anomaly Detection")
    st.write("Identifying unusually high spenders using the IQR method.")
    
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = df[df['Purchase'] > upper_bound]
    st.write(f"Detected {len(anomalies)} anomalies (Transactions > ${upper_bound:,.2f})")
    st.dataframe(anomalies.head(20))
