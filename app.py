import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="InsightMart Black Friday Analytics", layout="wide")

# --- Stage 2: Data Cleaning ---
@st.cache_data
def get_cleaned_data():
    df = pd.read_csv('BlackFriday.csv')
    # Handle missing values
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    # Numerical Encoding
    df['Gender_Num'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Num'] = df['Age'].map(age_map)
    return df

try:
    df = get_cleaned_data()
except Exception as e:
    st.error(f"Dataset not found! Ensure 'BlackFriday.csv' is in your GitHub folder. {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("Project Navigation")
app_mode = st.sidebar.selectbox("Choose a Stage:", 
    ["1. Project Scope", "2. Cleaned Dataset", "3. EDA", "4. Clustering", "6. Anomaly Detection", "7. Final Insights"])

# --- Stage 1: Scope ---
if app_mode == "1. Project Scope":
    st.header("Stage 1: Define Project Scope")
    st.write("Objective: Analyze Black Friday purchase data to identify customer segments and spending anomalies.")

# --- Stage 2: Cleaned Dataset ---
elif app_mode == "2. Cleaned Dataset":
    st.header("Stage 2: Final Cleaned Dataset")
    st.dataframe(df.head(100))
    st.write(f"Total Records: {len(df)}")

# --- Stage 3: EDA (Standard Plots) ---
elif app_mode == "3. EDA":
    st.header("Stage 3: Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Purchase Amount by Age")
        fig1, ax1 = plt.subplots()
        sns.barplot(data=df, x='Age', y='Purchase', palette='viridis', ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Top Product Categories")
        fig2, ax2 = plt.subplots()
        df['Product_Category_1'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        st.pyplot(fig2)

# --- Stage 4: Clustering ---
elif app_mode == "4. Clustering":
    st.header("Stage 4: Customer Segmentation")
    st.write("Grouping customers using K-Means Clustering.")
    
    # Scale and Cluster
    scaler = StandardScaler()
    sample_df = df.sample(1000, random_state=42).copy()
    scaled_data = scaler.fit_transform(sample_df[['Age_Num', 'Purchase']])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    sample_df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=sample_df, x='Age', y='Purchase', hue='Cluster', palette='bright', ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# --- Stage 6: Anomaly Detection ---
elif app_mode == "6. Anomaly Detection":
    st.header("Stage 6: Anomaly Detection")
    limit = df['Purchase'].quantile(0.75) + 1.5 * (df['Purchase'].quantile(0.75) - df['Purchase'].quantile(0.25))
    anomalies = df[df['Purchase'] > limit]
    st.error(f"Detected {len(anomalies)} anomalies above ${limit:,.2f}")
    st.dataframe(anomalies.head(20))

# --- Stage 7: Final Insights ---
elif app_mode == "7. Final Insights":
    st.header("Stage 7: Insights & Reporting")
    st.success("### Final Findings")
    st.markdown("""
    - **Customer Base:** 26-35 year olds are the most frequent shoppers.
    - **Segmentation:** Clustering successfully divided users into 'Budget', 'Regular', and 'Premium' tiers.
    - **Anomalies:** The detected outliers likely represent wholesale buyers or bulk transactions.
    """)
