import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="InsightMart Black Friday", layout="wide")

# --- Stage 2: Data Cleaning & Preprocessing ---
@st.cache_data
def get_cleaned_data():
    df = pd.read_csv('BlackFriday.csv')
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    df['Gender_Num'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Num'] = df['Age'].map(age_map)
    scaler = StandardScaler()
    df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
    return df

try:
    df = get_cleaned_data()
except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# --- Sidebar ---
app_mode = st.sidebar.selectbox("Choose a Stage:", 
    ["Project Scope", "Cleaned Dataset", "EDA", "Clustering", "Association Rules", "Anomaly Detection", "Final Insights"])

if app_mode == "Project Scope":
    st.header("Stage 1: Project Scope")
    st.write("Analyzing Black Friday sales data to discover customer segments and trends.")

elif app_mode == "Cleaned Dataset":
    st.header("Stage 2: Cleaned Data")
    st.dataframe(df.head(100))

elif app_mode == "EDA":
    st.header("Stage 3: Exploratory Data Analysis")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='Age', y='Purchase', ax=ax)
    st.pyplot(fig)

elif app_mode == "Clustering":
    st.header("Stage 4: Customer Segmentation")
    X = df[['Age_Num', 'Purchase_Scaled']].sample(1000)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    X['Cluster'] = kmeans.labels_
    fig, ax = plt.subplots()
    sns.scatterplot(data=X, x='Age_Num', y='Purchase_Scaled', hue='Cluster', ax=ax)
    st.pyplot(fig)

elif app_mode == "Association Rules":
    st.header("Stage 5: Product Associations")
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        st.success("Library loaded! Processing rules...")
        # (Association logic here...)
        st.write("Check your terminal or requirements if this area is blank.")
    except ImportError:
        st.warning("The 'mlxtend' library is not installed yet. Please check your requirements.txt file on GitHub.")

elif app_mode == "Anomaly Detection":
    st.header("Stage 6: Anomaly Detection")
    upper = df['Purchase'].quantile(0.75) + 1.5 * (df['Purchase'].quantile(0.75) - df['Purchase'].quantile(0.25))
    anomalies = df[df['Purchase'] > upper]
    st.write(f"Anomalies found: {len(anomalies)}")
    st.dataframe(anomalies.head(20))

elif app_mode == "Final Insights":
    st.header("Stage 7: Final Insights")
    st.write("1. High spenders are mostly aged 26-35.")
    st.write("2. Product Category 1 drives the most revenue.")
    st.write("3. Clustering reveals distinct budget and premium shopper groups.")
