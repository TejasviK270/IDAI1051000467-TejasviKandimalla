import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# --- Page Config ---
st.set_page_config(page_title="InsightMart Analytics - Black Friday", layout="wide")

# --- Stage 1: Define the Project Scope ---
st.title("Mining the Future: Black Friday Sales Insights")
st.markdown("""
**Scenario 1:** Beyond Discounts – Data Driven Black Friday Sales Insights.
This dashboard identifies shopping behaviors, segments customers, uncovers product associations, and detects spending anomalies.
""")

# --- Stage 2: Data Cleaning & Preprocessing ---
@st.cache_data
def load_and_clean_data():
    # Load dataset
    df = pd.read_csv('BlackFriday.csv')
    
    # Handle missing values in Product_Category_2 & 3 
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    
    # Encode categorical data 
    df['Gender_Num'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Num'] = df['Age'].map(age_map)
    
    return df

df = load_and_clean_data()

# --- Sidebar ---
st.sidebar.header("Navigation")
stage = st.sidebar.radio("Go to:", ["EDA", "Clustering", "Association Rules", "Anomaly Detection"])

# --- Stage 3: Exploratory Data Analysis (EDA) ---
if stage == "EDA":
    st.header("Stage 3: Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Purchase by Age Group")
        fig1, ax1 = plt.subplots()
        sns.barplot(data=df, x='Age', y='Purchase', palette='magma', ax=ax1)
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Top Product Categories")
        fig2, ax2 = plt.subplots()
        df['Product_Category_1'].value_counts().head(10).plot(kind='bar', ax=ax2, color='skyblue')
        st.pyplot(fig2)

# --- Stage 4: Clustering Analysis ---
elif stage == "Clustering":
    st.header("Stage 4: Customer Segmentation")
    st.write("Grouping customers based on Age, Gender, and Purchase amount.")
    
    # Prepare features
    cluster_data = df[['Age_Num', 'Gender_Num', 'Purchase']].sample(5000) # Sample for speed
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    k = st.slider("Select Number of Clusters (k)", 2, 6, 3)
    model = KMeans(n_clusters=k, random_state=42)
    cluster_data['Cluster'] = model.fit_predict(scaled_data)
    
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=cluster_data, x='Age_Num', y='Purchase', hue='Cluster', palette='viridis', ax=ax3)
    st.pyplot(fig3)
    st.write("**Insight:** Use these clusters to tailor marketing strategies.")

# --- Stage 5: Association Rule Mining ---
elif stage == "Association Rules":
    st.header("Stage 5: Product Associations")
    st.write("Identifying product categories frequently bought together.")
    
    # Simple cross-tab for association (Product_Category_1 & 2)
    basket = df.sample(1000).groupby(['User_ID', 'Product_Category_1'])['Purchase'].count().unstack().reset_index().fillna(0).set_index('User_ID')
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    st.write("Frequent Itemsets & Rules (Top 10):")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# --- Stage 6: Anomaly Detection ---
elif stage == "Anomaly Detection":
    st.header("Stage 6: Identifying Big Spenders")
    st.write("Detecting unusual purchase behaviors using the IQR method.")
    
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_step = Q3 + (1.5 * IQR)
    
    anomalies = df[df['Purchase'] > outlier_step]
    
    st.error(f"Alert: Found {len(anomalies)} transactions above the threshold of ${outlier_step:,.2f}")
    st.dataframe(anomalies[['User_ID', 'Age', 'Gender', 'Purchase']].head(20))

# --- Stage 7: Insights ---
st.sidebar.markdown("---")
st.sidebar.info("**Stage 7: Key Insight**\nIdentify your most loyal age groups and highest-performing categories to optimize Black Friday inventory.")
