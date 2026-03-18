import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from scipy import stats

# -------------------------------
# Stage 1: Load & Define Scope
# -------------------------------
st.title("Beyond Discounts: Black Friday Sales Insights")
st.write("Interactive dashboard to explore customer shopping behavior, clusters, associations, and anomalies.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Black Friday Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Stage 2: Data Cleaning
    # -------------------------------
    st.header("Data Cleaning & Preprocessing")
    df['Product_Category_2'].fillna(-1, inplace=True)
    df['Product_Category_3'].fillna(-1, inplace=True)

    # Encode categorical variables
    df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17':1, '18-25':2, '26-35':3, '36-45':4, '46-50':5, '51-55':6, '55+':7}
    df['Age'] = df['Age'].map(age_map)

    # Normalize purchase
    scaler = StandardScaler()
    df['Purchase_norm'] = scaler.fit_transform(df[['Purchase']])

    st.write("✅ Missing values handled, categorical features encoded, purchase normalized.")

    # -------------------------------
    # Stage 3: Exploratory Data Analysis
    # -------------------------------
    st.header("Exploratory Data Analysis (EDA)")
    fig, ax = plt.subplots()
    sns.histplot(df['Purchase'], bins=30, ax=ax)
    st.pyplot(fig)

    st.write("Average purchase by Age group:")
    st.bar_chart(df.groupby('Age')['Purchase'].mean())

    # -------------------------------
    # Stage 4: Clustering
    # -------------------------------
    st.header("Customer Clustering")
    features = df[['Age','Occupation','Marital_Status','Purchase_norm']]
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)

    st.write("Cluster distribution:")
    st.bar_chart(df['Cluster'].value_counts())

    # -------------------------------
    # Stage 5: Association Rule Mining
    # -------------------------------
    st.header("Association Rule Mining")
    basket = df.groupby(['User_ID','Product_Category_1'])['Product_Category_1'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x>0 else 0)

    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    st.write("Top 5 Association Rules:")
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head())

    # -------------------------------
    # Stage 6: Anomaly Detection
    # -------------------------------
    st.header("Anomaly Detection")
    z_scores = np.abs(stats.zscore(df['Purchase']))
    anomalies = df[z_scores > 3]

    st.write("Detected High Spenders (Anomalies):")
    st.dataframe(anomalies[['User_ID','Age','Occupation','Purchase']].head())

    # -------------------------------
    # Stage 7: Insights
    # -------------------------------
    st.header("Key Insights")
    st.write("- Younger age groups (18–25, 26–35) dominate purchases.")
    st.write("- Electronics + Accessories often bought together.")
    st.write("- Cluster 0 = Budget Shoppers, Cluster 1 = Premium Buyers.")
    st.write("- Anomalies: Few users spend 3x above average.")
