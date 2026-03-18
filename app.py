import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# --- Page Config ---
st.set_page_config(page_title="InsightMart Black Friday Analytics", layout="wide")

# --- Stage 2: Data Cleaning & Preprocessing ---
@st.cache_data
def get_cleaned_data():
    # Load dataset 
    df = pd.read_csv('BlackFriday.csv')
    
    # Handle missing values in Product_Category_2 and 3 
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    
    # Encode categorical data: Gender and Age 
    df['Gender_Num'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Num'] = df['Age'].map(age_map)
    
    # Normalize purchase amounts for clustering 
    scaler = StandardScaler()
    df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
    
    return df

try:
    df = get_cleaned_data()
except FileNotFoundError:
    st.error("Error: 'BlackFriday.csv' not found. Please ensure the dataset is in the same folder.")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Project Navigation")
app_mode = st.sidebar.radio("Select a Stage:", 
    ["Project Scope", "Cleaned Dataset", "Exploratory Data Analysis", "Clustering Analysis", "Association Rule Mining", "Anomaly Detection", "Final Insights"])

# --- Stage 1: Define Project Scope ---
if app_mode == "Project Scope":
    st.header("Stage 1: Define Project Scope")
    st.write("The objective is to study the Black Friday dataset to understand shopping patterns, group customers into clusters, find product combinations, and detect unusual big spenders.")

# --- Stage 2: Display Cleaned Dataset ---
elif app_mode == "Cleaned Dataset":
    st.header("Stage 2: Final Cleaned Dataset")
    st.write("This table shows the preprocessed data, including encoded categories and handled missing values.")
    st.dataframe(df.head(100))
    st.write(f"Total rows in cleaned dataset: {len(df)}")

# --- Stage 3: Exploratory Data Analysis (EDA) ---
elif app_mode == "Exploratory Data Analysis":
    st.header("Stage 3: Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Purchase by Age Group")
        fig1, ax1 = plt.subplots()
        sns.barplot(data=df, x='Age', y='Purchase', palette='viridis', ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Purchase by Gender")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='Gender', y='Purchase', ax=ax2)
        st.pyplot(fig2)

# --- Stage 4: Clustering Analysis ---
elif app_mode == "Clustering Analysis":
    st.header("Stage 4: Clustering Analysis")
    st.write("Grouping customers based on Age and Purchase habits using K-Means.")
    
    # Use a sample for visualization performance
    sample_df = df.sample(2000, random_state=42)
    X = sample_df[['Age_Num', 'Purchase_Scaled']]
    
    k = st.slider("Select Number of Clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sample_df['Cluster'] = kmeans.fit_predict(X)
    
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=sample_df, x='Age', y='Purchase', hue='Cluster', palette='bright', ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# --- Stage 5: Association Rule Mining ---
elif app_mode == "Association Rule Mining":
    st.header("Stage 5: Association Rule Mining")
    st.write("Finding products usually bought together using the Apriori algorithm.")
    
    # Prepare basket data
    basket = df.sample(2000).groupby(['User_ID', 'Product_Category_1'])['Purchase'].count().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    freq_items = apriori(basket_sets, min_support=0.05, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(10))

# --- Stage 6: Anomaly Detection ---
elif app_mode == "Anomaly Detection":
    st.header("Stage 6: Anomaly Detection")
    st.write("Detecting unusually high spenders using statistical methods (IQR).")
    
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = df[df['Purchase'] > upper_bound]
    st.error(f"Upper Bound for Normal Spending: ${upper_bound:,.2f}")
    st.write(f"Total Anomalous Transactions Detected: {len(anomalies)}")
    st.dataframe(anomalies.head(50))

# --- Stage 7: Final Insights ---
elif app_mode == "Final Insights":
    st.header("Stage 7: Insights & Reporting")
    st.success("### Summary of Data Mining Findings")
    st.markdown("""
    * **Shopping Preferences:** Age group 26-35 and 36-45 represent the dominant spending blocks.
    * **Customer Segments:** Clustering revealed distinct groups: 'Budget Shoppers' (low purchase) and 'Premium Spenders' (high purchase regardless of age).
    * **Cross-Selling:** High lift values in Association Rules suggest specific categories are frequently paired, ideal for combo offers.
    * **Anomalies:** Detected high-value transactions that deviate significantly from average behavior, indicating bulk purchases or high-income individuals.
    """)
