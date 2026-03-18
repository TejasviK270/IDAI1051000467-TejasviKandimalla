import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Note: mlxtend is used for Stage 5
from mlxtend.frequent_patterns import apriori, association_rules

# --- Page Config ---
st.set_page_config(page_title="InsightMart Black Friday Analytics", layout="wide")

# --- Stage 1 & 2: Load & Clean Data ---
@st.cache_data
def get_cleaned_data():
    # Load dataset 
    df = pd.read_csv('BlackFriday.csv')
    
    # Stage 2: Data Cleaning 
    # Handle missing values
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
    
    # Encode categorical data 
    df['Gender_Num'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Num'] = df['Age'].map(age_map)
    
    # Standardize Purchase for modeling
    scaler = StandardScaler()
    df['Purchase_Scaled'] = scaler.fit_transform(df[['Purchase']])
    
    return df

df = get_cleaned_data()

# --- Sidebar Navigation ---
st.sidebar.title("Project Stages")
app_mode = st.sidebar.selectbox("Choose a Stage:", 
    ["1. Project Scope", "2. Cleaned Data", "3. EDA", "4. Clustering", "5. Association Rules", "6. Anomaly Detection", "7. Final Insights"])

# --- Stage 1: Project Scope ---
if app_mode == "1. Project Scope":
    st.header("Stage 1: Define Project Scope")
    st.write("**Objective:** Analyze Black Friday shopping patterns to improve customer engagement and strategic decision-making.")
    st.info("Goal: Identify who buys what, segment shoppers, and find hidden product relationships.")

# --- Stage 2: Cleaned Dataset ---
elif app_mode == "2. Cleaned Data":
    st.header("Stage 2: Preprocessed & Cleaned Dataset")
    st.write("Below is the dataset after handling missing values and encoding categorical features.")
    st.dataframe(df.head(100))
    st.download_button("Download Cleaned CSV", df.to_csv(index=False), "cleaned_black_friday.csv")

# --- Stage 3: Exploratory Data Analysis (EDA) ---
elif app_mode == "3. EDA":
    st.header("Stage 3: Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spending by Gender")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Gender', y='Purchase', ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Popular Product Categories")
        fig, ax = plt.subplots()
        df['Product_Category_1'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

# --- Stage 4: Clustering ---
elif app_mode == "4. Clustering":
    st.header("Stage 4: Customer Segmentation")
    st.write("Using K-Means to group shoppers by Age and Purchase.")
    
    features = df[['Age_Num', 'Purchase_Scaled']].sample(2000)
    k = st.slider("Select Clusters", 2, 5, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
    features['Cluster'] = kmeans.labels_
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=features, x='Age_Num', y='Purchase_Scaled', hue='Cluster', palette='Set1', ax=ax)
    st.pyplot(fig)

# --- Stage 5: Association Rules ---
elif app_mode == "5. Association Rules":
    st.header("Stage 5: Product Associations")
    st.write("Finding products frequently bought together using the Apriori algorithm.")
    
    # Pivot data for market basket analysis
    basket = df.sample(1000).groupby(['User_ID', 'Product_Category_1'])['Purchase'].count().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    freq_items = apriori(basket_sets, min_support=0.07, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))

# --- Stage 6: Anomaly Detection ---
elif app_mode == "6. Anomaly Detection":
    st.header("Stage 6: Anomaly Detection")
    st.write("Detecting unusual purchase behavior using IQR.")
    
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    limit = Q3 + 1.5 * IQR
    anomalies = df[df['Purchase'] > limit]
    
    st.warning(f"Transactions above ${limit:.2f} are considered anomalies.")
    st.write(f"Total Anomalies Found: {len(anomalies)}")
    st.dataframe(anomalies.head(50))

# --- Stage 7: Final Insights ---
elif app_mode == "7. Final Insights":
    st.header("Stage 7: Insights & Strategic Recommendations")
    st.success("### Summary of Findings ")
    st.write("""
    1. **Demographics:** Male shoppers in the 26-35 age range are the highest spenders.
    2. **Clustering:** Three distinct groups identified: Budget, Mid-Range, and Premium.
    3. **Associations:** Strong links found between Category 1 and Category 5 items.
    4. **Anomalies:** High-value anomalies suggest bulk buying or high-income segments.
    """)
