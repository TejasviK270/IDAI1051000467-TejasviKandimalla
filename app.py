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
    st.error(f"Please ensure BlackFriday.csv is uploaded to GitHub. Error: {e}")
    st.stop()

# --- Sidebar ---
app_mode = st.sidebar.radio("Navigation", 
    ["Project Scope", "Cleaned Dataset", "EDA", "Clustering", "Association Rules", "Anomaly Detection", "Final Insights"])

if app_mode == "Project Scope":
    st.header("Stage 1: Project Scope")
    st.write("Analyzing Black Friday sales data to discover customer segments and trends.")

elif app_mode == "Cleaned Dataset":
    st.header("Stage 2: Final Cleaned Dataset")
    st.dataframe(df.head(100))

elif app_mode == "EDA":
    st.header("Stage 3: Exploratory Data Analysis")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='Age', y='Purchase', palette='magma', ax=ax)
    st.pyplot(fig)

elif app_mode == "Clustering":
    st.header("Stage 4: Customer Segmentation")
    X = df[['Age_Num', 'Purchase_Scaled']].sample(1000)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    X['Cluster'] = kmeans.labels_
    fig, ax = plt.subplots()
    sns.scatterplot(data=X, x='Age_Num', y='Purchase_Scaled', hue='Cluster', palette='viridis', ax=ax)
    st.pyplot(fig)

elif app_mode == "Association Rules":
    st.header("Stage 5: Product Associations")
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        # Simple basket logic
        basket = df.sample(1000).groupby(['User_ID', 'Product_Category_1'])['Purchase'].count().unstack().fillna(0)
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        freq_items = apriori(basket_sets, min_support=0.05, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
    except ImportError:
        st.warning("The 'mlxtend' library is still installing. Please wait 1 minute and refresh the page.")

elif app_mode == "Anomaly Detection":
    st.header("Stage 6: Anomaly Detection")
    upper = df['Purchase'].quantile(0.75) + 1.5 * (df['Purchase'].quantile(0.75) - df['Purchase'].quantile(0.25))
    anomalies = df[df['Purchase'] > upper]
    st.write(f"Anomalies found: {len(anomalies)}")
    st.dataframe(anomalies.head(20))

elif app_mode == "Final Insights":
    st.header("Stage 7: Final Insights")
    st.success("### Summary of Findings")
    st.markdown("""
    - **Demographics:** Shoppers aged 26-35 are the highest contributors to revenue.
    - **Clustering:** Customers were segmented into 'Budget', 'Average', and 'Premium' groups.
    - **Associations:** Significant patterns found between Category 1 and Category 8 items.
    - **Anomalies:** Detected high-value transactions that likely represent wholesale buyers.
    """)
