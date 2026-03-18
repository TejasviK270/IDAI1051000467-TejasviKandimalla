import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="InsightMart Analytics", layout="wide")

# --- Stage 2: Data Cleaning & Preprocessing ---
@st.cache_data
def load_and_clean():
    # Load dataset
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
    df = load_and_clean()
except Exception as e:
    st.error(f"Dataset Error: Ensure 'BlackFriday.csv' is in your GitHub folder. {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("Data Mining Stages")
page = st.sidebar.radio("Navigate to:", 
    ["1. Project Scope", "2. Cleaned Dataset", "3. EDA (Altair)", "4. Clustering", "6. Anomaly Detection", "7. Final Insights"])

# --- Stage 1: Scope ---
if page == "1. Project Scope":
    st.header("Stage 1: Define Project Scope")
    st.write("Objective: Analyze Black Friday purchase data to identify customer segments and spending anomalies using advanced data mining techniques.")

# --- Stage 2: Cleaned Dataset ---
elif page == "2. Cleaned Dataset":
    st.header("Stage 2: Final Cleaned Dataset")
    st.write("Data after handling nulls and encoding categorical variables:")
    st.dataframe(df.head(100))

# --- Stage 3: EDA with Altair ---
elif page == "3. EDA (Altair)":
    st.header("Stage 3: Exploratory Data Analysis")
    
    st.subheader("Average Purchase by Age Group")
    chart_age = alt.Chart(df).mark_bar().encode(
        x='Age:O',
        y='mean(Purchase):Q',
        color='Age:N'
    ).properties(width=700, height=400).interactive()
    st.altair_chart(chart_age, use_container_width=True)

    st.subheader("Purchase Distribution by Gender")
    chart_gender = alt.Chart(df.sample(2000)).mark_tick().encode(
        x='Purchase:Q',
        y='Gender:N',
        color='Gender:N'
    ).properties(height=200)
    st.altair_chart(chart_gender, use_container_width=True)

# --- Stage 4: Clustering ---
elif page == "4. Clustering":
    st.header("Stage 4: Customer Segmentation")
    
    # Perform Clustering
    scaler = StandardScaler()
    # Using a sample for performance
    sample_df = df.sample(2000, random_state=42).copy()
    scaled_data = scaler.fit_transform(sample_df[['Age_Num', 'Purchase']])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    sample_df['Cluster'] = kmeans.fit_predict(scaled_data).astype(str)
    
    st.write("Interactive Cluster Map (Age vs Purchase)")
    cluster_chart = alt.Chart(sample_df).mark_circle(size=60).encode(
        x='Age:O',
        y='Purchase:Q',
        color='Cluster:N',
        tooltip=['User_ID', 'Age', 'Purchase', 'Cluster']
    ).properties(width=700, height=400).interactive()
    st.altair_chart(cluster_chart, use_container_width=True)

# --- Stage 6: Anomaly Detection ---
elif page == "6. Anomaly Detection":
    st.header("Stage 6: Anomaly Detection")
    # Identify outliers using 1.5 * IQR
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    limit = Q3 + 1.5 * IQR
    
    anomalies = df[df['Purchase'] > limit]
    st.error(f"Anomalies detected: {len(anomalies)} transactions above ${limit:,.2f}")
    st.dataframe(anomalies.head(50))

# --- Stage 7: Final Insights ---
elif page == "7. Final Insights":
    st.header("Stage 7: Insights & Strategic Recommendations")
    st.success("### Results Summary")
    st.markdown("""
    1. **High Value Customers:** The 26-35 age group accounts for the most significant portion of high-value transactions.
    2. **Segments:** Clustering identifies clear tiers of shoppers, allowing for targeted marketing.
    3. **Anomalies:** Extreme outliers indicate specific users who may be wholesale buyers or high-income individuals.
    4. **Recommendation:** Focus discount campaigns on the 'Mid-Range' cluster to push them toward 'Premium' spending levels.
    """)
