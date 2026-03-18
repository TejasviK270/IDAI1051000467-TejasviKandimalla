import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Sales Insights",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Beyond Discounts: Data-Driven Black Friday Sales Insights")
st.markdown("**InsightMart Analytics** – Customer Purchase Pattern Analysis")
st.markdown("---")

# ─────────────────────────────────────────────
# STAGE 1: LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("BlackFriday.csv")
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error("❌ BlackFriday.csv not found. Please make sure it is in the same folder as app.py.")
    st.stop()

# ─────────────────────────────────────────────
# STAGE 2: DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def preprocess(df):
    df = df.copy()

    # Handle missing values
    df["Product_Category_2"].fillna(0, inplace=True)
    df["Product_Category_3"].fillna(0, inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Encode Gender: Male = 0, Female = 1
    df["Gender_enc"] = df["Gender"].map({"M": 0, "F": 1})

    # Encode Age groups into ordered numbers
    age_map = {
        "0-17": 1,
        "18-25": 2,
        "26-35": 3,
        "36-45": 4,
        "46-50": 5,
        "51-55": 6,
        "55+": 7
    }
    df["Age_enc"] = df["Age"].map(age_map)

    # Encode City_Category
    df["City_enc"] = df["City_Category"].map({"A": 1, "B": 2, "C": 3})

    # Encode Stay_In_Current_City_Years (handle '4+')
    df["Stay_enc"] = df["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)

    # Normalize Purchase (Min-Max)
    scaler = MinMaxScaler()
    df["Purchase_norm"] = scaler.fit_transform(df[["Purchase"]])

    return df

df = preprocess(df_raw)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
stage = st.sidebar.radio(
    "📋 Navigate Stages",
    [
        "📌 Project Scope",
        "🧹 Data Overview",
        "📊 EDA & Visualizations",
        "🔵 Clustering Analysis",
        "🔗 Association Rule Mining",
        "⚠️ Anomaly Detection",
        "💡 Insights & Report"
    ]
)

# ─────────────────────────────────────────────
# STAGE 1 – PROJECT SCOPE
# ─────────────────────────────────────────────
if stage == "📌 Project Scope":
    st.header("Stage 1: Project Scope Definition")
    st.markdown("""
    ### 🎯 Objectives
    As a Data Analyst at **InsightMart Analytics**, we aim to:

    - **Understand Shopping Preferences:** Discover how discounts influence purchases and which categories dominate sales.
    - **Segment Customers Effectively:** Use clustering to identify distinct shopping groups and tailor marketing strategies.
    - **Identify Cross-Selling Opportunities:** Apply association rule mining to uncover frequent product combinations.
    - **Detect Anomalies:** Spot unusual purchase behaviour such as exceptionally large transactions or bulk purchases.
    - **Deploy Insights:** Build this Streamlit app to visualize sales patterns and provide actionable insights.

    ### 📋 Dataset Columns
    | Column | Description |
    |---|---|
    | User_ID | Unique customer identifier |
    | Product_ID | Unique product identifier |
    | Gender | M / F |
    | Age | Age group bracket |
    | Occupation | Occupation code (0–20) |
    | City_Category | City type: A, B, or C |
    | Stay_In_Current_City_Years | Years lived in current city |
    | Marital_Status | 0 = Single, 1 = Married |
    | Product_Category_1/2/3 | Product category codes |
    | Purchase | Purchase amount in dollars |
    """)

    st.info(f"**Dataset loaded:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

# ─────────────────────────────────────────────
# STAGE 2 – DATA OVERVIEW
# ─────────────────────────────────────────────
elif stage == "🧹 Data Overview":
    st.header("Stage 2: Data Cleaning & Preprocessing")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Sample")
        st.dataframe(df_raw.head(10))

    with col2:
        st.subheader("Missing Values (Raw)")
        missing = df_raw.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        st.dataframe(missing)

    st.subheader("After Preprocessing")
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Shape:**", df.shape)
        st.write("**Duplicates removed.**")
        st.write("**Gender encoded:** M=0, F=1")
        st.write("**Age groups encoded:** 0-17→1 ... 55+→7")
        st.write("**Purchase normalized** using Min-Max scaling.")
    with col4:
        st.dataframe(df[["Gender", "Gender_enc", "Age", "Age_enc", "Purchase", "Purchase_norm"]].head(10))

    st.subheader("Descriptive Statistics")
    st.dataframe(df[["Age_enc", "Occupation", "Marital_Status", "Purchase"]].describe().round(2))

# ─────────────────────────────────────────────
# STAGE 3 – EDA
# ─────────────────────────────────────────────
elif stage == "📊 EDA & Visualizations":
    st.header("Stage 3: Exploratory Data Analysis (EDA)")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Purchase Distribution",
        "Product Categories",
        "Purchase vs Occupation",
        "Average Purchase",
        "Correlation Heatmap"
    ])

    # TAB 1 – Histograms / Boxplots by Age & Gender
    with tab1:
        st.subheader("Purchase Distribution by Age & Gender")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram by Age
        age_order = sorted(df["Age"].unique(), key=lambda x: {"0-17":1,"18-25":2,"26-35":3,"36-45":4,"46-50":5,"51-55":6,"55+":7}.get(x, 99))
        purchase_by_age = [df[df["Age"] == a]["Purchase"].values for a in age_order]
        axes[0].boxplot(purchase_by_age, labels=age_order)
        axes[0].set_title("Purchase by Age Group")
        axes[0].set_xlabel("Age Group")
        axes[0].set_ylabel("Purchase Amount ($)")
        axes[0].tick_params(axis='x', rotation=30)

        # Boxplot by Gender
        df.boxplot(column="Purchase", by="Gender", ax=axes[1])
        axes[1].set_title("Purchase by Gender")
        axes[1].set_xlabel("Gender")
        axes[1].set_ylabel("Purchase Amount ($)")
        plt.suptitle("")
        st.pyplot(fig)
        plt.close()

    # TAB 2 – Bar charts for product categories
    with tab2:
        st.subheader("Most Popular Product Categories")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(["Product_Category_1", "Product_Category_2", "Product_Category_3"]):
            cat_counts = df[df[col] > 0][col].value_counts().head(10)
            axes[i].bar(cat_counts.index.astype(str), cat_counts.values, color="steelblue")
            axes[i].set_title(f"Top 10: {col}")
            axes[i].set_xlabel("Category")
            axes[i].set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # TAB 3 – Scatter plots
    with tab3:
        st.subheader("Purchase vs Occupation & City Stay")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        occ_avg = df.groupby("Occupation")["Purchase"].mean().reset_index()
        axes[0].scatter(occ_avg["Occupation"], occ_avg["Purchase"], color="coral", s=80)
        axes[0].set_title("Avg Purchase vs Occupation")
        axes[0].set_xlabel("Occupation Code")
        axes[0].set_ylabel("Avg Purchase ($)")

        stay_avg = df.groupby("Stay_enc")["Purchase"].mean().reset_index()
        axes[1].scatter(stay_avg["Stay_enc"], stay_avg["Purchase"], color="mediumseagreen", s=80)
        axes[1].set_title("Avg Purchase vs Years in City")
        axes[1].set_xlabel("Years in Current City")
        axes[1].set_ylabel("Avg Purchase ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # TAB 4 – Average purchase per category
    with tab4:
        st.subheader("Average Purchase per Product Category 1")
        avg_cat = df.groupby("Product_Category_1")["Purchase"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        avg_cat.plot(kind="bar", ax=ax, color="mediumpurple")
        ax.set_title("Average Purchase per Product Category 1")
        ax.set_xlabel("Category")
        ax.set_ylabel("Avg Purchase ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # TAB 5 – Correlation heatmap
    with tab5:
        st.subheader("Correlation Heatmap")
        corr_cols = ["Age_enc", "Gender_enc", "Occupation", "City_enc",
                     "Stay_enc", "Marital_Status",
                     "Product_Category_1", "Product_Category_2",
                     "Product_Category_3", "Purchase"]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────
# STAGE 4 – CLUSTERING
# ─────────────────────────────────────────────
elif stage == "🔵 Clustering Analysis":
    st.header("Stage 4: K-Means Clustering Analysis")

    features = ["Age_enc", "Occupation", "Marital_Status", "Purchase_norm"]
    sample = df[features].dropna().sample(n=min(20000, len(df)), random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample)

    # Elbow method
    st.subheader("Elbow Method – Optimal Number of Clusters")
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K_range, inertias, "bo-", markersize=8)
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Apply K=4
    k = st.slider("Select number of clusters", 2, 8, 4)
    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    sample = sample.copy()
    sample["Cluster"] = km_final.fit_predict(X_scaled)

    # Cluster labels
    cluster_labels = {
        0: "💰 Budget Shoppers",
        1: "🛒 Discount Lovers",
        2: "⭐ Premium Buyers",
        3: "🎯 Occasional Spenders"
    }

    cluster_counts = sample["Cluster"].value_counts().sort_index()
    st.subheader("Cluster Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1","#ff9da7"]
    ax.bar(
        [cluster_labels.get(i, f"Cluster {i}") for i in cluster_counts.index],
        cluster_counts.values,
        color=colors[:k]
    )
    ax.set_title("Number of Customers per Cluster")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # PCA scatter
    st.subheader("Cluster Visualization (PCA 2D)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        mask = sample["Cluster"].values == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=cluster_labels.get(c, f"Cluster {c}"),
                   alpha=0.5, s=10, color=colors[c])
    ax.set_title("Customer Segments (PCA Projection)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cluster stats
    st.subheader("Cluster Summary Statistics")
    sample["Purchase_actual"] = sample["Purchase_norm"] * (df["Purchase"].max() - df["Purchase"].min()) + df["Purchase"].min()
    summary = sample.groupby("Cluster")[["Age_enc", "Occupation", "Marital_Status", "Purchase_actual"]].mean().round(2)
    summary.index = [cluster_labels.get(i, f"Cluster {i}") for i in summary.index]
    st.dataframe(summary)

# ─────────────────────────────────────────────
# STAGE 5 – ASSOCIATION RULE MINING
# ─────────────────────────────────────────────
elif stage == "🔗 Association Rule Mining":
    st.header("Stage 5: Association Rule Mining (Apriori)")
    st.markdown("Finding which **product categories** are frequently bought together.")

    # Build transactions: each User_ID -> list of Product_Category_1 values
    @st.cache_data
    def build_rules():
        transactions = df.groupby("User_ID")["Product_Category_1"].apply(
            lambda x: list(x.astype(str).unique())
        ).tolist()
        te = TransactionEncoder()
        te_arr = te.fit_transform(transactions)
        te_df = pd.DataFrame(te_arr, columns=te.columns_)
        freq_items = apriori(te_df, min_support=0.05, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
        return rules, freq_items

    rules, freq_items = build_rules()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Frequent Itemsets Found", len(freq_items))
    with col2:
        st.metric("Association Rules Generated", len(rules))

    st.subheader("Top Association Rules (by Lift)")
    rules_display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules_display = rules_display.sort_values("lift", ascending=False).head(15).reset_index(drop=True)
    rules_display[["support","confidence","lift"]] = rules_display[["support","confidence","lift"]].round(4)
    st.dataframe(rules_display)

    st.subheader("Support vs Confidence (Lift = bubble size)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        rules["support"], rules["confidence"],
        c=rules["lift"], cmap="YlOrRd", s=rules["lift"] * 20, alpha=0.7
    )
    plt.colorbar(sc, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association Rules: Support vs Confidence (colored by Lift)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────
# STAGE 6 – ANOMALY DETECTION
# ─────────────────────────────────────────────
elif stage == "⚠️ Anomaly Detection":
    st.header("Stage 6: Anomaly Detection – Unusual High Spenders")

    method = st.radio("Select detection method:", ["Z-Score", "IQR"])

    if method == "Z-Score":
        z_threshold = st.slider("Z-Score threshold", 2.0, 4.0, 3.0, 0.1)
        z_scores = np.abs(stats.zscore(df["Purchase"]))
        anomalies = df[z_scores > z_threshold].copy()
        normal = df[z_scores <= z_threshold].copy()
    else:
        Q1 = df["Purchase"].quantile(0.25)
        Q3 = df["Purchase"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        anomalies = df[(df["Purchase"] < lower) | (df["Purchase"] > upper)].copy()
        normal = df[(df["Purchase"] >= lower) & (df["Purchase"] <= upper)].copy()
        st.info(f"IQR bounds: ${lower:,.0f} – ${upper:,.0f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Anomalies Detected", f"{len(anomalies):,}")
    col3.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%")

    # Visualise
    st.subheader("Purchase Distribution with Anomalies Highlighted")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(normal["Purchase"], bins=60, color="steelblue", alpha=0.7, label="Normal")
    ax.hist(anomalies["Purchase"], bins=30, color="red", alpha=0.8, label="Anomaly")
    ax.set_title("Purchase Distribution – Normal vs Anomalous")
    ax.set_xlabel("Purchase Amount ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Anomaly Demographics")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots()
        anomalies["Age"].value_counts().sort_index().plot(kind="bar", ax=ax, color="tomato")
        ax.set_title("Anomalies by Age Group")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col_b:
        fig, ax = plt.subplots()
        anomalies["Occupation"].value_counts().head(10).plot(kind="bar", ax=ax, color="salmon")
        ax.set_title("Anomalies by Occupation (Top 10)")
        ax.set_xlabel("Occupation Code")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Sample Anomalous Records")
    st.dataframe(anomalies[["User_ID","Gender","Age","Occupation","City_Category","Purchase"]].head(20))

# ─────────────────────────────────────────────
# STAGE 7 – INSIGHTS & REPORT
# ─────────────────────────────────────────────
elif stage == "💡 Insights & Report":
    st.header("Stage 7: Key Insights & Business Recommendations")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Unique Customers", f"{df['User_ID'].nunique():,}")
    col3.metric("Avg Purchase ($)", f"${df['Purchase'].mean():,.0f}")
    col4.metric("Total Revenue ($)", f"${df['Purchase'].sum():,.0f}")

    st.markdown("---")

    st.subheader("🔍 Finding 1 – Which Age Group Spends the Most?")
    age_spend = df.groupby("Age")["Purchase"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    age_spend.plot(kind="bar", ax=ax, color="cornflowerblue")
    ax.set_title("Average Purchase by Age Group")
    ax.set_ylabel("Avg Purchase ($)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    top_age = age_spend.idxmax()
    st.success(f"✅ Age group **{top_age}** has the highest average spend of **${age_spend.max():,.0f}**.")

    st.subheader("🔍 Finding 2 – Popular Products by Gender")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, g in enumerate(["M", "F"]):
        top_cats = df[df["Gender"] == g]["Product_Category_1"].value_counts().head(8)
        axes[i].bar(top_cats.index.astype(str), top_cats.values, color="steelblue" if g == "M" else "lightcoral")
        axes[i].set_title(f"Top Categories – {'Male' if g=='M' else 'Female'}")
        axes[i].set_xlabel("Product Category 1")
        axes[i].set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("🔍 Finding 3 – City Category vs Spending")
    city_spend = df.groupby("City_Category")["Purchase"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    city_spend.plot(kind="bar", ax=ax, color=["#2196F3","#FF9800","#4CAF50"])
    ax.set_title("Average Purchase by City Category")
    ax.set_ylabel("Avg Purchase ($)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📋 Strategic Recommendations")
    st.markdown("""
    | # | Insight | Recommendation |
    |---|---|---|
    | 1 | Age group with highest spend identified | Target this group with premium product bundles |
    | 2 | Males buy certain categories more frequently | Create gender-specific promotional campaigns |
    | 3 | City A customers spend differently than B & C | Tailor discounts and offers by city type |
    | 4 | Association rules reveal frequent combos | Design combo-pack offers for those product pairs |
    | 5 | Anomaly users spend exceptionally high | Offer VIP/loyalty programmes to high-value outliers |
    """)
