import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Sales Insights",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Beyond Discounts: Black Friday Sales Insights")
st.markdown("**InsightMart Analytics** – Customer Purchase Pattern Analysis")
st.markdown("---")

# ──────────────────────────────────────────────────────────────
# LOAD & PREPROCESS  (cached so it only runs once)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_and_preprocess():
    df = pd.read_csv("BlackFriday.csv")
    df["Product_Category_2"] = df["Product_Category_2"].fillna(0).astype(int)
    df["Product_Category_3"] = df["Product_Category_3"].fillna(0).astype(int)
    df = df.drop_duplicates()
    df["Gender_enc"] = df["Gender"].map({"M": 0, "F": 1})
    age_map = {"0-17": 1, "18-25": 2, "26-35": 3,
               "36-45": 4, "46-50": 5, "51-55": 6, "55+": 7}
    df["Age_enc"] = df["Age"].map(age_map)
    df["City_enc"] = df["City_Category"].map({"A": 1, "B": 2, "C": 3})
    df["Stay_enc"] = df["Stay_In_Current_City_Years"].replace("4+", "4").astype(int)
    mm = MinMaxScaler()
    df["Purchase_norm"] = mm.fit_transform(df[["Purchase"]])
    return df

try:
    df = load_and_preprocess()
except FileNotFoundError:
    st.error("❌ BlackFriday.csv not found. Place it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

AGE_ORDER = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
stage = st.sidebar.radio(
    "📋 Navigate Stages",
    [
        "📌 Project Scope",
        "🧹 Data Overview",
        "📊 EDA & Visualizations",
        "🔵 Clustering Analysis",
        "🔗 Association Rule Mining",
        "⚠️ Anomaly Detection",
        "💡 Insights & Report",
    ],
)

def show(fig):
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════
# STAGE 1 – PROJECT SCOPE
# ══════════════════════════════════════════════════════════════
if stage == "📌 Project Scope":
    st.header("Stage 1: Project Scope Definition")
    st.markdown("""
### 🎯 Objectives
As a Data Analyst at **InsightMart Analytics** we aim to:

- **Understand Shopping Preferences** – discover how discounts influence purchases and which categories dominate sales.
- **Segment Customers Effectively** – use clustering to identify distinct shopping groups and tailor marketing strategies.
- **Identify Cross-Selling Opportunities** – apply association rule mining to uncover frequent product combinations.
- **Detect Anomalies** – spot unusual purchase behaviour such as exceptionally large transactions.
- **Deploy Insights** – build this Streamlit app to visualise sales patterns and provide actionable insights.

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
    st.info(f"**Dataset loaded:** {df.shape[0]:,} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════
# STAGE 2 – DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
elif stage == "🧹 Data Overview":
    st.header("Stage 2: Data Cleaning & Preprocessing")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Sample")
        st.dataframe(df.head(10))
    with c2:
        st.subheader("Missing Values (after cleaning)")
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column", "Missing"]
        st.dataframe(mv)

    st.subheader("Preprocessing steps applied")
    st.markdown("""
- ✅ Missing values in `Product_Category_2` & `_3` filled with **0**
- ✅ Duplicates removed
- ✅ Gender encoded: M = 0, F = 1
- ✅ Age groups encoded: 0-17 → 1 … 55+ → 7
- ✅ Purchase normalised using **Min-Max scaling**
""")
    st.dataframe(
        df[["Gender", "Gender_enc", "Age", "Age_enc",
            "Purchase", "Purchase_norm"]].head(10)
    )
    st.subheader("Descriptive Statistics")
    st.dataframe(df[["Age_enc","Occupation","Marital_Status","Purchase"]].describe().round(2))

# ══════════════════════════════════════════════════════════════
# STAGE 3 – EDA
# ══════════════════════════════════════════════════════════════
elif stage == "📊 EDA & Visualizations":
    st.header("Stage 3: Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Purchase by Age & Gender",
        "Product Categories",
        "Purchase vs Occupation",
        "Avg Purchase per Category",
        "Correlation Heatmap",
    ])

    with tab1:
        st.subheader("Boxplots – Purchase by Age Group & Gender")
        valid_ages = [a for a in AGE_ORDER if a in df["Age"].unique()]
        data_by_age = [df.loc[df["Age"] == a, "Purchase"].values for a in valid_ages]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].boxplot(data_by_age, labels=valid_ages, showfliers=False)
        axes[0].set_title("Purchase by Age Group")
        axes[0].set_xlabel("Age Group")
        axes[0].set_ylabel("Purchase ($)")
        axes[0].tick_params(axis="x", rotation=30)
        df.boxplot(column="Purchase", by="Gender", ax=axes[1], showfliers=False)
        axes[1].set_title("Purchase by Gender")
        axes[1].set_xlabel("Gender (M / F)")
        axes[1].set_ylabel("Purchase ($)")
        plt.suptitle("")
        plt.tight_layout()
        show(fig)

    with tab2:
        st.subheader("Top 10 Product Categories")
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for i, col in enumerate(["Product_Category_1",
                                   "Product_Category_2",
                                   "Product_Category_3"]):
            counts = df[df[col] > 0][col].value_counts().head(10)
            axes[i].bar(counts.index.astype(str), counts.values, color="steelblue")
            axes[i].set_title(f"Top 10 – {col}")
            axes[i].set_xlabel("Category")
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        show(fig)

    with tab3:
        st.subheader("Average Purchase vs Occupation & City Stay")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        occ = df.groupby("Occupation")["Purchase"].mean()
        axes[0].scatter(occ.index, occ.values, color="coral", s=80)
        axes[0].set_title("Avg Purchase vs Occupation")
        axes[0].set_xlabel("Occupation Code")
        axes[0].set_ylabel("Avg Purchase ($)")
        stay = df.groupby("Stay_enc")["Purchase"].mean()
        axes[1].scatter(stay.index, stay.values, color="mediumseagreen", s=80)
        axes[1].set_title("Avg Purchase vs Years in City")
        axes[1].set_xlabel("Years in Current City")
        axes[1].set_ylabel("Avg Purchase ($)")
        plt.tight_layout()
        show(fig)

    with tab4:
        st.subheader("Average Purchase per Product Category 1")
        avg_cat = df.groupby("Product_Category_1")["Purchase"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))
        avg_cat.plot(kind="bar", ax=ax, color="mediumpurple")
        ax.set_title("Average Purchase per Product Category 1")
        ax.set_xlabel("Category")
        ax.set_ylabel("Avg Purchase ($)")
        plt.tight_layout()
        show(fig)

    with tab5:
        st.subheader("Correlation Heatmap")
        corr_cols = ["Age_enc","Gender_enc","Occupation","City_enc",
                     "Stay_enc","Marital_Status",
                     "Product_Category_1","Product_Category_2",
                     "Product_Category_3","Purchase"]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        show(fig)

# ══════════════════════════════════════════════════════════════
# STAGE 4 – CLUSTERING
# ══════════════════════════════════════════════════════════════
elif stage == "🔵 Clustering Analysis":
    st.header("Stage 4: K-Means Clustering Analysis")

    @st.cache_data(show_spinner="Aggregating customer data…")
    def make_user_df(_df):
        agg = _df.groupby("User_ID").agg(
            Age_enc        = ("Age_enc",       "first"),
            Occupation     = ("Occupation",     "first"),
            Marital_Status = ("Marital_Status", "first"),
            Avg_Purchase   = ("Purchase",       "mean"),
            Num_Txn        = ("Purchase",       "count"),
        ).reset_index()
        mm = MinMaxScaler()
        agg["Purchase_norm"] = mm.fit_transform(agg[["Avg_Purchase"]])
        return agg

    user_df = make_user_df(df)
    feats   = ["Age_enc", "Occupation", "Marital_Status", "Purchase_norm"]
    X_raw   = user_df[feats].dropna().values
    X       = StandardScaler().fit_transform(X_raw)
    X_list  = X.tolist()   # serialisable for caching

    @st.cache_data(show_spinner="Computing elbow curve…")
    def elbow(Xl):
        Xa = np.array(Xl)
        return [KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xa).inertia_
                for k in range(2, 11)]

    inertias = elbow(X_list)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(2, 11), inertias, "bo-", markersize=8)
    ax.set_title("Elbow Method")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    show(fig)

    k = st.slider("Number of clusters", 2, 8, 4)

    @st.cache_data(show_spinner="Running K-Means…")
    def run_km(Xl, k):
        Xa     = np.array(Xl)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xa)
        coords = PCA(n_components=2).fit_transform(Xa)
        return labels.tolist(), coords.tolist()

    labels_list, pca_list = run_km(X_list, k)
    labels     = np.array(labels_list)
    pca_coords = np.array(pca_list)

    names   = ["💰 Budget Shoppers","🛒 Discount Lovers","⭐ Premium Buyers",
               "🎯 Occasional Spenders","🔥 Power Buyers","🌟 Loyal Customers",
               "💎 VIP Shoppers","🎪 Impulse Buyers"]
    colours = ["#4e79a7","#f28e2b","#e15759","#76b7b2",
               "#59a14f","#edc948","#b07aa1","#ff9da7"]

    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([names[i] for i in counts.index],
           counts.values,
           color=[colours[i] for i in counts.index])
    ax.set_title("Customers per Cluster")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    show(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        m = labels == c
        ax.scatter(pca_coords[m, 0], pca_coords[m, 1],
                   label=names[c], alpha=0.5, s=15, color=colours[c])
    ax.set_title("Customer Segments (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    show(fig)

    tmp = pd.DataFrame(X_raw, columns=feats)
    tmp["Cluster"] = labels
    summary = tmp.groupby("Cluster").mean().round(3)
    summary.index = [names[i] for i in summary.index]
    st.subheader("Cluster Summary")
    st.dataframe(summary)

# ══════════════════════════════════════════════════════════════
# STAGE 5 – ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════════
elif stage == "🔗 Association Rule Mining":
    st.header("Stage 5: Association Rule Mining")

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        st.error("mlxtend not installed. Add `mlxtend` to requirements.txt and redeploy.")
        st.stop()

    @st.cache_data(show_spinner="Running Apriori…")
    def build_rules(_df):
        # One transaction per user → list of unique category strings
        txns = (
            _df.groupby("User_ID")["Product_Category_1"]
               .apply(lambda x: list(x.astype(str).unique()))
               .tolist()
        )
        te     = TransactionEncoder()
        te_arr = te.fit_transform(txns)
        te_df  = pd.DataFrame(te_arr, columns=te.columns_)

        freq  = apriori(te_df, min_support=0.05, use_colnames=True, low_memory=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)

        # Convert frozensets → strings BEFORE returning (makes it serialisable)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

        keep = ["antecedents","consequents","support","confidence","lift"]
        return rules[keep].reset_index(drop=True), int(len(freq))

    rules_df, n_freq = build_rules(df)

    c1, c2 = st.columns(2)
    c1.metric("Frequent Itemsets", n_freq)
    c2.metric("Rules Generated",   len(rules_df))

    st.subheader("Top 15 Rules by Lift")
    top = rules_df.sort_values("lift", ascending=False).head(15).reset_index(drop=True)
    top[["support","confidence","lift"]] = top[["support","confidence","lift"]].round(4)
    st.dataframe(top)

    st.subheader("Support vs Confidence (colour = Lift)")
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(
        rules_df["support"], rules_df["confidence"],
        c=rules_df["lift"], cmap="YlOrRd",
        s=rules_df["lift"] * 20, alpha=0.7
    )
    plt.colorbar(sc, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association Rules – Support vs Confidence")
    plt.tight_layout()
    show(fig)

# ══════════════════════════════════════════════════════════════
# STAGE 6 – ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════
elif stage == "⚠️ Anomaly Detection":
    st.header("Stage 6: Anomaly Detection")

    method   = st.radio("Detection method", ["Z-Score", "IQR"])
    purchase = df["Purchase"].values

    if method == "Z-Score":
        thresh     = st.slider("Z-Score threshold", 2.0, 4.0, 3.0, 0.1)
        is_anomaly = np.abs(stats.zscore(purchase)) > thresh
    else:
        Q1, Q3 = np.percentile(purchase, 25), np.percentile(purchase, 75)
        iqr    = Q3 - Q1
        lo, hi = Q1 - 1.5 * iqr, Q3 + 1.5 * iqr
        is_anomaly = (purchase < lo) | (purchase > hi)
        st.info(f"IQR bounds: ${lo:,.0f} – ${hi:,.0f}")

    anomalies = df[is_anomaly]
    normal    = df[~is_anomaly]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Anomalies Detected", f"{len(anomalies):,}")
    c3.metric("Anomaly Rate",       f"{is_anomaly.mean()*100:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(normal["Purchase"],    bins=60, color="steelblue", alpha=0.7, label="Normal")
    ax.hist(anomalies["Purchase"], bins=30, color="red",       alpha=0.8, label="Anomaly")
    ax.set_title("Purchase Distribution – Normal vs Anomalous")
    ax.set_xlabel("Purchase ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    show(fig)

    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = anomalies["Age"].value_counts().reindex(AGE_ORDER).dropna()
        ax.bar(vc.index, vc.values, color="tomato")
        ax.set_title("Anomalies by Age Group")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        show(fig)
    with cb:
        fig, ax = plt.subplots(figsize=(6, 4))
        vc2 = anomalies["Occupation"].value_counts().head(10)
        ax.bar(vc2.index.astype(str), vc2.values, color="salmon")
        ax.set_title("Anomalies by Occupation (Top 10)")
        ax.set_xlabel("Occupation Code")
        ax.set_ylabel("Count")
        plt.tight_layout()
        show(fig)

    st.subheader("Sample Anomalous Records")
    st.dataframe(
        anomalies[["User_ID","Gender","Age","Occupation",
                   "City_Category","Purchase"]].head(20)
    )

# ══════════════════════════════════════════════════════════════
# STAGE 7 – INSIGHTS & REPORT
# ══════════════════════════════════════════════════════════════
elif stage == "💡 Insights & Report":
    st.header("Stage 7: Key Insights & Business Recommendations")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Customers",   f"{df['User_ID'].nunique():,}")
    c3.metric("Avg Purchase",       f"${df['Purchase'].mean():,.0f}")
    c4.metric("Total Revenue",      f"${df['Purchase'].sum():,.0f}")

    st.markdown("---")

    st.subheader("🔍 Finding 1 – Which Age Group Spends the Most?")
    age_spend = (df.groupby("Age")["Purchase"].mean()
                   .reindex(AGE_ORDER).dropna())
    fig, ax = plt.subplots(figsize=(8, 4))
    age_spend.plot(kind="bar", ax=ax, color="cornflowerblue")
    ax.set_title("Average Purchase by Age Group")
    ax.set_ylabel("Avg Purchase ($)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    show(fig)
    st.success(f"✅ **{age_spend.idxmax()}** has the highest avg spend: **${age_spend.max():,.0f}**")

    st.subheader("🔍 Finding 2 – Popular Product Categories by Gender")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, (g, label, colour) in enumerate([("M","Male","steelblue"),
                                             ("F","Female","lightcoral")]):
        top = df[df["Gender"] == g]["Product_Category_1"].value_counts().head(8)
        axes[i].bar(top.index.astype(str), top.values, color=colour)
        axes[i].set_title(f"Top Categories – {label}")
        axes[i].set_xlabel("Product Category 1")
        axes[i].set_ylabel("Count")
    plt.tight_layout()
    show(fig)

    st.subheader("🔍 Finding 3 – Spending by City Category")
    city_spend = df.groupby("City_Category")["Purchase"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    city_spend.plot(kind="bar", ax=ax, color=["#2196F3","#FF9800","#4CAF50"])
    ax.set_title("Average Purchase by City Category")
    ax.set_ylabel("Avg Purchase ($)")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    show(fig)

    st.markdown("---")
    st.subheader("📋 Strategic Recommendations")
    st.markdown("""
| # | Insight | Recommendation |
|---|---|---|
| 1 | Specific age group dominates spending | Target with premium product bundles |
| 2 | Males buy certain categories more | Gender-specific promotional campaigns |
| 3 | City category affects spend level | Tailor discounts per city type |
| 4 | Association rules reveal frequent combos | Design combo-pack offers |
| 5 | High-spend anomaly users detected | Offer VIP / loyalty programmes |
""")
