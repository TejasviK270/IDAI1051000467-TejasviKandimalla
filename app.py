import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Sales Insights",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Beyond Discounts: Black Friday Sales Insights")
st.markdown("**InsightMart Analytics** – Customer Purchase Pattern Analysis")
st.markdown("---")

AGE_ORDER = ["0-17","18-25","26-35","36-45","46-50","51-55","55+"]

# ── Load & clean (cached) ────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv("BlackFriday.csv")
    df["Product_Category_2"] = df["Product_Category_2"].fillna(0).astype(int)
    df["Product_Category_3"] = df["Product_Category_3"].fillna(0).astype(int)
    df = df.drop_duplicates().reset_index(drop=True)

    df["Gender_enc"] = (df["Gender"] == "F").astype(int)

    age_map = {"0-17":1,"18-25":2,"26-35":3,"36-45":4,"46-50":5,"51-55":6,"55+":7}
    df["Age_enc"] = df["Age"].map(age_map).fillna(3).astype(int)

    df["City_enc"] = df["City_Category"].map({"A":1,"B":2,"C":3}).fillna(2).astype(int)

    df["Stay_enc"] = (
        df["Stay_In_Current_City_Years"]
        .astype(str).str.replace("+","",regex=False)
        .astype(int)
    )

    p_min = df["Purchase"].min()
    p_max = df["Purchase"].max()
    df["Purchase_norm"] = (df["Purchase"] - p_min) / (p_max - p_min + 1e-9)

    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌  BlackFriday.csv not found — place it in the same folder as app.py.")
    st.stop()
except Exception as exc:
    st.error(f"❌  Could not load data: {exc}")
    st.stop()

# ── Helper ───────────────────────────────────────────────────
def show(fig):
    st.pyplot(fig)
    plt.close(fig)

# ── Sidebar ──────────────────────────────────────────────────
stage = st.sidebar.radio(
    "📋 Navigate",
    [
        "📌 Project Scope",
        "🧹 Data Overview",
        "📊 EDA & Visualizations",
        "🔵 Clustering Analysis",
        "🔗 Association Rule Mining",
        "⚠️  Anomaly Detection",
        "💡 Insights & Report",
    ],
)

# ════════════════════════════════════════════════════════════
# STAGE 1 – PROJECT SCOPE
# ════════════════════════════════════════════════════════════
if stage == "📌 Project Scope":
    st.header("Stage 1: Project Scope Definition")
    st.markdown(f"""
**Dataset loaded:** `{df.shape[0]:,}` rows × `{df.shape[1]}` columns

### 🎯 Objectives
| # | Goal |
|---|------|
| 1 | Understand shopping preferences – which categories & demographics dominate |
| 2 | Segment customers using K-Means clustering |
| 3 | Find frequent product combinations with association rule mining |
| 4 | Detect anomalous high-spend transactions |
| 5 | Present all findings in this interactive Streamlit dashboard |

### 📋 Dataset Columns
| Column | Description |
|---|---|
| User_ID | Unique customer ID |
| Product_ID | Unique product ID |
| Gender | M / F |
| Age | Age bracket (0-17 … 55+) |
| Occupation | Occupation code 0–20 |
| City_Category | City type A / B / C |
| Stay_In_Current_City_Years | 0 – 4+ years |
| Marital_Status | 0 = Single, 1 = Married |
| Product_Category_1/2/3 | Category codes |
| Purchase | Purchase amount ($) |
""")

# ════════════════════════════════════════════════════════════
# STAGE 2 – DATA OVERVIEW
# ════════════════════════════════════════════════════════════
elif stage == "🧹 Data Overview":
    st.header("Stage 2: Data Cleaning & Preprocessing")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sample Rows")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("Null Values After Cleaning")
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column","Nulls"]
        st.dataframe(mv)

    st.subheader("Steps Applied")
    st.markdown("""
- ✅ Product_Category_2 & 3 – nulls filled with **0**
- ✅ Duplicate rows removed
- ✅ Gender → 0 (Male) / 1 (Female)
- ✅ Age bracket → ordered integer 1–7
- ✅ Stay_In_Current_City_Years → integer (4+ → 4)
- ✅ Purchase → Min-Max normalised to [0, 1]
""")
    st.dataframe(
        df[["Gender","Gender_enc","Age","Age_enc","Purchase","Purchase_norm"]].head(10)
    )
    st.subheader("Descriptive Statistics")
    st.dataframe(df[["Age_enc","Occupation","Marital_Status","Purchase"]].describe().round(2))

# ════════════════════════════════════════════════════════════
# STAGE 3 – EDA
# ════════════════════════════════════════════════════════════
elif stage == "📊 EDA & Visualizations":
    st.header("Stage 3: Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Age & Gender","Product Categories","Occupation","Avg Purchase","Heatmap"
    ])

    with tab1:
        st.subheader("Purchase by Age Group & Gender")
        valid = [a for a in AGE_ORDER if a in df["Age"].unique()]
        data  = [df.loc[df["Age"]==a,"Purchase"].values for a in valid]
        fig, axes = plt.subplots(1, 2, figsize=(13,5))
        axes[0].boxplot(data, labels=valid, showfliers=False)
        axes[0].set_title("Purchase by Age Group")
        axes[0].set_xlabel("Age Group"); axes[0].set_ylabel("Purchase ($)")
        axes[0].tick_params(axis="x", rotation=30)
        for gval, glabel, gcol in [("M","Male","steelblue"),("F","Female","lightcoral")]:
            vals = df.loc[df["Gender"]==gval,"Purchase"]
            axes[1].hist(vals, bins=40, alpha=0.6, color=gcol, label=glabel)
        axes[1].set_title("Purchase Distribution by Gender")
        axes[1].set_xlabel("Purchase ($)"); axes[1].set_ylabel("Count")
        axes[1].legend()
        plt.tight_layout(); show(fig)

    with tab2:
        st.subheader("Top 10 Product Categories")
        fig, axes = plt.subplots(1, 3, figsize=(17,5))
        for i, col in enumerate(["Product_Category_1","Product_Category_2","Product_Category_3"]):
            vc = df[df[col]>0][col].value_counts().head(10)
            axes[i].bar(vc.index.astype(str), vc.values, color="steelblue")
            axes[i].set_title(f"Top 10 – {col}")
            axes[i].set_xlabel("Category"); axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=45)
        plt.tight_layout(); show(fig)

    with tab3:
        st.subheader("Average Purchase vs Occupation & City Stay")
        fig, axes = plt.subplots(1, 2, figsize=(13,5))
        occ = df.groupby("Occupation")["Purchase"].mean()
        axes[0].scatter(occ.index, occ.values, color="coral", s=80)
        axes[0].set_title("Avg Purchase vs Occupation")
        axes[0].set_xlabel("Occupation Code"); axes[0].set_ylabel("Avg Purchase ($)")
        stay = df.groupby("Stay_enc")["Purchase"].mean()
        axes[1].scatter(stay.index, stay.values, color="mediumseagreen", s=80)
        axes[1].set_title("Avg Purchase vs Years in City")
        axes[1].set_xlabel("Years in City"); axes[1].set_ylabel("Avg Purchase ($)")
        plt.tight_layout(); show(fig)

    with tab4:
        st.subheader("Average Purchase per Product Category 1")
        avg = df.groupby("Product_Category_1")["Purchase"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12,5))
        ax.bar(avg.index.astype(str), avg.values, color="mediumpurple")
        ax.set_title("Avg Purchase per Category 1")
        ax.set_xlabel("Category"); ax.set_ylabel("Avg Purchase ($)")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout(); show(fig)

    with tab5:
        st.subheader("Correlation Heatmap")
        cols = ["Age_enc","Gender_enc","Occupation","City_enc","Stay_enc",
                "Marital_Status","Product_Category_1","Product_Category_2",
                "Product_Category_3","Purchase"]
        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout(); show(fig)

# ════════════════════════════════════════════════════════════
# STAGE 4 – CLUSTERING
# ════════════════════════════════════════════════════════════
elif stage == "🔵 Clustering Analysis":
    st.header("Stage 4: K-Means Clustering")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    @st.cache_data(show_spinner="Aggregating users…")
    def get_user_matrix(_df):
        agg = _df.groupby("User_ID").agg(
            Age_enc        =("Age_enc",       "first"),
            Occupation     =("Occupation",    "first"),
            Marital_Status =("Marital_Status","first"),
            Purchase_norm  =("Purchase_norm", "mean"),
        ).reset_index(drop=True)
        agg = agg.dropna().reset_index(drop=True)
        sc  = StandardScaler()
        X   = sc.fit_transform(agg.values)
        return agg, X.tolist()

    user_df, X_list = get_user_matrix(df)
    X = np.array(X_list)

    @st.cache_data(show_spinner="Elbow curve…")
    def get_elbow(Xl):
        Xa = np.array(Xl)
        return [KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xa).inertia_
                for k in range(2, 9)]

    inertias = get_elbow(X_list)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(range(2,9), inertias, "bo-", markersize=8)
    ax.set_title("Elbow Method"); ax.set_xlabel("K"); ax.set_ylabel("Inertia")
    ax.grid(alpha=0.3); plt.tight_layout(); show(fig)

    k = st.slider("Choose number of clusters", 2, 7, 4)

    @st.cache_data(show_spinner="Running K-Means + PCA…")
    def run_kmeans(Xl, k):
        Xa     = np.array(Xl)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xa)
        coords = PCA(n_components=2, random_state=42).fit_transform(Xa)
        return labels.tolist(), coords.tolist()

    labels_list, coords_list = run_kmeans(X_list, k)
    labels = np.array(labels_list)
    coords = np.array(coords_list)

    NAMES  = ["💰 Budget","🛒 Discount","⭐ Premium","🎯 Occasional",
              "🔥 Power","🌟 Loyal","💎 VIP"]
    COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1"]

    # Bar chart
    vc = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar([NAMES[i] for i in vc.index], vc.values, color=[COLORS[i] for i in vc.index])
    ax.set_title("Customers per Cluster"); ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout(); show(fig)

    # Scatter
    fig, ax = plt.subplots(figsize=(9,6))
    for c in range(k):
        m = labels == c
        ax.scatter(coords[m,0], coords[m,1], label=NAMES[c],
                   alpha=0.5, s=12, color=COLORS[c])
    ax.set_title("Customer Segments (PCA 2D)")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(fontsize=8); plt.tight_layout(); show(fig)

    # Summary
    tmp = user_df.copy()
    tmp["Cluster"] = labels
    summary = tmp.groupby("Cluster").mean().round(3)
    summary.index = [NAMES[i] for i in summary.index]
    st.subheader("Cluster Averages")
    st.dataframe(summary)

# ════════════════════════════════════════════════════════════
# STAGE 5 – ASSOCIATION RULE MINING  (pure numpy – no mlxtend)
# ════════════════════════════════════════════════════════════
elif stage == "🔗 Association Rule Mining":
    st.header("Stage 5: Association Rule Mining")
    st.markdown("Discovering which **product categories** are frequently bought together.")

    @st.cache_data(show_spinner="Computing association rules…")
    def compute_rules(_df, min_support=0.05, min_confidence=0.3):
        # Build user × category presence matrix using only Product_Category_1
        user_cats = (
            _df.groupby("User_ID")["Product_Category_1"]
               .apply(lambda x: list(x.astype(int).unique()))
        )
        all_cats = sorted(_df["Product_Category_1"].unique().astype(int))
        n_users  = len(user_cats)

        # Count single-item and pair frequencies
        single_count = {c: 0 for c in all_cats}
        pair_count   = {}

        for cats in user_cats:
            cats_set = set(cats)
            for c in cats_set:
                single_count[c] += 1
            cats_sorted = sorted(cats_set)
            for i in range(len(cats_sorted)):
                for j in range(i+1, len(cats_sorted)):
                    key = (cats_sorted[i], cats_sorted[j])
                    pair_count[key] = pair_count.get(key, 0) + 1

        # Generate rules
        rows = []
        for (a, b), cnt in pair_count.items():
            sup = cnt / n_users
            if sup < min_support:
                continue
            conf_ab = sup / (single_count[a] / n_users)
            conf_ba = sup / (single_count[b] / n_users)
            lift    = sup / ((single_count[a] / n_users) * (single_count[b] / n_users))
            if conf_ab >= min_confidence:
                rows.append({"antecedents": str(a), "consequents": str(b),
                             "support": round(sup,4),
                             "confidence": round(conf_ab,4),
                             "lift": round(lift,4)})
            if conf_ba >= min_confidence:
                rows.append({"antecedents": str(b), "consequents": str(a),
                             "support": round(sup,4),
                             "confidence": round(conf_ba,4),
                             "lift": round(lift,4)})

        return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)

    rules_df = compute_rules(df)

    col1, col2 = st.columns(2)
    col1.metric("Rules Generated", len(rules_df))
    col2.metric("Min Support Used", "5%")

    if rules_df.empty:
        st.warning("No rules found – try lowering the support threshold.")
    else:
        st.subheader("Top 20 Rules by Lift")
        st.dataframe(rules_df.head(20))

        st.subheader("Support vs Confidence (colour = Lift)")
        fig, ax = plt.subplots(figsize=(9,5))
        sc = ax.scatter(
            rules_df["support"], rules_df["confidence"],
            c=rules_df["lift"], cmap="YlOrRd", s=60, alpha=0.8
        )
        plt.colorbar(sc, ax=ax, label="Lift")
        ax.set_xlabel("Support"); ax.set_ylabel("Confidence")
        ax.set_title("Association Rules – Support vs Confidence")
        plt.tight_layout(); show(fig)

        st.subheader("Top 10 Rules – Lift Bar Chart")
        top10 = rules_df.head(10)
        labels_bar = [f"{r['antecedents']} → {r['consequents']}" for _, r in top10.iterrows()]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(labels_bar[::-1], top10["lift"].values[::-1], color="teal")
        ax.set_xlabel("Lift"); ax.set_title("Top 10 Association Rules by Lift")
        plt.tight_layout(); show(fig)

# ════════════════════════════════════════════════════════════
# STAGE 6 – ANOMALY DETECTION  (pure numpy – no scipy)
# ════════════════════════════════════════════════════════════
elif stage == "⚠️  Anomaly Detection":
    st.header("Stage 6: Anomaly Detection")

    method = st.radio("Detection method", ["Z-Score", "IQR"])
    purchase = df["Purchase"].values.astype(float)

    if method == "Z-Score":
        thresh = st.slider("Z-Score threshold", 2.0, 4.0, 3.0, 0.1)
        mean, std = purchase.mean(), purchase.std()
        is_anomaly = np.abs((purchase - mean) / (std + 1e-9)) > thresh
    else:
        Q1  = np.percentile(purchase, 25)
        Q3  = np.percentile(purchase, 75)
        iqr = Q3 - Q1
        lo, hi = Q1 - 1.5*iqr, Q3 + 1.5*iqr
        is_anomaly = (purchase < lo) | (purchase > hi)
        st.info(f"IQR bounds: ${lo:,.0f} – ${hi:,.0f}")

    anomalies = df[is_anomaly]
    normal    = df[~is_anomaly]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Anomalies Detected", f"{len(anomalies):,}")
    c3.metric("Anomaly Rate",       f"{is_anomaly.mean()*100:.2f}%")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.hist(normal["Purchase"],    bins=60, color="steelblue", alpha=0.7, label="Normal")
    ax.hist(anomalies["Purchase"], bins=30, color="red",       alpha=0.8, label="Anomaly")
    ax.set_title("Purchase Distribution – Normal vs Anomalous")
    ax.set_xlabel("Purchase ($)"); ax.set_ylabel("Frequency"); ax.legend()
    plt.tight_layout(); show(fig)

    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(6,4))
        vc = anomalies["Age"].value_counts().reindex(AGE_ORDER).dropna()
        ax.bar(vc.index, vc.values, color="tomato")
        ax.set_title("Anomalies by Age Group")
        ax.set_xlabel("Age"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout(); show(fig)
    with cb:
        fig, ax = plt.subplots(figsize=(6,4))
        vc2 = anomalies["Occupation"].value_counts().head(10)
        ax.bar(vc2.index.astype(str), vc2.values, color="salmon")
        ax.set_title("Anomalies by Occupation (Top 10)")
        ax.set_xlabel("Occupation"); ax.set_ylabel("Count")
        plt.tight_layout(); show(fig)

    st.subheader("Sample Anomalous Records")
    st.dataframe(anomalies[["User_ID","Gender","Age","Occupation",
                             "City_Category","Purchase"]].head(20))

# ════════════════════════════════════════════════════════════
# STAGE 7 – INSIGHTS & REPORT
# ════════════════════════════════════════════════════════════
elif stage == "💡 Insights & Report":
    st.header("Stage 7: Key Insights & Business Recommendations")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Customers",   f"{df['User_ID'].nunique():,}")
    c3.metric("Avg Purchase",       f"${df['Purchase'].mean():,.0f}")
    c4.metric("Total Revenue",      f"${df['Purchase'].sum():,.0f}")
    st.markdown("---")

    # Finding 1
    st.subheader("🔍 Finding 1 – Age Group with Highest Spend")
    age_spend = df.groupby("Age")["Purchase"].mean().reindex(AGE_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(age_spend.index, age_spend.values, color="cornflowerblue")
    ax.set_title("Avg Purchase by Age Group")
    ax.set_ylabel("Avg Purchase ($)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(); show(fig)
    st.success(f"✅ **{age_spend.idxmax()}** spends the most on average: **${age_spend.max():,.0f}**")

    # Finding 2
    st.subheader("🔍 Finding 2 – Popular Categories by Gender")
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    for i, (g, label, colour) in enumerate([("M","Male","steelblue"),("F","Female","lightcoral")]):
        top = df[df["Gender"]==g]["Product_Category_1"].value_counts().head(8)
        axes[i].bar(top.index.astype(str), top.values, color=colour)
        axes[i].set_title(f"Top Categories – {label}")
        axes[i].set_xlabel("Category"); axes[i].set_ylabel("Count")
    plt.tight_layout(); show(fig)

    # Finding 3
    st.subheader("🔍 Finding 3 – Spending by City Category")
    city = df.groupby("City_Category")["Purchase"].mean()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(city.index, city.values, color=["#2196F3","#FF9800","#4CAF50"])
    ax.set_title("Avg Purchase by City Category")
    ax.set_ylabel("Avg Purchase ($)")
    plt.tight_layout(); show(fig)

    st.markdown("---")
    st.subheader("📋 Strategic Recommendations")
    st.markdown("""
| # | Insight | Recommendation |
|---|---|---|
| 1 | One age group dominates spending | Target them with premium bundles |
| 2 | Males buy specific categories more | Gender-specific promotions |
| 3 | City category affects spend level | Tailor discounts per city type |
| 4 | Frequent category pairs found | Design combo-pack offers |
| 5 | High-spend anomaly users exist | Launch a VIP / loyalty programme |
""")
