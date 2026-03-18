# IDAI1051000467-TejasviKandimalla

## Beyond Discounts: Data-Driven Black Friday Sales Insights

Student’s Full Name: Tejasvi Reddy Kandimalla

Candidate Registration Number:1000467

CRS Name : Artificial Intelligence

Course Name: Data Mining

School Name: Birla Open Minds International School




- **Live Streamlit App:** https://idai1051000467-tejasvikandimalla-oml3t9v4uayucgcanvn2w6.streamlit.app/

---

## 📌 Project Title & Scope

**Mining the Future: Unlocking Business Intelligence with AI**

This project analyses the Black Friday retail sales dataset to uncover customer purchase patterns, segment shoppers into meaningful groups, identify frequently bought product combinations, and detect unusual spending behaviour. The insights are presented through an interactive Streamlit dashboard deployed on Streamlit Cloud.

**Scenario:** Scenario 1 – Beyond Discounts: Data-Driven Black Friday Sales Insights

**Role:** Data Analyst at InsightMart Analytics

**Dataset:** BlackFriday.csv — contains 550,000+ transactions with customer demographics and purchase details.

---

## 🎯 Objectives

- Understand shopping preferences: discover how demographics influence purchases and which product categories dominate sales
- Segment customers effectively using K-Means clustering to identify distinct shopping groups
- Identify cross-selling opportunities using association rule mining to uncover frequent product combinations
- Detect anomalies such as unusually high spenders using Z-Score and IQR methods
- Deploy all insights through an interactive Streamlit dashboard

---

## 📋 Dataset Description

| Column | Description |
|---|---|
| User_ID | Unique customer identifier |
| Product_ID | Unique product identifier |
| Gender | M (Male) / F (Female) |
| Age | Age bracket: 0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+ |
| Occupation | Occupation code (0–20) |
| City_Category | City type: A, B, or C |
| Stay_In_Current_City_Years | Number of years lived in current city (0–4+) |
| Marital_Status | 0 = Single, 1 = Married |
| Product_Category_1 | Primary product category code |
| Product_Category_2 | Secondary product category code (nullable) |
| Product_Category_3 | Tertiary product category code (nullable) |
| Purchase | Purchase amount in dollars |

---

## 🧹 Stage 2: Data Preparation & Preprocessing

### Steps Applied

1. **Missing Values** — `Product_Category_2` and `Product_Category_3` had missing entries. These were filled with `0` to indicate no secondary/tertiary category was purchased.

2. **Duplicate Removal** — All duplicate rows were identified and removed using `drop_duplicates()`.

3. **Gender Encoding** — The `Gender` column was encoded numerically: Male = 0, Female = 1.

4. **Age Group Encoding** — Age bracket strings were mapped to ordered integers:
   - 0-17 → 1, 18-25 → 2, 26-35 → 3, 36-45 → 4, 46-50 → 5, 51-55 → 6, 55+ → 7

5. **City Category Encoding** — Mapped to integers: A = 1, B = 2, C = 3.

6. **Stay Years Encoding** — `Stay_In_Current_City_Years` contained "4+" which was cleaned and converted to integer 4.

7. **Purchase Normalisation** — The `Purchase` column was normalised to [0, 1] using Min-Max scaling so all features are on the same scale for clustering.

---

## 📊 Stage 3: EDA & Visualizations

The following visualisations were created to explore patterns in the dataset:

| Visualisation | Purpose |
|---|---|
| Boxplot – Purchase by Age Group | Understand how spending varies across age groups |
| Histogram – Purchase by Gender | Compare spending distributions between Male and Female customers |
| Bar Charts – Top 10 Product Categories | Identify the most purchased product categories (Cat 1, 2, 3) |
| Scatter Plot – Purchase vs Occupation | Examine whether occupation type correlates with spend |
| Scatter Plot – Purchase vs Years in City | See if time spent in a city affects purchasing behaviour |
| Bar Chart – Avg Purchase per Category | Compare average spend across all product categories |
| Correlation Heatmap | Identify which features have the strongest relationships with Purchase |

### Key EDA Findings

- The **26-35 age group** makes up the largest portion of transactions and highest total spend
- **Male customers** account for the majority of purchases across most product categories
- **Product Category 1** dominates in frequency — categories 1, 5, and 8 are the most purchased
- City Category **B** has the highest number of transactions
- **Occupation** and **City Category** show weak direct correlation with purchase amount individually

---

## 🔵 Stage 4: Clustering Analysis

### Method: K-Means Clustering

**Features used for clustering:**
- Age (encoded)
- Occupation
- Marital Status
- Average Purchase (normalised)

**Process:**
1. Data was aggregated per `User_ID` to get one row per unique customer
2. Features were standardised using `StandardScaler`
3. The **Elbow Method** was used to determine the optimal number of clusters by plotting inertia for K = 2 to 8
4. **K = 4** was selected as the optimal number of clusters
5. **PCA** (Principal Component Analysis) was used to reduce dimensions to 2D for visualisation

### Cluster Labels

| Cluster | Label | Description |
|---|---|---|
| 0 | 💰 Budget Shoppers | Low spenders, younger age group |
| 1 | 🛒 Discount Lovers | Mid-range spenders who buy frequently across categories |
| 2 | ⭐ Premium Buyers | High spenders, likely working professionals |
| 3 | 🎯 Occasional Spenders | Infrequent buyers with moderate spend |

### Tools Used
- `sklearn.cluster.KMeans`
- `sklearn.decomposition.PCA`
- `sklearn.preprocessing.StandardScaler`

---

## 🔗 Stage 5: Association Rule Mining

### Method: Custom Pair-Based Apriori (Pure NumPy/Pandas)

Association rules were generated by analysing which **product categories** customers frequently purchase together within the same shopping session.

**Parameters used:**
- Minimum Support: 5% (a category pair must appear in at least 5% of all user baskets)
- Minimum Confidence: 30%

**Metrics calculated for each rule:**

| Metric | Description |
|---|---|
| Support | How frequently the pair appears across all users |
| Confidence | How often the rule is correct (antecedent → consequent) |
| Lift | How much more likely the pair is bought together vs. independently (Lift > 1 = positive association) |

### Example Rules Found
- If a customer buys **Category 1**, they are also likely to buy **Category 5**
- If a customer buys **Category 5**, they are also likely to buy **Category 8**
- High lift values indicate strong cross-selling opportunities between product pairs

### Business Application
Retailers can use these rules to:
- Design **combo-pack promotions**
- Place frequently paired products near each other in store or online
- Create **targeted bundle discount** campaigns

---

## ⚠️ Stage 6: Anomaly Detection

### Methods Used: Z-Score & IQR

**Z-Score Method:**
- Calculates how many standard deviations a transaction is from the mean
- Transactions with |Z| > 3.0 are flagged as anomalies

**IQR Method:**
- Calculates the Interquartile Range (Q3 − Q1)
- Any transaction below Q1 − 1.5×IQR or above Q3 + 1.5×IQR is flagged

### Findings

- Approximately **2–4%** of transactions are flagged as anomalous depending on the method chosen
- Anomalous spenders tend to cluster in the **26-35 and 36-45** age groups
- Certain occupation codes appear disproportionately in the anomaly group
- These high-value outliers represent potential **VIP customers** for loyalty programmes

---

## 💡 Stage 7: Key Insights & Recommendations

### Insights

1. The **26-35 age group** is the most active and highest-spending demographic
2. **Male customers** are responsible for the majority of Black Friday transactions
3. **City Category B** has the most transactions; City A customers tend to spend more per transaction
4. **Product Categories 1, 5, and 8** are the most frequently purchased
5. A small percentage of customers are extreme high-spenders — these are high-value targets for retention

### Strategic Recommendations

| # | Insight | Recommendation |
|---|---|---|
| 1 | 26-35 age group dominates spending | Target this group with premium product bundles and early access deals |
| 2 | Males buy far more than females | Create gender-specific promotional campaigns for both segments |
| 3 | City A customers have higher per-transaction spend | Offer premium/exclusive products in City A stores |
| 4 | Categories 1+5, 5+8 are frequently paired | Design combo-pack offers and cross-category discounts |
| 5 | Anomaly users spend exceptionally high | Launch a VIP loyalty programme targeting this segment |

---

## 🚀 Stage 8: Deployment on Streamlit Cloud

### App Features

The Streamlit dashboard includes 7 interactive stages:

| Stage | Feature |
|---|---|
| 📌 Project Scope | Objectives and dataset description |
| 🧹 Data Overview | Preprocessing steps, null values, descriptive statistics |
| 📊 EDA | 5 interactive tabs with charts and heatmap |
| 🔵 Clustering | Elbow method, K slider, PCA scatter, cluster summary |
| 🔗 Association Rules | Rules table, support vs confidence scatter, lift bar chart |
| ⚠️ Anomaly Detection | Toggle Z-Score/IQR, distribution chart, demographic breakdown |
| 💡 Insights | KPI metrics, findings charts, recommendations table |

### How to Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/YourRepository.git
cd YourRepository

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure BlackFriday.csv is in the same folder as app.py

# 4. Run the app
streamlit run app.py
```

### Deployment Steps

1. Push all files to GitHub (app.py, requirements.txt, BlackFriday.csv)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in
3. Click **New App** → select your repository → set main file to `app.py`
4. Click **Deploy**

---

## 📁 Repository Structure
```
YourRepository/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── BlackFriday.csv         # Dataset
└── README.md               # This file
```

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| streamlit | latest | Web dashboard framework |
| pandas | latest | Data manipulation and analysis |
| numpy | latest | Numerical computations |
| matplotlib | latest | Data visualisation |
| scikit-learn | latest | K-Means clustering and PCA |

---

## 📚 References

- Analytics Vidhya – Market Basket Analysis: https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-market-basket-analysis/
- Neptune.ai – K-Means Clustering: https://neptune.ai/blog/k-means-clustering
- Scikit-learn Documentation: https://scikit-learn.org
- Streamlit Documentation: https://docs.streamlit.io
- DataCamp – Anomaly Detection in Python: https://www.datacamp.com/courses/anomaly-detection-in-python
- Seaborn – Statistical Plots: https://seaborn.pydata.org
- Data to Viz: https://www.data-to-viz.com
