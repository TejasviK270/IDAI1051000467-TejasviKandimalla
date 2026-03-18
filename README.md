# IDAI1051000467-TejasviKandimalla

**Project Scope (Stage 1)**

This project analyzes a large-scale Black Friday retail dataset to uncover customer shopping patterns. As a Data Analyst at InsightMart Analytics, the goal is to provide actionable intelligence to improve customer engagement and optimize retail resources.

**Objectives:**

Identify core shopping behaviors and preferences.

Segment customers into distinct clusters (e.g., "Discount Lovers" vs. "Premium Buyers").

Discover product associations for cross-selling opportunities.

Detect anomalies in spending, such as unusually large transactions.

Deploy an interactive Streamlit dashboard for stakeholder visualization.



**Data Preparation & Preprocessing (Stage 2)**

The raw dataset was cleaned and transformed to ensure high-quality mining results:


Missing Values: Handled null entries in Product_Category_2 and Product_Category_3.


Encoding: Converted Gender (Male=0, Female=1) and Age groups into numerical formats.


Normalization: Scaled purchase amounts to ensure features are on a consistent scale.


Deduplication: Removed irrelevant or duplicate records to maintain data integrity.



**Exploratory Data Analysis - EDA (Stage 3)**

Key insights derived from the initial data exploration include:


Spending Distributions: Analysis of purchase amounts across different Age and Gender demographics.


Popularity Trends: Identification of dominant product categories.


Correlations: Heatmaps showing the relationship between features like Occupation and Purchase behavior.




**Advanced Analytics (Stages 4, 5, & 6)**


**Clustering Analysis**

Algorithm: K-Means Clustering.

Method: Used the Elbow Method to determine the optimal number of customer segments.


**Association Rule Mining**

Algorithm: Apriori Algorithm.

Findings: Identified frequent product combinations based on Support, Confidence, and Lift metrics.

Anomaly Detection

Method: Used statistical methods (Z-score/IQR) to isolate outliers.

Insights: Detected "Bulk Purchasers" whose spending patterns deviate significantly from the average.




**Deployment (Stage 8)**

The project is deployed as an interactive web application using Streamlit Cloud.


Functionality: Users can interactively explore demand patterns, view customer clusters, and visualize anomalies.


Live App Link: [Paste your Streamlit Cloud Link Here]
