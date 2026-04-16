# CS6502 – Applied Big Data & Visualisation
## Netflix Content Classification using Apache Spark & SparkML
 
**Module:** CS6502 – Applied Big Data & Visualisation  
**Platform:** Databricks Community Edition  
**Spark Version:** 4.1.0  
**Language:** Python (PySpark)
 
---
 
## Project Overview
 
This project implements a complete large-scale machine learning pipeline using Apache Spark to classify Netflix titles as **Movies** or **TV Shows**. The pipeline covers data ingestion via SparkSQL, exploratory data analysis (EDA), feature engineering, training and evaluation of five SparkML classifiers, and Spark-level performance optimisations.
 
**Task:** Binary Classification — Movie vs. TV Show  
**Dataset:** Netflix Titles (Kaggle) — 8,791 records, 10 features  
**Best Result:** Logistic Regression & Tuned Random Forest — Accuracy 0.9976, AUC-ROC 0.9999
 
---
 
## Project Files
 
```
├── net-final 1.ipynb          # Main Spark notebook (final version — use this)
├── net-final 1.html           # Exported HTML version with all rendered outputs and figures
├── netflix-project (4).ipynb  # Earlier development notebook (reference only)
├── Big_Data_Report 1 .docx    # Final written report (CS6502 submission)
├── Netflix_CS6502_Report.docx # Refined final report (polished version)
└── README.md                  # This file
```
 
> **Note:** Use `net-final 1.ipynb` as the primary code submission. The earlier notebook `netflix-project (4).ipynb` is an intermediate version retained for reference. The HTML export (`net-final 1.html`) contains all rendered visualisations and can be viewed in any web browser without running the notebook.
 
---
 
## Requirements
 
### Platform
- **Databricks Community Edition** (free tier) — [sign up here](https://community.cloud.databricks.com/)
- Apache Spark **4.1.0** (pre-installed on Databricks)
- Python **3.12** (pre-installed on Databricks)
 
### Python Libraries (all pre-installed on Databricks)
 
| Library | Purpose |
|---|---|
| `pyspark.sql` | SparkSQL, DataFrames, data ingestion |
| `pyspark.ml.feature` | StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler |
| `pyspark.ml.classification` | LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, NaiveBayes, GBTClassifier |
| `pyspark.ml.evaluation` | MulticlassClassificationEvaluator, BinaryClassificationEvaluator |
| `pyspark.ml.tuning` | CrossValidator, ParamGridBuilder |
| `pyspark.ml` | Pipeline |
| `matplotlib` | Visualisation (pie charts, bar charts, histograms, ROC curves, confusion matrices) |
| `numpy` | Numerical operations |
| `sklearn.metrics` | roc_curve, auc (for ROC curve plotting only) |
 
---
 
## Dataset Setup
 
1. Download the Netflix Titles dataset from Kaggle:  
   [https://www.kaggle.com/datasets/shivamb/netflix-shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
 
2. Upload the CSV file to your Databricks workspace:
   - Go to **Data** → **Add Data** → **Upload File**
   - Upload `netflix_titles.csv`
 
3. Register the table in Unity Catalog or DBFS. The notebook loads it as:
   ```python
   df_raw = spark.table("`netflix-project`.default.netflix_1")
   ```
   If using DBFS, replace this line with:
   ```python
   df_raw = spark.read.csv("/FileStore/netflix_titles.csv", header=True, inferSchema=True)
   ```
 
---
 
## How to Run
 
### Step 1 — Import the Notebook
 
1. Log in to [Databricks Community Edition](https://community.cloud.databricks.com/)
2. Go to **Workspace** → click your username → **Import**
3. Select `net-final 1.ipynb` and click **Import**
 
### Step 2 — Attach a Cluster
 
1. Go to **Compute** → **Create Cluster**
2. Use the default Single Node configuration (Databricks Runtime 14.x or higher)
3. Click **Create** and wait for the cluster to start
4. Open the imported notebook and attach it to the cluster using the dropdown at the top
 
### Step 3 — Configure the Dataset (if needed)
 
If you have not loaded the dataset into Unity Catalog, update **Cell 1** as described in the Dataset Setup section above.
 
### Step 4 — Run All Cells
 
Click **Run All** (`Ctrl+Shift+Enter` or top menu → **Run** → **Run All**).
 
Cells execute sequentially. Do not skip cells — each step depends on the outputs of the previous one.
 
---
 
## Notebook Structure
 
The notebook (`net-final 1.ipynb`) is organised into the following sections, all implemented as code cells:
 
| Cell(s) | Section | Description |
|---|---|---|
| 0 | **Imports & Setup** | Imports all PySpark and Python libraries; verifies Spark session |
| 1 | **Data Ingestion** | Loads dataset from Unity Catalog; prints schema and row count |
| 2 | **SparkSQL Overview** | Registers temp view; queries row counts, content types, year range |
| 3 | **Missing Value Audit** | Counts null/empty values per column using DataFrame API |
| 4 | **Cleaning Pipeline** | Deduplication, null imputation, date parsing, feature extraction |
| 5 | **Null Verification** | SparkSQL query to confirm zero nulls in critical columns post-cleaning |
| 6 | **Summary Statistics** | describe() on numerical columns; per-type aggregations via SparkSQL |
| 7–21 | **EDA & Visualisation** | Content type distribution, top countries, growth trends, ratings, genres, monthly additions |
| 22 | **Correlation Matrix** | Pearson correlation heatmap of numerical features |
| 23 | **Cache Clear** | Manual cache/variable cleanup before ML pipeline |
| 24 | **Encoding** | StringIndexer for type/rating/country; OneHotEncoder for rating |
| 25 | **Feature Scaling** | Manual Z-score standardisation using Spark SQL aggregations |
| 26 | **Feature Assembly** | VectorAssembler combines scaled numericals + OHE features (dim=17) |
| 27 | **Train/Test Split** | 80:20 random split with seed=42 (train: 7,097 / test: 1,692) |
| 28–29 | **Evaluator Setup** | Multiclass and binary evaluators; confusion matrix helper function |
| 30–31 | **Logistic Regression** | Train, predict, evaluate; plot confusion matrix |
| 32–33 | **Random Forest** | 100 trees, depth 10; train, predict, evaluate; confusion matrix |
| 34–35 | **Decision Tree** | Depth 10; train, predict, evaluate; confusion matrix |
| 36 | **Naive Bayes** | Non-negative feature assembly; train, predict, evaluate |
| 37–38 | **GBT** | 50 iterations, depth 5; train, predict, evaluate; confusion matrix |
| 39–40 | **Results Comparison** | Summary table + grouped bar chart across all 5 models |
| 41 | **Feature Importance** | Extract and visualise RF feature importances |
| 42–43 | **Spark Optimisations** | coalesce(4), repartition(4), column pruning |
| 44–49 | **Hyperparameter Tuning** | Manual grid search over 18 RF configurations; best params; tuning chart |
| 50–53 | **Prediction Generation** | Generate predictions for all models; Naive Bayes feature rebuild |
| 54 | **ROC Curve** | ROC curve plotted using sklearn + Spark probabilities |
| 55–56 | **Tuned RF Final Eval** | Confusion matrix and summary for best tuned model |
 
---
 
## Key Results
 
| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.9976 | 0.9976 | **0.9999** |
| Random Forest | 0.9970 | 0.9970 | 0.9995 |
| Decision Tree | 0.9970 | 0.9970 | 0.9976 |
| Naive Bayes | 0.9959 | 0.9959 | 0.9999 |
| GBT | 0.9970 | 0.9970 | 0.9977 |
| **Tuned RF (Grid CV)** | **0.9976** | **0.9976** | 0.9995 |
 
**Best Hyperparameters (Tuned RF):** `numTrees=50`, `maxDepth=5`, `minInstancesPerNode=5`
 
**Top Feature Importance:** `duration_num (scaled)` — **93.4%**
 
---
 
## Important Notes
 
### Databricks Serverless Limitations
The notebook was developed on Databricks Serverless compute, which has the following restrictions relevant to the code:
 
- `spark.sparkContext.cache()` and RDD-level operations are **not supported** — all caching uses DataFrame-level `coalesce()` and `repartition()` instead.
- `CrossValidator` with full k-fold cross-validation triggers RDD operations in some configurations — the tuning section uses a manual Python loop over parameter combinations as a workaround.
- Spark version displayed as **4.1.0** (Databricks Connect runtime).
 
### GenAI Usage Disclosure
In accordance with CS6502 academic integrity requirements: Claude (Anthropic) was used to assist with report writing, refinement, and README generation. All code in the notebook was written and executed by the project group. Where AI-assisted text was used, it has been reviewed, edited, and verified by group members.
 
---
 
## Authors
 
| Name | Student ID |
|---|---|
| Vivek Reddy Kesavarapu | 25269933 |
| Sreenidhi Jorepally | 25339281 |
| Christopher Thomas | 21338485 |
| Conor O'Dwyer | 21315442 |
| Ganga Bhavani Chintalapalli | 25303031 |
| Yaswitha Vasipalli | 25290703 |
 
---
 
## References
 
- Zaharia, M. et al. (2016) 'Apache Spark: A Unified Engine for Big Data Processing', *Communications of the ACM*, 59(11), pp. 56–65.
- Bansal, N. (2022) *Netflix Movies and TV Shows Dataset*. Kaggle. Available at: https://www.kaggle.com/datasets/shivamb/netflix-shows
- Databricks (2024) *Apache Spark on Databricks Documentation*. Available at: https://docs.databricks.com/en/spark/index.html
Databricks - Sign In
 
Databricks - Sign In
 
