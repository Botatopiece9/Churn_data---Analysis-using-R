# Customer Churn Prediction — Logistic Regression in R

An end-to-end machine learning pipeline in R for predicting customer churn. Covers data cleaning, missing value imputation, logistic regression with cross-validation, and model evaluation using ROC and Precision-Recall curves.

![Pipeline Overview](pipeline_overview.png)

---

## Overview

Customer churn prediction is a classic binary classification problem in business analytics. This notebook takes raw, messy customer data and walks through every step from cleaning to a trained, evaluated model — producing a final report and publication-ready evaluation plots.

---

## Pipeline

```
raw_churn_data.txt
       │
       ▼
 1. Data Cleaning        — handle blanks, "unknown", type conversion, case fixing
       │
       ▼
 2. Imputation           — median for numerics, mode for categoricals
       │
       ▼
 3. Feature Engineering  — factorize categoricals, drop CustomerID
       │
       ▼
 4. Train/Test Split     — 80/20 stratified split
       │
       ▼
 5. Logistic Regression  — 5-fold cross-validation, optimized for AUC-ROC
       │
       ▼
 6. Evaluation           — confusion matrix, ROC curve, Precision-Recall curve
       │
       ▼
cleaned_churn_data.csv · roc_curve.png · pr_curve.png · churn_report.Rmd
```

---

## Features

- Replaces blank strings and `"unknown"` values with `NA` before any processing
- Converts columns to correct types (numeric, integer, factor)
- Fixes case inconsistencies in categorical columns (`toTitleCase`)
- Flags and removes negative `TotalCharges` values as invalid
- Imputes missing numerics with **median**, missing categoricals with **mode**
- Stratified 80/20 train/test split using `caret::createDataPartition`
- Logistic regression with **5-fold cross-validation**, optimized for ROC-AUC
- Generates confusion matrix, ROC curve with AUC annotation, and PR curve with AUC-PR annotation
- Renders a final R Markdown report

---

## Requirements

**R packages:**

```r
install.packages(c(
  "dplyr",
  "readr",
  "tidyr",
  "caret",
  "pROC",
  "PRROC",
  "ggplot2",
  "rmarkdown",
  "knitr"
))
```

---

## Input Data

The script expects a file named `raw_churn_data.txt` in the working directory — a CSV-like text file with the following columns:

| Column | Type | Description |
|---|---|---|
| `CustomerID` | character | Unique customer identifier (dropped before modeling) |
| `Age` | numeric | Customer age |
| `TenureMonths` | numeric | Months as a customer |
| `MonthlyCharges` | numeric | Monthly bill amount |
| `TotalCharges` | numeric | Total amount billed |
| `InternetService` | categorical | Type of internet service |
| `ContractType` | categorical | Contract length/type |
| `PaymentMethod` | categorical | Payment method used |
| `Churn` | binary (0/1) | Target variable — 1 = churned |

---

## Usage

Open `Untitled3.ipynb` in Jupyter (with an R kernel) or RStudio, place `raw_churn_data.txt` in the same directory, and run all cells in order.

Alternatively, run each block directly in an R session:

```r
# Step 1 — install packages (first time only)
install.packages(c("knitr", "readr"))

# Step 2 — run cleaning and modeling
source("churn_pipeline.R")

# Step 3 — render report
rmarkdown::render('churn_report.Rmd')
```

---

## Outputs

| File | Description |
|---|---|
| `cleaned_churn_data.csv` | Cleaned, imputed, type-corrected dataset |
| `roc_curve.png` | ROC curve with AUC score annotated |
| `pr_curve.png` | Precision-Recall curve with AUC-PR annotated |
| `churn_report.Rmd` | R Markdown report rendering all results |

---

## Model Details

**Algorithm:** Logistic Regression (`glm`, binomial family)

**Validation:** 5-fold cross-validation via `caret::trainControl`

**Optimization metric:** ROC-AUC

**Evaluation metrics:**
- Confusion matrix (sensitivity, specificity, precision, F1)
- ROC curve + AUC
- Precision-Recall curve + AUC-PR

The PR curve is included alongside the ROC curve because churn datasets are often class-imbalanced — the PR curve gives a more informative picture of model performance on the minority (churned) class than ROC alone.

---

## License

MIT License — free to use, modify, and distribute.
