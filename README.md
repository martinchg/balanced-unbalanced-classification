# Balanced vs. Unbalanced Classification

## Project Overview
This project is developed as part of the **Development Project in Machine Learning** 

The main objective is to implement a **reusable Machine Learning workflow** and apply it to two distinct binary classification datasets to compare performance and handling strategies. The project specifically focuses on addressing data quality challenges: **missing values** and **class imbalance**.

## Repository Structure
In accordance with the project requirements, the code is organized as follows:

## Datasets

### 1. SPAM Dataset (UCI)
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/94/spambase) 
* **Task:** Binary classification (Spam vs. Non-Spam).
* **Characteristics:** 5,000 entries, 50 features.
* **Specific Challenge:** Contains **missing values** requiring imputation.
* **Balance:** Balanced dataset.

### 2. Diabetes Health Indicators (Kaggle)
* **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) 
* **Task:** Predict diabetes presence based on health indicators.
* **Characteristics:** 250,000 entries, 21 features.
* **Specific Challenge:** Very large dataset with **class imbalance**.
* **Balance:** Unbalanced dataset.

## Machine Learning Workflow
The core logic is implemented in `lib.py` and follows this pipeline:

1.  **Import:** Loading datasets from CSV/Excel files.
2.  **Preprocessing:**
    * **Cleaning:** Handling missing values (Mean/Median imputation).
    * **Normalization:** Centering and scaling features.
3.  **Splitting:** Creating Training and Testing sets.
4.  **Training:** Model training with Feature Selection.
5.  **Validation:** Evaluation using metrics adapted to each problem (e.g., Accuracy, F1-Score, ROC-AUC).

## Usage

### 1. Prerequisites
Install the required libraries:
'''bash
pip install -r requirements.txt '''

### 2. Running the Analysis
The results, curves, and comparisons are presented in the Jupyter Notebooks:

SPAM Analysis: Open analysis_spam.ipynb

Diabetes Analysis: Open analysis_diabetes.ipynb

### 3. Running Tests
To validate the reliability of the functions in lib.py, run the unit tests:


## Contributors

CHASSAING Martin 

ROUBERTOU Amaury

GARRANA Simon 

LESAGE Thomas