# Customer_churn_Analysis-USING-RandomForest-Bagging-
This project predicts bank customer churn using machine learning techniques. It includes exploratory data analysis, feature engineering, and model training using Logistic Regression and Random Forest. The goal is to identify customers likely to leave based on demographic and financial attributes.


Project Overview

Customer churn prediction is a critical problem for businesses, especially in the banking sector. Identifying customers who are likely to leave helps companies take preventive measures to retain them.
This project uses Machine Learning techniques to predict whether a customer will exit (churn) or stay with the bank based on their demographic and financial attributes.
The workflow includes:

1.Exploratory Data Analysis (EDA)

  a->Data preprocessing and feature engineering

  b->Logistic Regression baseline model

  c->Random Forest model improvement

  d->Model evaluation using classification metrics

  e->Feature importance analysis.

  2.Dataset Description:-

The dataset contains customer information such as:

| Feature         | Description                           |
| --------------- | ------------------------------------- |
| CreditScore     | Customer credit score                 |
| Geography       | Customer location                     |
| Gender          | Male/Female                           |
| Age             | Customer age                          |
| Tenure          | Number of years with the bank         |
| Balance         | Account balance                       |
| NumOfProducts   | Number of bank products used          |
| IsActiveMember  | Whether customer is active            |
| EstimatedSalary | Estimated salary                      |
| Exited          | Target variable (1 = churn, 0 = stay) |

3. Exploratory Data Analysis (EDA)
  A.Customer Age Distribution

  This plot shows how churn varies with customer age.
  <img width="684" height="464" alt="Screenshot 2026-03-05 032411" src="https://github.com/user-attachments/assets/122b05f8-73d7-4748-9f42-7789e428a5a4" />


  Observation:
  
  Customers between 40–60 years show higher churn probability.

  B.Customer Activity vs Churn
  
  <img width="652" height="470" alt="Screenshot 2026-03-05 032357" src="https://github.com/user-attachments/assets/4b02310f-e8e9-4b98-a385-fd7e64ee8a20" />

  Observation:
  
  Inactive members are more likely to churn.

  c.Geography vs Churn
  
  <img width="698" height="488" alt="Screenshot 2026-03-05 032500" src="https://github.com/user-attachments/assets/0f027cba-f7b4-472c-b610-3083ac7e8acf" />

  Observation:
  
  Customers in Germany show higher churn rates compared to France and Spain.

4.Model Implementation:

1️⃣ Logistic Regression

Initially, Logistic Regression was used as a baseline model.

However, due to:

1.Non-linear feature relationships

2.Data imbalance

The performance was limited.

2️⃣ Random Forest Classifier

To improve performance and handle non-linear relationships, Random Forest was used.

Model Configuration:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    class_weight='balanced',
    bootstrap=True,
    criterion='gini',
    min_samples_split=0.3,
    n_estimators=250,
    random_state=42,
    max_features='log2'
)


Reasons for choosing Random Forest:

  1.Handles non-linear patterns
  
  2.Works well with mixed numerical & categorical features
  
  3.Reduces bias-variance tradeoff
  
  4.Uses ensemble learning
  

5.Model Evaluation

Classification Report:


| Class      | Precision | Recall | F1 Score |
| ---------- | --------- | ------ | -------- |
| 0 (Stayed) | 0.92      | 0.89   | 0.91     |
| 1 (Exited) | 0.56      | 0.61   | 0.58     |

Overall Accuracy: 85%

Interpretation:

  1.Model performs very well for non-churn customers
  
  2.Churn prediction is harder due to class imbalance

Confusion Matrix

Confusion Matrix Values:


|             | Predicted Stay | Predicted Exit |
| ----------- | -------------- | -------------- |
| Actual Stay | 1474           | 136            |
| Actual Exit | 173            | 217            |


<img width="661" height="479" alt="Screenshot 2026-03-05 032250" src="https://github.com/user-attachments/assets/f68d837d-9f66-49ad-993c-0acee7a2f5b5" />


6.Feature Importance:

Random Forest allows us to see which features influence churn the most.

Top Features:

| Feature         | Importance |
| --------------- | ---------- |
| Age             | 0.31       |
| NumOfProducts   | 0.19       |
| Balance         | 0.12       |
| EstimatedSalary | 0.11       |
| CreditScore     | 0.11       |
| Tenure          | 0.05       |

Key Insight:

Age is the most important feature affecting churn behavior.


7.Tech Stack:

  1.Python
  
  2.Pandas
  
  3.NumPy
  
  4.Scikit-learn
  
  5.Matplotlib

  6.Seaborn
  
  7.Jupyter Notebook
  
  8.Machine Learning
  

8.PROJECT WORKFLOW::::


  Data Collection
      ↓
Exploratory Data Analysis
      ↓
Feature Encoding & Scaling
      ↓
Train Test Split
      ↓
Logistic Regression (Baseline)
      ↓
Random Forest Classifier
      ↓
Model Evaluation
      ↓
Feature Importance Analysis


9.Key Learnings:

  1.Logistic regression struggles with non-linear datasets
  
  2.Random Forest improves performance by combining multiple decision trees
  
  3.Feature importance helps understand which factors drive churn
  
