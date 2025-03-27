# Bank-Custumor-Churn-
Customer Churn Prediction using Random Forest
Project Overview
This project focuses on predicting customer churn (i.e., whether a customer will leave or stay) using machine learning techniques. The dataset contains customer details such as their age, account balance, credit score, and more. The goal is to predict whether a customer will churn (exit the service) based on these features.

In this project, we utilize the Random Forest classifier to build a machine learning model, and we perform hyperparameter tuning using GridSearchCV to find the best parameters for the model. We also evaluate the model performance using various metrics like accuracy, precision, recall, F1-score, and ROC-AUC. Finally, we save the trained model and scaler for future predictions in a production environment.

Technologies Used
Python: The primary programming language used for this project.

Pandas: For data manipulation and analysis.

Scikit-learn: For machine learning, model building, and evaluation.

Matplotlib/Seaborn: For data visualization (e.g., ROC curve).

Joblib: For saving and loading the model and scaler.

Dataset
The dataset consists of customer details including:

Age: Customer's age.

Tenure: Number of years the customer has been with the company.

Balance: Account balance.

NumOfProducts: Number of products the customer uses.

HasCrCard: Whether the customer has a credit card (1 = Yes, 0 = No).

IsActiveMember: Whether the customer is an active member (1 = Yes, 0 = No).

EstimatedSalary: The estimated salary of the customer.

Exited: Whether the customer exited (1 = Yes, 0 = No).

The target variable is Exited, where the model will predict whether the customer will churn.

Steps Followed in the Project
Step 1: Data Preprocessing
Removed unnecessary columns like RowNumber, CustomerId, and Surname.

One-hot encoded categorical features (Geography, Gender, Card Type).

Split the data into features (X) and target (y).

Step 2: Train-Test Split
Split the dataset into training and testing sets with a 70/30 ratio, using stratification to ensure balanced class distribution.

Step 3: Feature Scaling
Scaled the features using StandardScaler to standardize the feature values, which improves the performance of many machine learning models.

Step 4: Model Building
Trained a Random Forest classifier with class_weight='balanced' to handle the imbalanced churn dataset.

Applied GridSearchCV to find the best hyperparameters for the model (e.g., n_estimators, max_depth, and min_samples_split).

Step 5: Model Evaluation
Evaluated the model using:

Classification Report (precision, recall, F1-score)

Confusion Matrix

ROC-AUC score

ROC Curve visualization

Step 6: Model Saving
Saved the trained Random Forest model and scaler using joblib.

Saved the list of feature columns for future reference.
