Predicting Bank Term Deposit Subscriptions
Project Overview
This project focuses on predicting whether a bank client will subscribe to a term deposit using the “bank-full.csv” (https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset. We implement and compare multiple classification models, including logistic regression (baseline and regularized) and K-Nearest Neighbors (KNN), to identify the most effective approach for subscriber prediction.
Project Steps
1. Baseline Logistic Regression
   
    •	Loaded the dataset and preprocessed data:
        o	Converted target variable y to binary (yes → 1, no → 0)
        o	One-hot encoding for categorical features
        o	Standardized numerical features
   
    •	Split data into 80% training and 20% testing sets
   
    •	Trained a logistic regression model without hyperparameter tuning
   
    •	Evaluated model using:
        o	Accuracy, Precision, Recall, F1 Score, ROC-AUC
        o	Confusion Matrix and Classification Report
   
    •	Observations: High accuracy (90.38%) and precision (67.15%) but very low recall (34.78%), indicating many missed positive cases.
   
3. Optimized Logistic Regression with Hyperparameter Tuning
   
    •	Used GridSearchCV to tune hyperparameters:
        o	Regularization strength (C)
        o	Penalty type (l1)
        o	Solver type
        o	Class weights
   
    •	Tuned model achieved improved recall (83.08%) and F1 score (0.563), while maintaining strong ROC-AUC (0.9142)
   
    •	Plotted F1 scores vs regularization parameter (C) to visualize optimization results
5. Regularized Logistic Regression Models
   
    •	Built two additional models with:
        o	L2 (Ridge) Regularization
        o	ElasticNet Regularization
   
    •	Hyperparameter tuning performed using the same evaluation framework as Q1
   
    •	Results:
        o	L2: Recall = 83.08%, F1 = 0.563
        o	ElasticNet: Recall = 75.14%, F1 = 0.568, Accuracy = 86.63%, ROC-AUC = 0.907
   
    •	Interpretation: ElasticNet balances precision and recall best, making it the most practical model.
   
7. Model Comparison
   
    •	Compared Base, L1, L2, and ElasticNet models on:
        o	Accuracy, Precision, Recall, F1 Score, ROC-AUC
        o	Training time
   
    •	ElasticNet identified as the best-performing model due to its balanced metrics and practical deployment potential.
   
9. K-Nearest Neighbors (KNN)
    
    •	Implemented KNN and tuned K using cross-validated F1 score
    •	Optimal K found to be 3
    •	Evaluated performance on test data:
        o	Accuracy = 88.90%, Recall = 34.31%, F1 = 0.420, ROC-AUC = 0.773
    •	Comparison with Logistic Regression:
        o	KNN trains faster and requires no trainable parameters (non-parametric)
        o	Lower F1 and recall make it less effective for subscriber identification than ElasticNet
   
Key Insights
    •	Logistic regression with ElasticNet regularization provides the best balance between precision and recall
    •	Regularization helps improve generalization and reduce overfitting
    •	KNN is simple and fast but less suitable for imbalanced data with a rare positive class
    •	High recall is crucial in marketing campaigns to identify potential subscribers effectively

Technologies Used
    •	Python 3.x
    •	Pandas, NumPy
    •	Scikit-learn (LogisticRegression, KNeighborsClassifier, GridSearchCV)
    •	Matplotlib, Seaborn
    •	Joblib (for model and scaler saving)
    
File Structure
    •	bank-full.csv — Dataset used for training
    •	bank.csv — Additional test dataset
    •	best_l1_model.pkl — Saved tuned logistic regression model
    •	scaler.pkl — Saved standard scaler
    •	model_comparison.csv — Model performance comparison results
    •	README.md — Project description and instructions
    
Usage
    1.	Clone the repository
    2.	Install dependencies:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib

