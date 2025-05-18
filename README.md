# 🤖🧠 Validata - Predicting Loan Approval Using KNN and Decision Trees

This project is a machine learning solution developed for a bank to predict whether a loan application should be approved or not based on 
applicant data such as the applicant's income, credit score, loan amount, loan term, and employment status.

The project is used to build two different models to predict the loan approval status using KNN and Decision Trees algorithms, respectively.
It can also be used to test the models and also evaluate and compare the results between them.

---

## 🚀 Getting Started

### 1. Activate the virtual environment which contains the necessary libraries and dependencies
```bash
# On macOS and Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```
The additional libraries which are used for machine learning and manipulating with data can also be found in `requirements.txt` file.

### 2. Training the models
First thing one needs to do is to train the models using the raw data which can be found in `data/raw/loan_data.csv`. As mentioned in the 
description the project supports training models using both KNN and decision trees algorithms. Before the training of the models, the raw 
data taken from `.csv` needs to be preprocessed first using the functions defined in `src/data_preprocessing.py`, in order to make sure 
that:

- Categorical variables like `employment_status` are converted to numerical format using label encoding.
- Numerical features such as `income`, `credit_score` and `loan_amount` are scaled using `StandardScaler` to ensure uniform feature 
distribution, especially important for distance-based models like KNN.
- The dataset is split into training and testing sets, allowing the models to be evaluated on unseen data.

To preprocess the raw data one can run:
```bash
cd src
python data_preprocessing.py
```
This will create the preprocessed dataset for both training and testing stored under `data/preprocessed` which are then used to train the 
models.

To train the model using KNN algorithm one can run:
```bash
cd src
python train_knn.py
```
This will create a trained model file like this: `models/knn_model.joblib`.

To train the model using Decision Tree algorithm one can run:
```bash
cd src
python train_decision_tree.py
```
This will create a trained model file like this: `models/tree_model.joblib`.

### 3. Testing the models
To test the models and see which one is more precise, one can run this command:
```bash
cd src
python evaluate.py
```
This will print evaluation metrics such as accuracy, precision, recall, and F1-score for each training model, and it will also create a 
reports image under `reports/figures/tree_feature_importance.png` showing with a graph the features importance and comparison between them.


## 📊 Report summarizing the findings and insights

When testing and evaluating the models following the steps on chapter #Testing the models I got these results:
```
--- KNN Evaluation ---
              precision    recall  f1-score   support

           0       0.75      1.00      0.86         3
           1       1.00      0.50      0.67         2

    accuracy                           0.80         5
   macro avg       0.88      0.75      0.76         5
weighted avg       0.85      0.80      0.78         5

--- Decision Tree Evaluation ---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       1.00      1.00      1.00         2

    accuracy                           1.00         5
   macro avg       1.00      1.00      1.00         5
weighted avg       1.00      1.00      1.00         5
```

From the results I can see that the Decision Tree algorithm is more precise than the KNN algorithm in this case because it achieved perfect 
precision (1.00 for both classes `0` and `1` and also for overall averages: macro average and weighted average). Also for recall and 
F1-score, the Decision Tree model outperformed KNN by achieving perfect scores (1.00) across all metrics. In contrast, the KNN model had 
lower recall for class 1 (0.50), which means it failed to correctly identify some of the actual positive cases.

The Decision Tree model also provides insights into which features were most influential in making predictions. As shown in the chart in
`reports/figures/tree_feature_importance.png` the most important features for determining loan approval were:
    
- Credit Score
- Income

Both of these features had the highest importance scores (close to 0.5 each), indicating that the model heavily relies on them when deciding 
whether to approve a loan. On the other hand, features such as `loan_amount`, `loan_term`, and `employment_status` had negligible or no 
influence in the model’s decision-making process. From this we can say that, at least for this dataset, financial stability indicators like 
`income`and `credit_score` are far more critical than other factors, and we can advise the bank to prioritize these metrics heavily in their 
risk assessment models, potentially assigning them a higher weight or establishing stricter thresholds compared to other variables when 
evaluating loan applications.