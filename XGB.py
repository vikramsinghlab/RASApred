import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Load data
data = pd.read_csv('input_RASA.csv')

# Separate features and labels
X = data.drop('Labels', axis=1)
y = data['Labels']


# Initialize 5-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store performance metrics
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
auc_scores = []

lb = LabelBinarizer()
y_bin = lb.fit_transform(y)  # Binarize labels for multi-class AUC

# 5-Fold Cross Validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Weakened XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(np.unique(y)), 
        eval_metric='mlogloss',
        n_estimators=500,  # Very few trees
        learning_rate=0.5,  # High learning rate
        max_depth=10,  # Shallow trees
        gamma=5,  # High regularization
        reg_lambda=10  # Strong L2 regularization
    )
    
    # Train the model
    xgb_model.fit(X_train, y_train)

    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_prob = xgb_model.predict_proba(X_test)

    # Compute performance metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    
    # AUC score
    y_test_bin = lb.transform(y_test)  # Convert test labels to binary
    auc_scores.append(roc_auc_score(y_test_bin, y_pred_prob, average='weighted', multi_class='ovr'))

# Print average results with standard deviation
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Mean Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Mean AUC (Weighted): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

import pickle

filename = 'XGB_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

