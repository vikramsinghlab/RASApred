import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# Load data
data = pd.read_csv('input_RASA.csv')

# Separate features and labels
X = data.drop('Labels', axis=1)
y = data['Labels']

# Standardize features (SVM performs better with standardized data)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize SVM Classifier with probability estimation enabled
svm = SVC(kernel='poly', probability=True, random_state=42)

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

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train model
    svm.fit(X_train, y_train)

    # Predictions
    y_pred = svm.predict(X_test)
    y_pred_prob = svm.predict_proba(X_test)

    # Compute performance metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    
    # AUC score
    y_test_bin = lb.transform(y_test)  # Convert test labels to binary
    auc_scores.append(roc_auc_score(y_test_bin, y_pred_prob, average='weighted', multi_class='ovr'))

# Print average results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Mean Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Mean AUC (Weighted): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
import pickle

filename = 'SVM_model.sav'
pickle.dump(svm, open(filename, 'wb'))

