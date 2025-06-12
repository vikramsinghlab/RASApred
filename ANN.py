import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Load data
data = pd.read_csv('input_RASA.csv')

# Separate features and labels
X = data.drop('Labels', axis=1).values
y = data['Labels'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels for multi-class classification
lb = LabelBinarizer()
y_bin = lb.fit_transform(y)

# Initialize 5-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store performance metrics
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
auc_scores = []

# Initialize MLPClassifier
model_ANN = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=500, random_state=42)

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_bin[train_index], y_bin[test_index]

    # Train model
    model_ANN.fit(X_train, y_train)

    # Predictions
    y_pred_prob = model_ANN.predict_proba(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Compute performance metrics
    accuracy_scores.append(accuracy_score(y_test_labels, y_pred))
    f1_scores.append(f1_score(y_test_labels, y_pred, average='weighted'))
    precision_scores.append(precision_score(y_test_labels, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test_labels, y_pred, average='weighted'))
    auc_scores.append(roc_auc_score(y_test, y_pred_prob, average='weighted', multi_class='ovr'))

# Print average results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Mean Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Mean AUC (Weighted): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

import pickle

filename = 'ANN_model.sav'
pickle.dump(model_ANN, open(filename, 'wb'))
