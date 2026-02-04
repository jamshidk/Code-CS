import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("adult.csv")

x = df.drop(columns=["class"], errors="ignore", axis=1)
y = df["class"]

# Encode categorical columns to numeric (DecisionTree and sklearn need numeric input)
x = pd.get_dummies(x, drop_first=False, dtype=float)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)



best_model = None
best_tuple = (-1, -1)  # (macro_f1, acc)
best_params = None

for max_depth in range(1, 21):
    for min_samples_split in range(2, 11):
        for min_samples_leaf in range(1, 6):
            base=DecisionTreeClassifier(random_state=42, 
            criterion='gini',
            class_weight='balanced',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf)

            base.fit(x_train, y_train)
            pred=base.predict(x_val)

            macro_f1 = f1_score(y_val, pred, average='macro')
            acc = accuracy_score(y_val, pred)
            current_tuple = (macro_f1, acc)

            if current_tuple > best_tuple:
                best_tuple = current_tuple
                best_model = base
                best_params = {
                    'max_depth':max_depth,
                    'min_samples_split':min_samples_split,
                    'min_samples_leaf':min_samples_leaf
                }       
print("Best Model:", best_model)
print("Best Macro F1 Score:", best_tuple[0])
print("Best Accuracy:", best_tuple[1])
print("Best Parameters:", best_params)

best_model.fit(x_train, y_train)
pred=best_model.predict(x_test)
print(classification_report(y_test, pred, zero_division=0))

