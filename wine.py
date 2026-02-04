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


df=pd.read_csv("wine.csv")

x=df.drop(columns=["quality", "class"], errors="ignore", axis=1)
y=df["quality"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

"""
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
#print("Best Model:", best_model)
#print("Best Macro F1 Score:", best_tuple[0])
#print("Best Accuracy:", best_tuple[1])
#print("Best Parameters:", best_params)

#best_model.fit(x_train, y_train)
#pred=best_model.predict(x_test)
#print(classification_report(y_test, pred))
"""

#################################################

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

"""
k_values = [1, 3, 5, 7, 11, 21, 31, 51]

best_knn = None
best_tuple = (-1, -1)
best_k = None
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_scaled, y_train)
    pred = knn.predict(x_val_scaled)
    macro_f1 = f1_score(y_val, pred, average='macro', zero_division=0)
    acc = accuracy_score(y_val, pred)
    results.append((macro_f1, acc))
    current_tuple = (macro_f1, acc)
    if current_tuple > best_tuple:
        best_tuple = current_tuple
        best_knn = knn
        best_k = k
print("Best K:", best_k)
print("Best Macro F1 Score:", best_tuple[0])
print("Best Accuracy:", best_tuple[1])
print("Best KNN Model:", best_knn)

best_knn.fit(x_train_scaled, y_train)
pred = best_knn.predict(x_test_scaled)
print(classification_report(y_test, pred, zero_division=0))

plt.plot(k_values, [t[0] for t in results], label="Macro-F1")
plt.plot(k_values, [t[1] for t in results], label="Accuracy")
plt.title("KNN Performance")
plt.xlabel("k")
plt.ylabel("Score")
plt.legend()
plt.savefig("knn_performance.png")
plt.show()

"""
"""
best_nonlin = None
best_nonlin_score = (-1, -1)
best_nonlin_C = None
best_nonlin_gamma = None
C_values = [0.01, 0.1, 1, 10, 100, 1000]
gamma_values = [0.01, 0.1, 1, 10, 100, 1000]
results_nonlin = []

for C in C_values:
    for gamma in gamma_values:
        svm_nonlin = SVC(kernel="rbf", C=C, gamma=gamma, decision_function_shape="ovr")
        svm_nonlin.fit(x_train_scaled, y_train)
        pred = svm_nonlin.predict(x_val_scaled)
        macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        acc = accuracy_score(y_val, pred)
        results_nonlin.append((C, gamma, macro_f1, acc))
        if (macro_f1, acc) > best_nonlin_score:
            best_nonlin_score = (macro_f1, acc)
            best_nonlin = svm_nonlin
            best_nonlin_C = C
            best_nonlin_gamma = gamma

print("Best Nonlinear C:", best_nonlin_C)
print("Best Nonlinear Gamma:", best_nonlin_gamma)
print("Best Nonlinear VAL Macro-F1:", best_nonlin_score)
print("Best Nonlinear Model:", best_nonlin)

best_nonlin.fit(x_train_scaled, y_train)
pred = best_nonlin.predict(x_test_scaled)
print(classification_report(y_test, pred, zero_division=0))

# Model complexity curve 1: C vs val macro F1 (gamma fixed at best)
gamma_fixed = best_nonlin_gamma
c_curve = [(r[0], r[2]) for r in results_nonlin if r[1] == gamma_fixed]
c_curve.sort(key=lambda t: t[0])
plt.figure()
plt.plot([t[0] for t in c_curve], [t[1] for t in c_curve], "o-", label=f"Val (gamma={gamma_fixed})")
plt.xscale("log")
plt.xlabel("C (regularization)")
plt.ylabel("Macro F1 Score")
plt.title("SVM RBF - Vary C (gamma fixed)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("svm_rbf_complexity_C.png")
plt.show()

# Model complexity curve 2: Gamma vs val macro F1 (C fixed at best)
c_fixed = best_nonlin_C
gamma_curve = [(r[1], r[2]) for r in results_nonlin if r[0] == c_fixed]
gamma_curve.sort(key=lambda t: t[0])
plt.figure()
plt.plot([t[0] for t in gamma_curve], [t[1] for t in gamma_curve], "s-", label=f"Val (C={c_fixed})")
plt.xscale("log")
plt.xlabel("Gamma (kernel width)")
plt.ylabel("Macro F1 Score")
plt.title("SVM RBF - Vary Gamma (C fixed)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("svm_rbf_complexity_gamma.png")
plt.show()
"""


# Train MLP epoch-by-epoch to record training and validation loss (to spot overfitting)
n_epochs = 500
train_losses = []
val_losses = []

mlp = MLPClassifier(
    hidden_layer_sizes=(32,),
    learning_rate_init=1e-4,
    learning_rate="constant",
    activation="relu",
    solver="sgd",
    max_iter=1,
    warm_start=True,
    random_state=42,
    alpha=0.001
)

for epoch in range(n_epochs):
    mlp.fit(x_train_scaled, y_train)
    train_losses.append(mlp.loss_curve_[-1])
    val_proba = mlp.predict_proba(x_val_scaled)
    val_losses.append(log_loss(y_val, val_proba))

# Final model has been trained for n_epochs; evaluate
pred = mlp.predict(x_val_scaled)
print("Val Accuracy:", accuracy_score(y_val, pred))
print("Val Macro-F1:", f1_score(y_val, pred, average="macro", zero_division=0))
pred = mlp.predict(x_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, pred))
print("Test Macro-F1:", f1_score(y_test, pred, average="macro", zero_division=0))

# Plot training loss and validation loss to see when overfitting starts
plt.figure()
plt.plot(train_losses, color="tab:blue", label="Training Loss")
plt.plot(val_losses, color="tab:orange", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (cross-entropy)")
plt.title("MLP: Training vs Validation Loss (overfitting when val loss rises)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlp_train_val_loss.png")
plt.show()
