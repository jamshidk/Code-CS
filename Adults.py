import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss, confusion_matrix, ConfusionMatrixDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve


df = pd.read_csv("adult.csv")

x = df.drop(columns=["class"], errors="ignore", axis=1)
y = df["class"]

# Encode categorical columns to numeric (DecisionTree and sklearn need numeric input)
x = pd.get_dummies(x, drop_first=False, dtype=float)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


"""
best_model = None
best_tuple = (-1, -1)  # (macro_f1, acc)
best_params = None


"""
base=DecisionTreeClassifier(random_state=42, criterion='gini')
path=base.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas=path.ccp_alphas

records=[]
for ccp_alpha in ccp_alphas:
    model=DecisionTreeClassifier(random_state=42, criterion='gini', ccp_alpha=ccp_alpha)
    model.fit(x_train, y_train)
    pred=model.predict(x_val)


    records.append({
        "ccp_alpha": ccp_alpha,
        "val_macro_f1": f1_score(y_val, pred, average="macro", zero_division=0),
        "val_acc": accuracy_score(y_val, pred),
        "depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
    })


df_prune = pd.DataFrame(records)
plt.figure()
plt.plot(df_prune["ccp_alpha"], df_prune["val_macro_f1"], marker="o")
plt.xscale("log")
plt.xlabel("ccp_alpha (post-pruning strength)")
plt.ylabel("Validation Macro-F1")
plt.title("Decision Tree Pruning Curve (Adults Income): Macro-F1 vs ccp_alpha")
plt.grid(True)
plt.savefig("Adults_DecisionTree_Pruning_Curve_Macro-F1_vs_ccp_alpha.png")
plt.show()


best_row=df_prune.sort_values(by=["val_macro_f1","val_acc"], ascending=False).iloc[0]

best_alpha = best_row["ccp_alpha"]
final_dt = DecisionTreeClassifier(
    random_state=42,
    criterion='gini',
    ccp_alpha=best_alpha,
)
final_dt.fit(x_train, y_train)
pred=final_dt.predict(x_test)
print(classification_report(y_test, pred, zero_division=0))

print("Chosen ccp_alpha:", best_alpha)
print("Final depth:", final_dt.get_depth())
print("Final leaves:", final_dt.get_n_leaves())
print("Test Macro-F1:", f1_score(y_test, pred, average="macro", zero_division=0))
print("Test Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))


cm = confusion_matrix(y_test, pred, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y.unique()))
disp.plot(xticks_rotation=45)
plt.title("Adults Decision Tree Confusion Matrix (Test)")
plt.savefig("Adults_DecisionTree_Confusion_Matrix_Test.png")
plt.show()

fractions = np.linspace(0.1, 1.0, 10)
train_scores, val_scores = [], []

n = len(x_train)
idx = np.arange(n)

for frac in fractions:
    m = int(frac * n)
    sub_idx = idx[:m]  # for fairness, shuffle idx once if you want
    x_sub = x_train.iloc[sub_idx]
    y_sub = y_train.iloc[sub_idx]

    clf = DecisionTreeClassifier(random_state=42, criterion="gini", ccp_alpha=best_alpha)
    clf.fit(x_sub, y_sub)

    pred_tr = clf.predict(x_sub)
    pred_val = clf.predict(x_val)

    train_scores.append(f1_score(y_sub, pred_tr, average="macro", zero_division=0))
    val_scores.append(f1_score(y_val, pred_val, average="macro", zero_division=0))

plt.figure()
plt.plot(fractions, train_scores, marker="o", label="Train Macro-F1")
plt.plot(fractions, val_scores, marker="o", label="Val Macro-F1")
plt.xlabel("Training fraction")
plt.ylabel("Macro-F1")
plt.title("Learning Curve (Adults Income Decision Tree): Train vs Val Macro-F1")
plt.legend()
plt.grid(True)
plt.savefig("Wine_DecisionTree_Learning_Curve_Train_vs_Val_Macro-F1.png")
plt.show()

t0 = time.perf_counter()
final_dt.fit(x_train, y_train)
fit_time = time.perf_counter() - t0

# Predict time
t0 = time.perf_counter()
_ = final_dt.predict(x_test)
pred_time = time.perf_counter() - t0

runtime_df = pd.DataFrame([{
    "Model": "DecisionTree (ccp_alpha tuned)",
    "Fit time (s)": fit_time,
    "Predict time (s)": pred_time,
    "Notes": "CPU timing; report your machine specs (e.g., MacBook M2 / i7 etc.)"
}])

print(runtime_df)

"""
####
## KNN

print(x_train.head())
scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_val_scaled=scaler.transform(x_val)
x_test_scaled=scaler.transform(x_test)

macro_f1_list = []
acc_list = []
k_list = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]

for n_neighbors in k_list:
    knn=KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    knn.fit(x_train_scaled, y_train)
    pred=knn.predict(x_val_scaled)
    macro_f1 = f1_score(y_val, pred, average='macro')
    acc = accuracy_score(y_val, pred)
    macro_f1_list.append(macro_f1)
    acc_list.append(acc)
    current_tuple = (macro_f1, acc)
    if current_tuple > best_tuple:
        best_tuple = current_tuple
        best_model = knn
        best_params = {'n_neighbors': n_neighbors}

print("Best Model:", best_model)
print("Best Macro F1 Score:", best_tuple[0])
print("Best Accuracy:", best_tuple[1])
print("Best Parameters:", best_params)

best_model.fit(x_train_scaled, y_train)
pred=best_model.predict(x_test_scaled)
print(classification_report(y_test, pred, zero_division=0))



plt.plot(k_list, macro_f1_list, label='Macro F1 Score')
plt.plot(k_list, acc_list, label='Accuracy')
plt.xlabel('K')
plt.ylabel('Score')
plt.title('Score vs K')
plt.legend()
plt.show()


##############

## SVM

macro_f1_list = []
acc_list = []
c_list = [0.001, 0.01, 0.1, 1, 10, 100]
gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]
best_tuple = (-1, -1)
best_params = None

for c in c_list:
    for gamma in gamma_list:
        svm=SVC(kernel='rbf', C=c, gamma=gamma, probability=False, random_state=42)
        svm.fit(x_train_scaled, y_train)
        pred=svm.predict(x_val_scaled)
        macro_f1 = f1_score(y_val, pred, average='macro')
        acc = accuracy_score(y_val, pred)
        macro_f1_list.append(macro_f1)
        acc_list.append(acc)
        current_tuple = (macro_f1, acc)
        if current_tuple > best_tuple:
            best_tuple = current_tuple
            best_model = svm
            best_params = {'C': c, 'gamma': gamma}

print("Best Parameters:", best_params)
print("Best Macro F1 Score:", best_tuple[0])
print("Best Accuracy:", best_tuple[1])

best_model.fit(x_train_scaled, y_train)
pred=best_model.predict(x_test_scaled)
print(classification_report(y_test, pred, zero_division=0))

c_curve=learning_curve(best_model, x_train_scaled, y_train, cv=5, scoring='macro_f1')
plt.plot(c_curve[0], c_curve[1], label='Macro F1 Score')
plt.xlabel('Training Size')
plt.ylabel('Macro F1 Score')
plt.title('Macro F1 Score vs Training Size')
plt.legend()
plt.show()




###############

"""