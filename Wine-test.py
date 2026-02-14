import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

data = pd.read_csv('adult.csv')

df=pd.read_csv("wine.csv")

X=df.drop(columns=["quality", "class"], errors="ignore", axis=1)
y=df["quality"]

#X = pd.get_dummies(X, drop_first=True, dtype=float)


#print(X.columns)  # show encoded feature names (no single 'education' column after one-hot)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

base=DecisionTreeClassifier(random_state=42, criterion='gini')
path=base.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas=path.ccp_alphas

records=[]
for ccp_alpha in ccp_alphas:
    model=DecisionTreeClassifier(random_state=42, criterion='gini', ccp_alpha=ccp_alpha)
    model.fit(X_train, y_train)
    pred=model.predict(X_val)


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
plt.title("Decision Tree Pruning Curve (Wine): Macro-F1 vs ccp_alpha")
plt.grid(True)
plt.savefig("Wine_DecisionTree_Pruning_Curve_Macro-F1_vs_ccp_alpha.png")
plt.show()

best_row=df_prune.sort_values(by=["val_macro_f1","val_acc"], ascending=False).iloc[0]

best_alpha = best_row["ccp_alpha"]
final_dt = DecisionTreeClassifier(
    random_state=42,
    criterion="gini",
    ccp_alpha=best_alpha
)
final_dt.fit(X_train, y_train)

print("Chosen ccp_alpha:", best_alpha)
print("Final depth:", final_dt.get_depth())
print("Final leaves:", final_dt.get_n_leaves())

pred_test = final_dt.predict(X_test)
print("Test Macro-F1:", f1_score(y_test, pred_test, average="macro", zero_division=0))
print("Test Accuracy:", accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test, zero_division=0))

cm = confusion_matrix(y_test, pred_test, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y.unique()))
disp.plot(xticks_rotation=45)
plt.title("Wine Decision Tree Confusion Matrix (Test)")
plt.show()

fractions = np.linspace(0.1, 1.0, 10)
train_scores, val_scores = [], []

n = len(X_train)
idx = np.arange(n)

for frac in fractions:
    m = int(frac * n)
    sub_idx = idx[:m]  # for fairness, shuffle idx once if you want
    X_sub = X_train.iloc[sub_idx]
    y_sub = y_train.iloc[sub_idx]

    clf = DecisionTreeClassifier(random_state=42, criterion="gini", ccp_alpha=best_alpha)
    clf.fit(X_sub, y_sub)

    pred_tr = clf.predict(X_sub)
    pred_val = clf.predict(X_val)

    train_scores.append(f1_score(y_sub, pred_tr, average="macro", zero_division=0))
    val_scores.append(f1_score(y_val, pred_val, average="macro", zero_division=0))

plt.figure()
plt.plot(fractions, train_scores, marker="o", label="Train Macro-F1")
plt.plot(fractions, val_scores, marker="o", label="Val Macro-F1")
plt.xlabel("Training fraction")
plt.ylabel("Macro-F1")
plt.title("Learning Curve (Wine DT): Train vs Val Macro-F1")
plt.legend()
plt.grid(True)
plt.savefig("Wine_DecisionTree_Learning_Curve_Train_vs_Val_Macro-F1.png")
plt.show()

t0 = time.perf_counter()
final_dt.fit(X_train, y_train)
fit_time = time.perf_counter() - t0

# Predict time
t0 = time.perf_counter()
_ = final_dt.predict(X_test)
pred_time = time.perf_counter() - t0

runtime_df = pd.DataFrame([{
    "Model": "DecisionTree (ccp_alpha tuned)",
    "Fit time (s)": fit_time,
    "Predict time (s)": pred_time,
    "Notes": "CPU timing; report your machine specs (e.g., MacBook M2 / i7 etc.)"
}])

print(runtime_df)


"""
best_model = None
best_tuple = (-1, -1)  # (macro_f1, acc)
best_params = None

for max_depth in range(1, 21):
    for min_samples_split in range(2, 11):
        for min_samples_leaf in range(1, 11):
            model = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            base_model=model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            macro_f1 = f1_score(y_val, y_pred, average='macro')
            acc = accuracy_score(y_val, y_pred)
            
            current_tuple = (macro_f1, acc)
            


            if current_tuple > best_tuple:
                best_tuple = current_tuple
                best_model = base_model
                best_params = {
                    'max_depth':max_depth,
                    'min_samples_split':min_samples_split,
                    'min_samples_leaf':min_samples_leaf
                }

print("Best Model:", best_model)
print("Best Macro F1 Score:", best_tuple[0])
print("Best Accuracy:", best_tuple[1])
print("Best Parameters:", best_params)

best_model.fit(X_train, y_train)
pred=best_model.predict(X_test)
print(classification_report(y_test, pred))

##################
"""