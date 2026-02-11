import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

df=pd.read_csv("wine.csv")
y = LabelEncoder().fit_transform(df["quality"])
X = df.drop(columns=["quality", "class"], errors="ignore", axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_val_scaled = scaler.transform(X_val)
x_test_scaled = scaler.transform(X_test)


def run_mlp_learning_curve_sklearn(
    arch,
    x_train_scaled, y_train,
    x_val_scaled, y_val,
    alpha=1e-4,
    lr=1e-3,
    batch_size=32,
    max_epochs=200,
    patience=10,
    tol=1e-4,
    random_state=42,):

    mlp=MLPClassifier(hidden_layer_sizes=arch,
    alpha=alpha,
    learning_rate_init=lr,
    learning_rate="constant",
    batch_size=batch_size,
    solver='sgd',
    max_iter=1,
    random_state=random_state,
    warm_start=True,
    activation='relu',
    early_stopping=False,)
    
    classes=np.unique(y_train)

    train_losses=[]
    val_losses=[]
    train_f1s=[]
    val_f1s=[]
    val_accs=[]
    train_accs=[]
    best_val_loss=float('inf')
    best_epoch=0
    best_model=None
    wait=0

    for epoch in range(1,max_epochs+1):
        mlp.fit(x_train_scaled, y_train)

        pred_train=mlp.predict_proba(x_train_scaled)
        pred_val=mlp.predict_proba(x_val_scaled)

        train_loss=log_loss(y_train, pred_train, labels=classes)
        val_loss=log_loss(y_val, pred_val, labels=classes)

        pred_train=mlp.predict(x_train_scaled)
        pred_val=mlp.predict(x_val_scaled)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_f1s.append(f1_score(y_train, pred_train, average='macro', zero_division=0))
        val_f1s.append(f1_score(y_val, pred_val, average='macro', zero_division=0))
        
        train_accs.append(accuracy_score(y_train, pred_train))
        val_accs.append(accuracy_score(y_val, pred_val))

        if val_loss < best_val_loss-tol:
            best_val_loss=val_loss
            best_epoch=epoch
            best_model=mlp.get_params()
            wait=0
        else:
            wait+=1
            if wait>=patience:
                break

        history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_f1": train_f1s,
            "val_f1": val_f1s,
            "train_acc": train_accs,
            "val_acc": val_accs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }
    return best_model, history


def plot_history(history, title_prefix="MLP"):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.axvline(history["best_epoch"], linestyle="--", label="Early stop (best)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{title_prefix}: Train vs Val Loss")
    plt.legend()
    plt.savefig(f"{title_prefix}_train_vs_val_loss.png")
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_f1"], label="Train Macro-F1")
    plt.plot(epochs, history["val_f1"], label="Val Macro-F1")
    plt.axvline(history["best_epoch"], linestyle="--", label="Early stop (best)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"{title_prefix}: Train vs Val Macro-F1")
    plt.legend()
    plt.savefig(f"{title_prefix}_train_vs_val_macro_f1.png")
    plt.show()

hidden_sizes = [(128,), (64,64), (48,48,48), (32,32,32,32)]
alpha_value = 1e-4
lr_value = 1e-3

best_arch = None
best_score = (-1, -1)
best_model = None
best_history = None

for arch in hidden_sizes:
    model, hist = run_mlp_learning_curve_sklearn(
        arch,
        x_train_scaled, y_train,
        x_val_scaled, y_val,
        alpha=alpha_value,
        lr=lr_value,
        batch_size=32,
        max_epochs=300,
        patience=10,
        tol=1e-4,
        random_state=42,
        
    )

    # score at best epoch (use val metrics at best_epoch-1 index)
    idx = hist["best_epoch"] - 1
    val_macro = hist["val_f1"][idx]
    val_acc = hist["val_acc"][idx]
    score = (val_macro, val_acc)

    print(f"Arch {arch} | best_epoch={hist['best_epoch']} | val_macro={val_macro:.4f} | val_acc={val_acc:.4f}")

    if score > best_score:
        best_score = score
        best_arch = arch
        best_model = model
        best_history = hist

print("\nBest architecture:", best_arch)
print("Best VAL (Macro-F1, Acc):", best_score)

# Plot curves for the best architecture
plot_history(best_history, title_prefix=f"MLP {best_arch}")



