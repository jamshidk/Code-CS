import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt



df=pd.read_csv("wine.csv")
y=df["quality"]
X=df.drop(columns=["quality", "class"], errors="ignore")

classes = sorted(y.unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = y.map(class_to_idx)


X_train, X_test, y_train, y_test = train_test_split(X, y_idx, test_size=0.2, random_state=42, stratify=y_idx)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

scaler = StandardScaler(with_mean=True)  # important for sparse-ish one-hot
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_s, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s, dtype=torch.float32)


y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test_t = torch.tensor(y_test.to_numpy(), dtype=torch.long)
y_val_t = torch.tensor(y_val.to_numpy(), dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, activation,output_size, dropout=0):
        super(MLP, self).__init__()
        layers = []
        prev=input_size
        for h in hidden_size:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev=h
        layers.append(nn.Linear(prev, output_size))
        self.layers=nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def evaluate(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss=0.0
        all_preds=[]
        all_targets=[]
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits=model(X_batch)
                loss=criterion(logits, y_batch)
                total_loss+=loss.item()*X_batch.size(0)
                preds=torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
            average_loss=total_loss/len(loader.dataset)
            y_true=np.concatenate(all_targets)
            y_pred=np.concatenate(all_preds)
            acc=accuracy_score(y_true, y_pred)
            f1=f1_score(y_true, y_pred, average='macro', zero_division=0)
            return average_loss, acc, f1
    

def train_mlp_sgd(model, X_train, y_train, X_val, y_val, epochs=200, lr=1e-3, batch_size=128, seed=42, weight_decay=1e-4, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)

    
    
    input_dim=X_train.shape[1]
    output_dim=int(torch.unique(y_train).numel())
    
    train_loader=DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader=DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)

    history={
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': [],
        'best_epoch': None,
        'best_val_loss': float('inf'),
        
    }


    best_state=None
    patience_left=patience

    for epoch in range(1,epochs+1):
        model.train()
        running_loss=0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits=model(X_batch)
            loss=criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*X_batch.size(0)

        train_loss=running_loss/len(train_loader.dataset)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_loss < history['best_val_loss']:
            history['best_val_loss']=val_loss
            history['best_epoch']=epoch
            best_state={k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left=patience
        else:
            patience_left-=1
            if patience_left==0:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Best Val Loss: {history['best_val_loss']:.4f}, Best Epoch: {history['best_epoch']}")


    model.load_state_dict(best_state)

    
    return model, history


# Keep SAME architecture / training protocol for all activations
hidden_size = (64,64)      # or (64,64) if you prefer, but keep constant
epochs = 1000
lr = 1e-3
batch_size = 128
weight_decay = 1e-4
patience = 20
seed = 42
dropout = 0.0

activations = {
    "ReLU": nn.ReLU(),
    "tanh": nn.Tanh(),
    "GELU": nn.GELU(),
    "SiLU": nn.SiLU(),   # Swish-like
}

results = []
histories = {}

# loaders for final TEST eval
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()

for name, act in activations.items():
    model = MLP(
        input_size=X_train_t.shape[1],
        hidden_size=hidden_size,
        output_size=int(y_train_t.unique().numel()),
        activation=act,
        dropout=dropout
    )

    model, hist = train_mlp_sgd(
        model, X_train_t, y_train_t, X_val_t, y_val_t,
        epochs=epochs, lr=lr, batch_size=batch_size,
        weight_decay=weight_decay, patience=patience, seed=seed
    )

    # Evaluate best checkpoint on TEST
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)

    best_ep = hist["best_epoch"]
    best_val_f1 = hist["val_f1"][best_ep - 1]
    best_val_acc = hist["val_acc"][best_ep - 1]
    best_val_loss = hist["val_loss"][best_ep - 1]

    results.append({
        "Activation": name,
        "Best Epoch": best_ep,
        "Best Val Loss": best_val_loss,
        "Best Val Acc": best_val_acc,
        "Best Val Macro-F1": best_val_f1,
        "Test Acc (best ckpt)": test_acc,
        "Test Macro-F1 (best ckpt)": test_f1,
    })

    histories[name] = hist

print(results)
df_results = pd.DataFrame(results).sort_values(by="Best Val Macro-F1", ascending=False)
print(df_results)


plt.figure(figsize=(8,5))
for name, hist in histories.items():
    plt.plot(hist["val_f1"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Macro-F1")
plt.title(f"Wine MLP {hidden_size} (SGD): Val Macro-F1 vs Epoch by Activation")
plt.legend()
plt.grid(True)
plt.savefig(f"Wine_MLP_{hidden_size}_SGD_Val_Macro-F1_vs_Epoch_by_Activation.png")
plt.show()


plt.figure(figsize=(8,5))
for name, hist in histories.items():
    plt.plot(hist["val_loss"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Cross-Entropy Loss")
plt.title(f"Wine MLP {hidden_size} (SGD): Val Loss vs Epoch by Activation")
plt.legend()
plt.grid(True)
plt.savefig(f"Wine_MLP_{hidden_size}_SGD_Val_Loss_vs_Epoch_by_Activation.png")
plt.show()
