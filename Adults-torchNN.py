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



df=pd.read_csv("adult.csv")
y=df["class"]
X=df.drop(columns=["class"], errors="ignore")

X=pd.get_dummies(X,drop_first=False, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

scaler = StandardScaler(with_mean=False)  # important for sparse-ish one-hot
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_s.to_numpy() if hasattr(X_train_s, "to_numpy") else X_train_s, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_s.to_numpy()   if hasattr(X_val_s, "to_numpy")   else X_val_s, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s.to_numpy()  if hasattr(X_test_s, "to_numpy")  else X_test_s, dtype=torch.float32)


y_train_int = (y_train == ">50K").astype(int)
y_train_t = torch.tensor(y_train_int.to_numpy(), dtype=torch.long)
y_test_int = (y_test == ">50K").astype(int)
y_test_t = torch.tensor(y_test_int.to_numpy(), dtype=torch.long)
y_val_int = (y_val == ">50K").astype(int)
y_val_t = torch.tensor(y_val_int.to_numpy(), dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(MLP, self).__init__()
        layers = []
        prev=input_size
        for h in hidden_size:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev=h
        layers.append(nn.Linear(prev, output_size))
        self.layers=nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

def train_mlp_sgd(model, X_train, y_train, X_val, y_val, epochs=200, hidden_size=(128,), lr=1e-3, batch_size=128, dropout=0, seed=42, weight_decay=1e-4, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)

    
    
    input_dim=X_train.shape[1]
    output_dim=int(torch.unique(y_train).numel())
    
    train_loader=DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader=DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)


    train_losses=[]
    val_losses=[]
    val_f1s=[]
    val_accs=[]
    best_val_loss=float('inf')
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
        train_losses.append(train_loss)


        model.eval()
        running_val_loss=0.0
        all_preds=[]
        all_targets=[]
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits=model(X_batch)
                loss=criterion(logits, y_batch)
                running_val_loss+=loss.item()*X_batch.size(0)
                preds=torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_loss=running_val_loss/len(val_loader.dataset)
        val_losses.append(val_loss)

        y_true=np.concatenate(all_targets)
        y_pred=np.concatenate(all_preds)
        val_acc=accuracy_score(y_true, y_pred)
        val_f1=f1_score(y_true, y_pred, average='macro', zero_division=0)
        val_f1s.append(val_f1)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1 Score: {val_f1:.4f}")

        if val_loss < best_val_loss-1e-4:
            best_val_loss=val_loss
            best_state={k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left=patience
        else:
            patience_left-=1
            if patience_left==0:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    history={
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1s': val_f1s,
        'val_accs': val_accs
    }
    return model, history


model = MLP(input_size=X_train_t.shape[1], hidden_size=(128,), output_size=2, dropout=0)

model, history = train_mlp_sgd(
    model,
    X_train_t, y_train_t,
    X_val_t, y_val_t,
    epochs=200, lr=1e-3, batch_size=128,
    weight_decay=1e-4, patience=20
)



#print("Best Model:", model)
#print("Best Parameters:", history)

plt.figure(figsize=(8,5))
plt.plot(history['train_losses'], label="Train Loss")
plt.plot(history['val_losses'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training vs Validation Loss")
plt.savefig("train_val_loss.png")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(history['val_f1s'], label="Validation Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("Validation Macro-F1 vs Epoch")
plt.savefig("val_f1s.png")
plt.legend()
plt.grid(True)
plt.show()
