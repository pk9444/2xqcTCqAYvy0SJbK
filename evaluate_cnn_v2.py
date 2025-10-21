import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from train_cnn_v2 import FADCNN, MelDataset
import os
import seaborn as sns
sns.set_style("darkgrid")
# -----------------------------
# 1️⃣ Load Dataset Splits
# -----------------------------
df = pd.read_csv("data/fad_mel_dataset.csv")
df["path"] = df["path"].str.replace("\\", "/", regex=False)
ds_full = MelDataset(df, augment=False)

n = len(ds_full)
train_n = int(0.7 * n)
val_n   = int(0.15 * n)
test_n  = n - train_n - val_n

train_ds, val_ds, test_ds = torch.utils.data.random_split(ds_full, [train_n, val_n, test_n], generator=torch.Generator().manual_seed(42))
loaders = {
    "train": DataLoader(train_ds, batch_size=32, shuffle=False),
    "val":   DataLoader(val_ds, batch_size=32, shuffle=False),
    "test":  DataLoader(test_ds, batch_size=32, shuffle=False)
}

# -----------------------------
# 2️⃣ Load Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FADCNN().to(device)
model.load_state_dict(torch.load("models/cnn_v2.pt", map_location=device))
model.eval()

# -----------------------------
# 3️⃣ Evaluation Helper
# -----------------------------
def evaluate(loader):
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device).float()
            logits = model(xb).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            y_true.extend(yb.cpu().numpy().astype(int))
            y_pred.extend(preds)
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# -----------------------------
# 4️⃣ Evaluate all splits
# -----------------------------
metrics_all = {}
os.makedirs("results", exist_ok=True)

for split_name, loader in loaders.items():
    y_true, y_pred, y_prob = evaluate(loader)

    # --- metrics ---
    report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], output_dict=True)
    metrics_all[split_name] = {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"]
    }

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    plt.title(f"Confusion Matrix - {split_name.upper()}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/confmat_{split_name}.png")
    plt.close()

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {split_name.upper()}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/roc_{split_name}.png")
    plt.close()

    print(f"\n=== {split_name.upper()} SET ===")
    print(classification_report(y_true, y_pred, target_names=["Real","Fake"], digits=3))

# -----------------------------
# 5️⃣ Save overall snapshot
# -----------------------------
df_metrics = pd.DataFrame(metrics_all).T
df_metrics.to_csv("results/eval_snapshot.csv", index=True)
print("\n✅ Saved confusion matrices & ROC curves to /results/")
print(df_metrics)
