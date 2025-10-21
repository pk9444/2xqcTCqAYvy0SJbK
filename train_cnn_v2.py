import os, time, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
import random

# -----------------------------
# 1. Augment the Dataset
# -----------------------------
class MelDataset(Dataset):
    def __init__(self, df, target_frames=128, augment=True):
        self.df = df.reset_index(drop=True)
        self.target_frames = target_frames
        self.cache = {}
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, label = self.df.iloc[idx]
        if path not in self.cache:
            self.cache[path] = np.load(path)
        mel = np.array(self.cache[path])

        # --- ensure correct shape ---
        if mel.ndim == 3:
            mel = mel.squeeze(0)
        elif mel.ndim == 1:
            mel = mel.reshape(64, -1)
        elif mel.ndim != 2:
            raise ValueError(f"Unexpected mel shape: {mel.shape} for file {path}")

        # --- pad or crop to fixed length ---
        n_mels, T = mel.shape
        if T < self.target_frames:
            mel = np.pad(mel, ((0, 0), (0, self.target_frames - T)), mode="constant")
        elif T > self.target_frames:
            mel = mel[:, :self.target_frames]

        # --- augmentations ---
        if self.augment:
            mel = self._augment(mel)

        # x = torch.tensor(mel).unsqueeze(0)         # (1, n_mels, T)
        # y = torch.tensor(label, dtype=torch.float32)

        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    def _augment(self, mel):
        # Random time masking
        if random.random() < 0.3:
            t = random.randint(0, mel.shape[1] - 10)
            mel[:, t:t + 10] = 0
        # Random frequency masking
        if random.random() < 0.3:
            f = random.randint(0, mel.shape[0] - 8)
            mel[f:f + 8, :] = 0
        # Add small Gaussian noise
        if random.random() < 0.3:
            mel = mel + 0.01 * np.random.randn(*mel.shape)
        return mel


# ----------------------------------
# 2. Create the CNN model blueprint
# ----------------------------------
class FADCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------
# 3. Training Function
# -----------------------------

def train():
    
    # 3.1. Load the Dataset and apply augmentation using the MelDataSet blueprint
    df = pd.read_csv("data/fad_mel_dataset.csv")
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    ds = MelDataset(df)
    n = len(ds)

    # 3.2. Split into the training, validation and test sets
    train_n = int(0.7 * n)
    val_n   = int(0.15 * n)
    test_n  = n - train_n - val_n
    train_ds, val_ds, test_ds = random_split(ds, [train_n, val_n, test_n])

    # 3.3. Define the model configs using CNN blueprint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FADCNN().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    best_f1, patience, counter = 0.0, 3, 0

    # 3.4. run the CNN on train and val sets for the optimal number of epochs 
    def run_epoch(loader, train=False):
        model.train(mode=train)
        total_loss, preds, labels = 0.0, [], []
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device).float()
            with torch.set_grad_enabled(train):
                logits = model(xb).squeeze(1)

                # --- label smoothing ---
                yb_smooth = yb * 0.9 + 0.05

                loss = crit(logits, yb_smooth)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.extend((probs >= 0.5).astype(int))
            labels.extend(yb.cpu().numpy().astype(int))

        return total_loss / len(loader), f1_score(labels, preds)

    print("🚀 Starting training...")
    for epoch in range(30):
        tr_loss, tr_f1 = run_epoch(train_loader, True)
        va_loss, va_f1 = run_epoch(val_loader, False)

        print(f"Epoch {epoch+1:02d} | Train {tr_loss:.3f}/{tr_f1:.3f} | Val {va_loss:.3f}/{va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1, counter = va_f1, 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/cnn_v2.pt")
            print("✅ Saved best → models/cnn_v2.pt")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping (no F1 improvement).")
                break

    print("✅ Training complete.")
    return model, train_loader, val_loader, device


if __name__ == "__main__":
    start = time.time()
    model, train_loader, val_loader, device = train()
    print(f"⏱️ Done in {(time.time()-start)/60:.1f} min")
