import torch, torchaudio, numpy as np, matplotlib.pyplot as plt
from train_cnn_v2 import FADCNN
import os

# -----------------------------
# 1️⃣ Mel-Spectrogram Preprocessing
# -----------------------------
def preprocess_audio(path, target_sr=16000, n_mels=64, target_frames=128):
    """
    Loads a .wav file, converts to mel-spectrogram,
    and pads/crops to the same input size as used in training.
    """
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # Mono and normalize
    wav = wav.mean(dim=0, keepdim=True)
    wav = wav / (wav.abs().max() + 1e-9)

    # Mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=256, n_mels=n_mels
    )(wav)

    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel = mel_db.squeeze(0).numpy()

    # Pad or crop
    n_mels, T = mel.shape
    if T < target_frames:
        mel = np.pad(mel, ((0,0), (0,target_frames - T)), mode="constant")
    elif T > target_frames:
        mel = mel[:, :target_frames]

    return mel.astype(np.float32)


# -----------------------------
# 2️⃣ Load Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/cnn_v2.pt"

model = FADCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Loaded model from {model_path}")

# -----------------------------
# 3️⃣ Test Audio File
# -----------------------------
# Update this path for your test file:

'''
"data/timit/test/fake/106_0670.wav", 
"data/commonvoice/real/common_voice_en_41269547.wav",
"data/unseen/test_fake_sample_1.wav"
'''
test_file = "data/timit/test/fake/0_1320.wav"
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test file not found: {test_file}")

mel = preprocess_audio(test_file)
x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)  # shape: (1, 1, 64, 128)

# -----------------------------
# 4️⃣ Inference
# -----------------------------
with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits).item()

pred_label = "FAKE" if prob >= 0.5 else "REAL"
print("\n🎧 Prediction Result")
print(f"File       : {test_file}")
print(f"Prediction : {pred_label}")
print(f"Confidence : {prob:.4f}")

# -----------------------------
# 5️⃣ Optional: Visualize Mel
# -----------------------------
plt.figure(figsize=(8, 4))
plt.imshow(mel, aspect="auto", origin="lower", cmap="magma")
plt.title(f"Predicted: {pred_label} ({prob:.3f})")
plt.xlabel("Frames"); plt.ylabel("Mel Bands")
plt.tight_layout()
plt.show()
