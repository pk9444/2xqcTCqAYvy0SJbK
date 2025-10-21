

# import torch, torchaudio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# import warnings
# warnings.filterwarnings('ignore')

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)

# def transcribe_audio(file_path):
#     wav, sr = torchaudio.load(file_path)
#     wav = wav.mean(dim=0)
#     if sr != 16000:
#         wav = torchaudio.functional.resample(wav, sr, 16000)
#     inputs = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
#     with torch.no_grad():
#         logits = model(inputs).logits
#     ids = torch.argmax(logits, dim=-1)
#     text = processor.decode(ids[0])
#     return text.lower().strip()

# real_file = "data/timit/test/real/9_1237.wav"
# fake_file = "data/timit/test/fake/9_1237.wav"

# print("🎧 Transcribing real...")
# print(transcribe_audio(real_file))
# print("\n🎭 Transcribing fake...")
# print(transcribe_audio(fake_file))

# from jiwer import wer

# real_text = transcribe_audio(real_file)
# fake_text = transcribe_audio(fake_file)

# rel_wer = wer(real_text, fake_text)
# print(f"\nRelative WER (Fake vs Real ASR): {rel_wer:.3f}")

#-------------------------------------------------------------------------------------------------#
# import os
# import torch
# import torchaudio
# import pandas as pd
# from tqdm import tqdm
# from jiwer import wer
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# # ============================================================
# # CONFIG
# # ============================================================
# REAL_DIR = "data/timit/test/real"
# FAKE_DIR = "data/timit/test/fake"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ============================================================
# # 1️⃣ Load ASR model
# # ============================================================
# print("🔤 Loading Wav2Vec2 ASR model...")
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
# model.eval()

# # ============================================================
# # 2️⃣ Transcription helper
# # ============================================================
# def transcribe_audio(path):
#     wav, sr = torchaudio.load(path)
#     wav = wav.mean(dim=0)  # mono
#     if sr != 16000:
#         wav = torchaudio.functional.resample(wav, sr, 16000)
#     input_values = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
#     with torch.no_grad():
#         logits = model(input_values).logits
#     pred_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.decode(pred_ids[0])
#     return transcription.lower().strip()

# # ============================================================
# # 3️⃣ Iterate and compute WER
# # ============================================================
# results = []

# for fname in tqdm(os.listdir(REAL_DIR), desc="Evaluating test samples"):
#     if not fname.endswith(".wav"):
#         continue

#     real_path = os.path.join(REAL_DIR, fname)
#     fake_path = os.path.join(FAKE_DIR, fname)
#     txt_path  = real_path.replace(".wav", ".txt")

#     if not os.path.exists(fake_path):
#         print(f"⚠️ Missing fake counterpart for {fname}")
#         continue

#     try:
#         pred_real = transcribe_audio(real_path)
#         pred_fake = transcribe_audio(fake_path)
#     except Exception as e:
#         print(f"⚠️ Skipping {fname}: {e}")
#         continue

#     # Case 1: If ground truth text exists
#     if os.path.exists(txt_path):
#         with open(txt_path, "r") as f:
#             gt_text = f.read().strip().lower()
#         wer_real = wer(gt_text, pred_real)
#         wer_fake = wer(gt_text, pred_fake)
#         rel_wer = wer(pred_real, pred_fake)  # still useful to log
#     else:
#         # Case 2: No transcripts — use relative WER
#         wer_real, wer_fake = None, None
#         rel_wer = wer(pred_real, pred_fake)

#     results.append({
#         "filename": fname,
#         "gt_exists": os.path.exists(txt_path),
#         "wer_real": wer_real,
#         "wer_fake": wer_fake,
#         "relative_wer": rel_wer,
#         "pred_real": pred_real,
#         "pred_fake": pred_fake
#     })

# # ============================================================
# # 4️⃣ Aggregate and Save
# # ============================================================
# df = pd.DataFrame(results)
# os.makedirs("results", exist_ok=True)
# df.to_csv("results/testset_wer_results.csv", index=False)

# # Compute averages
# avg_wer_real = df["wer_real"].dropna().mean() if not df["wer_real"].dropna().empty else float("nan")
# avg_wer_fake = df["wer_fake"].dropna().mean() if not df["wer_fake"].dropna().empty else float("nan")
# avg_rel_wer  = df["relative_wer"].mean()

# # ============================================================
# # 5️⃣ Print report
# # ============================================================
# print("\n📊 Voice Cloning Evaluation Metrics (Test Set)")
# print("------------------------------------------------")
# print(f"Average WER (Real vs GT):  {avg_wer_real:.3f}" if not pd.isna(avg_wer_real) else "Average WER (Real vs GT): N/A")
# print(f"Average WER (Fake vs GT):  {avg_wer_fake:.3f}" if not pd.isna(avg_wer_fake) else "Average WER (Fake vs GT): N/A")
# print(f"Average Relative WER (Fake vs Real): {avg_rel_wer:.3f}")
# print("------------------------------------------------")
# print("Detailed per-file results saved → results/testset_wer_results.csv")


import os
import io
import torch
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
sns.set_style("darkgrid")
# ============================================================
# CONFIG
# ============================================================
REAL_DIR = "data/timit/test/real"
FAKE_DIR = "data/timit/test/fake"
CNN_RESULTS_CSV = "results/test_predictions.csv"  # from evaluate_cnn_v2.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)

# ============================================================
# 1️⃣ Load Wav2Vec2 ASR
# ============================================================
print("🔤 Loading ASR model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
model.eval()

# ============================================================
# 2️⃣ Transcription helper
# ============================================================
def transcribe_audio(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    input_values = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(pred_ids[0])
    return transcription.lower().strip()

# ============================================================
# 3️⃣ Evaluate all test files
# ============================================================
records = []

for fname in tqdm(os.listdir(REAL_DIR), desc="Evaluating WER across test set"):
    if not fname.endswith(".wav"):
        continue
    real_path = os.path.join(REAL_DIR, fname)
    fake_path = os.path.join(FAKE_DIR, fname)
    if not os.path.exists(fake_path):
        continue

    try:
        text_real = transcribe_audio(real_path)
        text_fake = transcribe_audio(fake_path)
        rel_wer = wer(text_real, text_fake)
    except Exception as e:
        print(f"⚠️ Skipping {fname}: {e}")
        continue

    records.append({
        "filename": fname,
        "relative_wer": rel_wer,
        "real_transcript": text_real,
        "fake_transcript": text_fake
    })

df_wer = pd.DataFrame(records)
df_wer.to_csv("results/testset_relative_wer.csv", index=False)
print(f"✅ Saved detailed WER results → results/testset_relative_wer.csv")

# ============================================================
# 4️⃣ Average WER and summary TXT
# ============================================================
avg_rel_wer = df_wer["relative_wer"].mean()
summary_text = (
    "Voice Cloning Evaluation Metrics (Test Set)\n"
    "------------------------------------------------\n"
    "Average WER (Real vs GT): N/A\n"
    "Average WER (Fake vs GT): N/A\n"
    f"Average Relative WER (Fake vs Real): {avg_rel_wer:.3f}\n"
    "------------------------------------------------\n"
)

with open("results/voice_cloning_eval_summary.txt", "w") as f:
    f.write(summary_text)
print(summary_text)

# ============================================================
# 5️⃣ Plot WER distribution
# ============================================================
sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(8,5))
sns.histplot(df_wer["relative_wer"], bins=20, kde=True, color="skyblue")
plt.title("WER Distribution (Fake vs Real Speech)")
plt.xlabel("Relative Word Error Rate")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("results/wer_distribution.png", dpi=300)
plt.close()
print("📈 Saved WER distribution plot → results/wer_distribution.png")

# ============================================================
# 6️⃣ Optional: Correlate WER with CNN Fake Probability
# ============================================================
if os.path.exists(CNN_RESULTS_CSV):
    df_cnn = pd.read_csv(CNN_RESULTS_CSV)
    df_cnn["basename"] = df_cnn["path"].apply(lambda x: os.path.basename(str(x)).replace(".npy", ".wav"))
    df_merge = pd.merge(df_wer, df_cnn, left_on="filename", right_on="basename", how="inner")

    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df_merge, x="relative_wer", y="prob", alpha=0.6)
    sns.regplot(data=df_merge, x="relative_wer", y="prob", scatter=False, color="red", ci=None)
    plt.title("Correlation between WER and CNN Fake Probability")
    plt.xlabel("Relative WER (Fake vs Real)")
    plt.ylabel("CNN Fake Probability")
    plt.tight_layout()
    plt.savefig("results/wer_vs_cnn_prob.png", dpi=300)
    plt.close()
    print("📈 Saved correlation plot → results/wer_vs_cnn_prob.png")

    # save merged df for later analysis
    df_merge.to_csv("results/wer_cnn_correlation.csv", index=False)
    print("✅ Saved merged WER–CNN dataframe → results/wer_cnn_correlation.csv")
else:
    print("⚠️ CNN predictions CSV not found. Skipping correlation plot.")
