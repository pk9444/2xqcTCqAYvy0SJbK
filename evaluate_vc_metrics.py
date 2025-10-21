
import os
import torch
import torchaudio
import pandas as pd
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
# 1. Load Wav2Vec2 ASR
# ============================================================
print("Loading ASR model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
model.eval()

# ============================================================
# 2. Transcription helper
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
# 3. Evaluate all test files
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
        print(f"Skipping {fname}: {e}")
        continue

    records.append({
        "filename": fname,
        "relative_wer": rel_wer,
        "real_transcript": text_real,
        "fake_transcript": text_fake
    })

df_wer = pd.DataFrame(records)
df_wer.to_csv("results/testset_relative_wer.csv", index=False)
print(f"Saved detailed WER results → results/testset_relative_wer.csv")

# ============================================================
# 4. Average WER and summary TXT
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
# 5. Plot WER distribution
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
print("Saved WER distribution plot → results/wer_distribution.png")

# ============================================================
# 6. Optional: Correlate WER with CNN Fake Probability
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
    print("Saved correlation plot => results/wer_vs_cnn_prob.png")

    # save merged df for later analysis
    df_merge.to_csv("results/wer_cnn_correlation.csv", index=False)
    print("Saved merged WER CNN dataframe => results/wer_cnn_correlation.csv")
else:
    print("CNN predictions CSV not found. Skipping correlation plot.")
