from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch, torchaudio, numpy as np, os, io, base64, matplotlib.pyplot as plt
from train_cnn_v2 import FADCNN
import seaborn as sns
sns.set_theme(style="darkgrid")  #  add this line
from environs import load_dotenv
from openai import OpenAI

#  Set your API key (better: load from env var)
load_dotenv()

x = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = x
client = OpenAI(api_key=x)
# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Fake Audio Detection Web App", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_audio(file_path, target_sr=16000, n_mels=64, target_frames=128):
    wav, sr = torchaudio.load(file_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.mean(dim=0, keepdim=True)
    wav = wav / (wav.abs().max() + 1e-9)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=256, n_mels=n_mels
    )(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel = mel_db.squeeze(0).numpy()
    n_mels, T = mel.shape
    if T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")
    elif T > target_frames:
        mel = mel[:, :target_frames]
    return mel.astype(np.float32)

#-----------------------------
# Extract Features
#-----------------------------
def extract_features(wav_tensor, sr):
    """
    wav_tensor: (T,) mono, normalized to [-1, 1]
    """
    wav = wav_tensor.cpu().numpy() if hasattr(wav_tensor, "cpu") else wav_tensor
    T = len(wav)
    dur = T / sr

    # Energy (RMS, dBFS-ish)
    rms = np.sqrt(np.mean(wav**2) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)

    # Silence ratio (below -40 dBFS relative)
    thr = (10 ** (-40/20))  # relative linear threshold
    silence_ratio = float(np.mean(np.abs(wav) < thr))

    # Short-time features via STFT
    spec = np.abs(np.fft.rfft(wav, n=2048))
    freqs = np.fft.rfftfreq(2048, d=1.0/sr)
    spec = spec + 1e-9
    spec_norm = spec / spec.sum()

    # Spectral centroid & bandwidth (simple global)
    centroid_hz = float((freqs * spec_norm).sum())
    spread = float(np.sqrt(((freqs - centroid_hz) ** 2 * spec_norm).sum()))

    # MFCC (with torchaudio)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=13, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 64}
    )
    mfcc = mfcc_transform(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()  # (13, frames)
    mfcc_mean = np.mean(mfcc, axis=1).tolist()
    mfcc_std  = np.std(mfcc, axis=1).tolist()

    return {
        "duration_sec": round(dur, 3),
        "rms_db": round(rms_db, 2),
        "silence_ratio": round(silence_ratio, 3),
        "spectral_centroid_hz": round(centroid_hz, 1),
        "spectral_spread_hz": round(spread, 1),
        "mfcc_mean": [round(v, 3) for v in mfcc_mean],
        "mfcc_std":  [round(v, 3) for v in mfcc_std],
    }

def summarize_with_gpt(features: dict, label: str, prob: float):
    """
    Calls o3-mini to produce a concise explanation using features + model prediction.
    """
    prompt = f"""
You are helping explain a fake-audio detector’s result to a user.
Given acoustic features and the model’s prediction, write a concise 2–3 sentence summary
describing what the audio likely sounds like and why it may be classified as {label}.
Avoid jargon; keep it intuitive. Do not repeat the raw numbers verbatim—interpret them.

Prediction: {label} (confidence {prob:.3f})
Features (JSON):
{features}
"""
    try:
        resp = client.chat.completions.create(
            model="o3-mini",  # if unavailable, try "gpt-4o-mini"
            messages=[
                {"role": "system", "content": "You explain audio traits clearly to non-experts."},
                {"role": "user", "content": prompt},
            ],
            
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Explanation unavailable: {e})"

# -----------------------------
# Model loading
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/cnn_v2.pt"
model = FADCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f" Model loaded from {model_path}")

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})




#==============================BASELINE WORKING FUNCTION====================================#
# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request, file: UploadFile = File(...)):
#     try:
#         # 1️⃣ Save uploaded file temporarily
#         tmp_path = f"temp_{file.filename}"
#         with open(tmp_path, "wb") as f:
#             f.write(await file.read())

#         # 2️⃣ Load waveform
#         wav, sr = torchaudio.load(tmp_path)
#         wav = wav.mean(dim=0)  # mono
#         wav = wav / (wav.abs().max() + 1e-9)
#         waveform = wav.cpu().numpy().tolist()

#         # 3️⃣ Preprocess to mel spectrogram
#         mel = preprocess_audio(tmp_path)
#         mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

#         # 4️⃣ Model inference
#         with torch.no_grad():
#             logits = model(mel_tensor)
#             prob = torch.sigmoid(logits).item()
#         label = "FAKE" if prob >= 0.5 else "REAL"

#         # 5️⃣ Clean up temp file
#         os.remove(tmp_path)

#         # 6️⃣ Render HTML with both waveform & mel spectrogram
#         return templates.TemplateResponse(
#             "index.html",
#             {
#                 "request": request,
#                 "result": {
#                     "filename": file.filename,
#                     "label": label,
#                     "prob": round(prob, 3),
#                     "mel_data": mel.tolist(),
#                     "waveform": waveform,
#                 },
#             },
#         )

#     except Exception as e:
#         return templates.TemplateResponse(
#             "index.html",
#             {"request": request, "error": str(e), "result": None},
#         )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        tmp_path = f"temp_{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        # Load & normalize waveform
        wav, sr = torchaudio.load(tmp_path)      # (C, T)
        wav = wav.mean(dim=0)                    # mono (T,)
        wav = wav / (wav.abs().max() + 1e-9)
        waveform = wav.cpu().numpy().tolist()

        # Preprocess to mel for CNN
        mel = preprocess_audio(tmp_path)
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()
        label = "FAKE" if prob >= 0.5 else "REAL"

        # === Smart explanation ===
        feats = extract_features(wav, sr)
        explanation = summarize_with_gpt(feats, label, prob)

        os.remove(tmp_path)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": {
                    "filename": file.filename,
                    "label": label,
                    "prob": round(prob, 3),
                    "mel_data": mel.tolist(),
                    "waveform": waveform,
                    "explanation": explanation,
                },
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e), "result": None},
        )


# # Run via:
# uvicorn app:app --reload --port 8000
