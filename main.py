import os
import json
import numpy as np
import torch
import scipy.io.wavfile
from transformers import pipeline
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import replicate
import librosa
import requests
import joblib
import tensorflow_hub as hub
import tensorflow as tf

# -------------------------
# Config
# -------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
AUDIO_DIR = os.path.join(STATIC_DIR, "audio")
DATA_FILE = os.path.join(APP_DIR, "tracks.jsonl")

MODEL_DIR = os.path.join(APP_DIR, "model")
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"

os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------------
# Load models & encoders once
# -------------------------
print("🔄 Loading models and encoders...")

vggish = hub.load(VGGISH_MODEL_URL)
genre_model = joblib.load(os.path.join(MODEL_DIR, "genre_classifier.pkl"))
lang_model = joblib.load(os.path.join(MODEL_DIR, "lang_classifier.pkl"))
genre_encoder = joblib.load(os.path.join(MODEL_DIR, "genre_encoder.pkl"))
lang_encoder = joblib.load(os.path.join(MODEL_DIR, "lang_encoder.pkl"))

print("✅ All models and encoders loaded successfully.")

SAMPLE_RATE = 16000
DURATION = 40

# -------------------------
# FastAPI app setup
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------
# Page Routes
# -------------------------
@app.get("/")
def read_home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/start_composing")
def read_start_composing():
    html_path = os.path.join(STATIC_DIR, "start_composing.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="start_composing.html not found")


@app.get("/predict_page")
def read_predict_page():
    html_path = os.path.join(STATIC_DIR, "predict.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="predict.html not found")

# -------------------------
# Music Generation (Local)
# -------------------------
MUSIC_PIPE = None

def get_music_pipeline():
    global MUSIC_PIPE
    if MUSIC_PIPE is None:
        device = 0 if torch.cuda.is_available() else -1
        print("🔄 Loading local MusicGen model (facebook/musicgen-small)...")
        MUSIC_PIPE = pipeline("text-to-audio", "facebook/musicgen-small", device=device)
        print("✅ MusicGen ready on", "GPU" if device == 0 else "CPU")
    return MUSIC_PIPE


@app.post("/generate_music_local")
async def generate_music_local(
    prompt: str = Form(...),
    genre: str = Form("ambient"),
    mood: str = Form("calm"),
    duration: int = Form(10)
):
    try:
        pipe = get_music_pipeline()
        duration = min(max(int(duration), 1), 40)
        full_prompt = f"{genre} {mood} style, {prompt}"
        print(f"🎵 Generating: {full_prompt} ({duration}s)")

        output = pipe(full_prompt, forward_params={"max_new_tokens": int(duration * 50)})
        if not isinstance(output, dict) or "audio" not in output:
            return JSONResponse({"error": "Invalid response from MusicGen"}, status_code=500)

        audio_data = output["audio"]
        sampling_rate = int(output["sampling_rate"])
        fname = f"song_{np.random.randint(100000,999999)}.wav"
        fpath = os.path.join(AUDIO_DIR, fname)

        scipy.io.wavfile.write(fpath, rate=sampling_rate, data=(audio_data * 32767).astype(np.int16))
        meta = {
            "id": fname.replace(".wav", ""),
            "file_url": f"/static/audio/{fname}",
            "filename": fname,
            "genre": genre,
            "mood": mood,
            "prompt": prompt,
            "duration": duration,
        }

        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        return {"status": "success", "file_url": meta["file_url"], "meta": meta}

    except Exception as e:
        print("Error generate local:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# -------------------------
# Genre/Language Prediction
# -------------------------
@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    temp_path = os.path.join(AUDIO_DIR, f"temp_{file.filename}")

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        if len(y) < SAMPLE_RATE * DURATION:
            y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)), mode="constant")
        waveform = tf.convert_to_tensor(y, dtype=tf.float32)

        embedding = vggish(waveform).numpy()
        feature_vector = np.mean(embedding, axis=0).reshape(1, -1)

        genre_pred = genre_model.predict(feature_vector)
        lang_pred = lang_model.predict(feature_vector)

        genre_label = genre_encoder.inverse_transform(genre_pred)[0]
        lang_label = lang_encoder.inverse_transform(lang_pred)[0]

        return {
            "status": "success",
            "predicted_genre": genre_label,
            "predicted_language": lang_label
        }

    except Exception as e:
        print("Prediction error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------
# File Management (upload, delete, import)
# -------------------------
@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...), genre: str = Form("unknown"), mood: str = Form("unknown")):
    try:
        fname = f"upload_{np.random.randint(100000,999999)}_{file.filename}"
        fpath = os.path.join(AUDIO_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(await file.read())
        meta = {
            "id": fname.replace(".", "_"),
            "file_url": f"/static/audio/{fname}",
            "filename": fname,
            "genre": genre,
            "mood": mood,
            "prompt": "uploaded"
        }
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        return {"status": "success", "meta": meta}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/add_from_drive")
async def add_from_drive(drive_url: str = Form(...), genre: str = Form("unknown"), mood: str = Form("unknown")):
    try:
        if "drive.google.com" in drive_url:
            file_id = drive_url.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        else:
            download_url = drive_url

        r = requests.get(download_url, stream=True, timeout=60)
        if r.status_code != 200:
            raise Exception("Download failed")

        fname = f"drive_{np.random.randint(100000,999999)}.mp3"
        fpath = os.path.join(AUDIO_DIR, fname)
        with open(fpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        meta = {
            "id": fname.replace(".", "_"),
            "file_url": f"/static/audio/{fname}",
            "filename": fname,
            "genre": genre,
            "mood": mood,
            "prompt": "from_drive"
        }
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        return {"status": "success", "meta": meta}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -------------------------
# Library & Playlist Endpoints
# -------------------------
@app.get("/tracks")
def get_all_tracks():
    """Return all saved tracks for Library."""
    if not os.path.exists(DATA_FILE):
        return {"status": "success", "tracks": []}

    tracks = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tracks.append(json.loads(line.strip()))
            except:
                continue

    return {"status": "success", "tracks": tracks}


@app.get("/get_tracks")
def get_tracks():
    """Return grouped tracks for Playlist."""
    if not os.path.exists(DATA_FILE):
        return {"tracks": [], "grouped": {}}

    tracks = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tracks.append(json.loads(line.strip()))
            except:
                continue

    grouped = {}
    for t in tracks:
        grouped.setdefault(t["genre"], []).append(t)

    return {"tracks": tracks, "grouped": grouped}


@app.post("/delete_track")
async def delete_track(track_id: str = Form(...)):
    """Delete a track from JSON and remove audio file."""
    if not os.path.exists(DATA_FILE):
        return {"status": "error", "message": "No data file found"}

    new_tracks = []
    deleted = False
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line.strip())
                if t.get("id") == track_id:
                    deleted = True
                    fpath = os.path.join(AUDIO_DIR, t.get("filename", ""))
                    if os.path.exists(fpath):
                        os.remove(fpath)
                else:
                    new_tracks.append(t)
            except:
                continue

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for t in new_tracks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    return {"status": "deleted" if deleted else "not_found"}

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server on http://127.0.0.1:8018")
    uvicorn.run(app, host="127.0.0.1", port=8018)
