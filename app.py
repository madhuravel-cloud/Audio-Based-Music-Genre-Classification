import os
import tempfile
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import main
import main2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeedbackRequest(BaseModel):
    filename: str
    genre: str
    instrument: str
    genre_features: list
    instrument_features: list

@app.post("/analyze")
def analyze(music: UploadFile = File(...)):
    suffix = os.path.splitext(music.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(music.file.read())
        path = temp.name

    try:
        y, sr = librosa.load(path, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        step = max(1, len(y) // 150)
        waveform = [round(float(val), 4) for val in y[::step][:150]]

        pred_g, feat_g = main.genre(path)
        pred_i, feat_i = main2.instrument(path)

        return {
            "filename": music.filename,
            "genre": pred_g,
            "instrument": pred_i,
            "genre_features": feat_g,
            "instrument_features": feat_i,
            "librosa": {
                "tempo": round(float(np.mean(tempo)), 1),
                "centroid": round(cent, 1),
                "rolloff": round(rolloff, 1),
                "zcr": round(zcr, 4),
                "waveform": waveform
            }
        }
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.post("/add-to-dataset")
def add_to_dataset(req: FeedbackRequest):
    main.add_genre_data(req.genre_features, req.genre, req.filename)
    main2.add_instrument_data(req.instrument_features, req.instrument, req.filename)
    return {"message": f"Added {req.filename} to dataset with Genre '{req.genre}' and Instrument '{req.instrument}'."}

@app.get("/")
def home():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, "index.html"))