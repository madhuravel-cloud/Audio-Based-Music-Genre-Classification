import os
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from main import genre
from main2 import instrument

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
def analyze(music: UploadFile = File(...)):
    suffix = os.path.splitext(music.filename or "")[1] or ".wav"
    path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(music.file.read())
            path = temp.name

        return {
            "filename": music.filename,
            "genre": str(genre(path)),
            "instrument": str(instrument(path)),
        }
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {
            "filename": music.filename,
            "genre": f"ERROR:\n{err}",
            "instrument": "See error above",
        }
    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.get("/")
def home():
    return {"message": "Music Analyzer API is running"}