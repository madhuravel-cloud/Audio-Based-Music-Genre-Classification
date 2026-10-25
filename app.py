
import os
import tempfile

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
async def analyze(music: UploadFile = File(...)):
    suffix = os.path.splitext(music.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await music.read())
        path = temp.name

    try:
        return {
            "filename": music.filename,
            "genre": genre(path),
            "instrument": instrument(path),
        }
    finally:
        os.remove(path)


@app.get("/")
def home():
    return {"message": "Music Analyzer API is running"}
