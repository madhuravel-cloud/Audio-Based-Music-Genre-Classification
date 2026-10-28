import os
import numpy as np
import pandas as pd
from input_audio_extractor_instrument import audiofeatures, audiofeatures_chunks
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Instrument-4.csv")

DROP_COLS = ["id", "clip name", "instrument_label"]

model = None
ALL_COLUMNS = []
TRAIN_COLUMNS = []
CLASSES = []

def train_instrument_model():
    global model, ALL_COLUMNS, TRAIN_COLUMNS, CLASSES
    data = pd.read_csv(CSV_PATH)
    data = data.drop(columns=DROP_COLS, errors="ignore")
    y = data["instrument"].astype(str)
    X = data.drop(columns=["instrument"])
    ALL_COLUMNS = list(X.columns)

    TRAIN_COLUMNS = [
        "cent mean",
        "bandwidth mean",
        "rolloff mean",
        "zcr mean",
    ] + [f"mfcc_mean {i}" for i in range(1, 20)]

    CLASSES = sorted(y.unique().tolist())

    model = make_pipeline(
        StandardScaler(),
        Normalizer(),
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
    )
    model.fit(X[TRAIN_COLUMNS], y)

train_instrument_model()

def instrument(music):
    chunk_features = audiofeatures_chunks(music)
    df_chunks = pd.DataFrame(chunk_features, columns=ALL_COLUMNS)
    probs = model.predict_proba(df_chunks[TRAIN_COLUMNS])
    avg_probs = np.mean(probs, axis=0)
    best_idx = int(np.argmax(avg_probs))
    pred = str(model.classes_[best_idx])
    avg_features = [float(x) for x in np.mean(chunk_features, axis=0)]
    return pred, avg_features

def add_instrument_data(features, label, filename="new_song.wav"):
    full_data = pd.read_csv(CSV_PATH, nrows=1)
    row = {}
    feat_map = dict(zip(ALL_COLUMNS, features))
    for col in full_data.columns:
        if col in feat_map:
            row[col] = feat_map[col]
        elif col == "instrument":
            row[col] = label
        elif col == "clip name":
            row[col] = filename
        else:
            row[col] = 0
    pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
    train_instrument_model()