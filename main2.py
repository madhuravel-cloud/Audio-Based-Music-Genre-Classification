import os
import pandas as pd
from input_audio_extractor_instrument import audiofeatures
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Instrument-4.csv")

DROP_COLS = ["id", "clip name", "instrument_label"]

model = None
COLUMNS = []
CLASSES = []

def train_instrument_model():
    global model, COLUMNS, CLASSES
    data = pd.read_csv(CSV_PATH)
    data = data.drop(columns=DROP_COLS, errors="ignore")
    y = data["instrument"].astype(str)
    X = data.drop(columns=["instrument"])
    COLUMNS = list(X.columns)
    CLASSES = sorted(y.unique().tolist())

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
    model.fit(X, y)

train_instrument_model()

def instrument(music):
    new_features = audiofeatures(music)
    new_df = pd.DataFrame([new_features], columns=COLUMNS)
    return str(model.predict(new_df)[0]), new_features

def add_instrument_data(features, label, filename="new_song.wav"):
    full_data = pd.read_csv(CSV_PATH, nrows=1)
    row = {}
    feat_map = dict(zip(COLUMNS, features))
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