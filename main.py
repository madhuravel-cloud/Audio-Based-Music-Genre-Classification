import os
import pandas as pd
from input_audio_extractor_genre import audiofeatures
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "features_30_sec.csv")

DROP_COLS = [
    "filename", "length", "id", "clip name", "genre_label",
    "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var"
]

model = None
COLUMNS = []
TARGET_COL = "label"
CLASSES = []

def train_genre_model():
    global model, COLUMNS, TARGET_COL, CLASSES
    data = pd.read_csv(CSV_PATH)
    data = data.drop(columns=DROP_COLS, errors="ignore")
    TARGET_COL = "label" if "label" in data.columns else "genre"
    y = data[TARGET_COL].astype(str)
    X = data.drop(columns=[TARGET_COL])
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

train_genre_model()

def genre(music):
    new_features = audiofeatures(music)
    new_df = pd.DataFrame([new_features], columns=COLUMNS)
    return str(model.predict(new_df)[0]), new_features

def add_genre_data(features, label, filename="new_song.wav"):
    full_data = pd.read_csv(CSV_PATH, nrows=1)
    row = {}
    feat_map = dict(zip(COLUMNS, features))
    for col in full_data.columns:
        if col in feat_map:
            row[col] = feat_map[col]
        elif col == TARGET_COL:
            row[col] = label
        elif col in ["filename", "clip name"]:
            row[col] = filename
        else:
            row[col] = 0
    pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
    train_genre_model()