import os
import pandas as pd
from input_audio_extractor_genre import audiofeatures
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "features_30_sec.csv")

data = pd.read_csv(CSV_PATH)
data = data.drop(
    columns=[
        "filename", "length", "id", "clip name", "genre_label",
        "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var"
    ],
    errors="ignore"
)

target_col = "label" if "label" in data.columns else "genre"
y = data[target_col]
X = data.drop(columns=[target_col])
COLUMNS = X.columns

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)
model.fit(X, y)
del data, X, y


def genre(music):
    new_features = audiofeatures(music)
    new_df = pd.DataFrame([new_features], columns=COLUMNS)
    return str(model.predict(new_df)[0])