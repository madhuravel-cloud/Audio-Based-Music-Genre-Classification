import os
import pandas as pd
import streamlit as st
from input_audio_extractor_genre import audiofeatures
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "features_30_sec.csv")


@st.cache_resource
def load_genre_model():
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
    cols = list(X.columns)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
    model.fit(X, y)
    return model, cols


def genre(music):
    model, columns = load_genre_model()
    new_features = list(audiofeatures(music))
    if hasattr(audiofeatures, "__globals__"):
        for val in audiofeatures.__globals__.values():
            if isinstance(val, list) and len(val) >= len(columns):
                val.clear()
    new_features = new_features[-len(columns):]
    new_df = pd.DataFrame([new_features], columns=columns)
    return str(model.predict(new_df)[0])