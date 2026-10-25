import pandas as pd
from input_audio_extractor_instrument import audiofeatures
from sklearn.ensemble import RandomForestClassifier

def instrument(music):
    data = pd.read_csv(
        r"C:\Users\user\OneDrive\Desktop\audio-based music-genre classifier\Instrument-4.csv"
    )
    data = data.drop(columns=["id", "clip name", "instrument_label"], errors="ignore")

    y = data["instrument"]
    X = data.drop(columns=["instrument"])

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    new_features = audiofeatures(music)
    new_df = pd.DataFrame([new_features], columns=X.columns)
    
    return str(model.predict(new_df)[0])
