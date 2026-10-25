import pandas as pd
from input_audio_extractor_instrument import audiofeatures
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("Instrument-4.csv")
data = data.drop(columns=["id", "clip name", "instrument_label"], errors="ignore")

y = data["instrument"]
X = data.drop(columns=["instrument"])
COLUMNS = X.columns

model = RandomForestClassifier(
    n_estimators=200, max_depth=20, class_weight="balanced", random_state=42, n_jobs=1
)
model.fit(X, y)
del data,X,y 

def instrument(music):
    new_features = audiofeatures(music)
    new_df = pd.DataFrame([new_features], columns=COLUMNS)
    return str(model.predict(new_df)[0])