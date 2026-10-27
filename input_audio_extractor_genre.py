import librosa
import numpy as np

def audiofeatures(audio):
    features = []
    y, sr = librosa.load(audio, duration=30)
    c = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(float(np.mean(np.mean(c, axis=1))))
    features.append(float(np.var(c)))
    r = librosa.feature.rms(y=y)[0]
    features.append(float(np.mean(r)))
    features.append(float(np.var(r)))
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(sc)))
    features.append(float(np.var(sc)))
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(sb)))
    features.append(float(np.var(sb)))
    srf = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(srf)))
    features.append(float(np.var(srf)))
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.append(float(np.mean(zcr)))
    features.append(float(np.var(zcr)))
    ons = librosa.onset.onset_strength(y=y, sr=sr)
    tem = librosa.feature.tempo(onset_envelope=ons, sr=sr)
    features.append(float(np.mean(tem)))
    mfc = librosa.feature.mfcc(y=y, sr=sr)
    mm = np.mean(mfc, axis=1)
    mv = np.var(mfc, axis=1)
    for i, j in zip(mm, mv):
        features.append(float(i))
        features.append(float(j))
    return features