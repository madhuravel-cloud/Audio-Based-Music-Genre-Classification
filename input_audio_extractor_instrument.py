import librosa
import numpy as np

def audiofeatures(audio):
    features = []
    y, sr = librosa.load(audio, duration=30)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(np.mean(tempo)))
    features.append(float(np.sum(beat_frames)))
    features.append(float(np.mean(beat_frames)))
    features.append(float(np.var(beat_frames)))
    features.append(float(np.std(beat_frames)))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(float(np.mean(cent)))
    features.append(float(np.var(cent)))
    features.append(float(np.std(cent)))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(float(np.mean(bandwidth)))
    features.append(float(np.var(bandwidth)))
    features.append(float(np.std(bandwidth)))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(float(np.mean(rolloff)))
    features.append(float(np.var(rolloff)))
    features.append(float(np.std(rolloff)))

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features.append(float(np.sum(zcr)))
    features.append(float(np.mean(zcr)))

    rmse = librosa.feature.rms(y=y)[0]
    features.append(float(np.mean(rmse)))
    features.append(float(np.sum(y**2)))

    mfc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    mm = np.mean(mfc, axis=1)
    ms = np.sum(mfc, axis=1)
    md = np.std(mfc, axis=1)
    mv = np.var(mfc, axis=1)

    for i in mm:
        features.append(float(i))
    for i in ms:
        features.append(float(i))
    for i in md:
        features.append(float(i))
    for i in mv:
        features.append(float(i))

    return features