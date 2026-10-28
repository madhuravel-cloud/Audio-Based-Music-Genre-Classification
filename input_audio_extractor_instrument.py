import librosa
import numpy as np
from scipy.signal import butter, sosfilt

def preprocess_chunk(y, sr):
    target_len = 66048
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    sos = butter(4, 195, btype="highpass", fs=sr, output="sos")
    y = sosfilt(sos, y).astype(np.float32)

    current_rms_sum = float(np.sum(librosa.feature.rms(y=y)[0]))
    if current_rms_sum > 1e-6:
        y = y * (2.2 / current_rms_sum)

    return y

def extract_from_signal(y, sr):
    features = []
    y = preprocess_chunk(y, sr)

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
    features.append(float(np.sum(rmse)))
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

def audiofeatures(audio):
    y, sr = librosa.load(audio, sr=22050, duration=30)
    y, _ = librosa.effects.trim(y, top_db=25)
    return extract_from_signal(y, sr)

def audiofeatures_chunks(audio):
    y, sr = librosa.load(audio, sr=22050, duration=30)
    y, _ = librosa.effects.trim(y, top_db=25)
    chunk_len = 66048
    if len(y) <= chunk_len:
        return [extract_from_signal(y, sr)]

    chunks = []
    step = chunk_len // 2
    for start in range(0, len(y) - chunk_len + 1, step):
        segment = y[start : start + chunk_len]
        if np.sqrt(np.mean(segment**2)) > 0.008:
            chunks.append(extract_from_signal(segment, sr))
        if len(chunks) >= 10:
            break

    if not chunks:
        chunks.append(extract_from_signal(y[:chunk_len], sr))
    return chunks