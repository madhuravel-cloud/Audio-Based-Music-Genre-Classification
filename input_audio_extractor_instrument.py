import librosa
import numpy as np

def audiofeatures(audio):
    features=[]
    y,sr=librosa.load(audio)

    tempo,beat_frames=librosa.beat.beat_track(y=y,sr=sr)
    features.append(np.mean(tempo))
    features.append(np.sum(beat_frames))
    features.append(np.mean(beat_frames))
    features.append(np.var(beat_frames))
    features.append(np.std(beat_frames))

    cent=librosa.feature.spectral_centroid(y=y,sr=sr)[0]
    features.append(np.mean(cent))
    features.append(np.var(cent))
    features.append(np.std(cent))

    bandwidth=librosa.feature.spectral_bandwidth(y=y,sr=sr)[0]
    features.append(np.mean(bandwidth))
    features.append(np.var(bandwidth))
    features.append(np.std(bandwidth))

    rolloff=librosa.feature.spectral_rolloff(y=y,sr=sr)[0]
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))
    features.append(np.std(rolloff))

    zcr=librosa.feature.zero_crossing_rate(y=y)[0]
    features.append(np.sum(zcr))
    features.append(np.mean(zcr))

    rmse=librosa.feature.rms(y=y)[0]
    features.append(np.mean(rmse))
    features.append(np.sum(y**2))

    mfc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)

    mm=np.mean(mfc,axis=1)
    ms=np.sum(mfc,axis=1)
    md=np.std(mfc,axis=1)
    mv=np.var(mfc,axis=1)

    for i in mm:
        features.append(i)

    for i in ms:
        features.append(i)

    for i in md:
        features.append(i)

    for i in mv:
        features.append(i)

    return features