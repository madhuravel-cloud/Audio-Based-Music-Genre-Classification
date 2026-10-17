from librosa.util.files import librosa_version
from ctypes import LibraryLoader
import librosa
import librosa.display
import matplotlib.pyplot as plt  
import numpy as np
file=librosa.example("trumpet")
features=[]
y,sr=librosa.load(file)
c=librosa.feature.chroma_stft(y=y,sr=sr)
features.append(np.mean(np.mean(c,axis=1)))
features.append(np.var(np.var(c,axis=1)))
r=librosa.feature.rms(y=y)[0]
features.append(np.mean(r))
features.append(np.var(r))
sc=librosa.feature.spectral_centroid(y=y,sr=sr)
features.append(np.mean(sc))
features.append(np.var(sc))
sb=librosa.feature.spectral_bandwidth(y=y,sr=sr)
features.append(np.mean(sb))
features.append(np.var(sb))
srf=librosa.feature.spectral_rolloff(y=y,sr=sr)
features.append(np.mean(srf))
features.append(np.var(srf))
zcr=librosa.feature.zero_crossing_rate(y=y)
features.append(np.mean(zcr))
features.append(np.var(zcr))
ons=librosa.onset.onset_strength(y=y,sr=sr)
tem=librosa.feature.tempo(onset_envelope=ons,sr=sr)
features.append(np.mean(tem))
features.append(np.var(tem))
mfc=librosa.feature.mfcc(y=y,sr=sr)
mm=np.mean(mfc,axis=1)
mv=np.var(mfc,axis=1)
for i,j in zip(mm,mv):
    features.append(i)
    features.append(j)
print(features)