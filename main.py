from ctypes.util import test
import pandas as pd 
from input_audio_extractor import audiofeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
data=pd.read_csv(r"C:\Users\user\Downloads\music genre classifier\Data\features_30_sec.csv")
model=RandomForestClassifier(n_estimators=300,random_state=42)
data=data.drop(columns=["length","filename","harmony_mean","harmony_var","perceptr_mean","perceptr_var"])
y=data["label"]
X=data.drop(columns=["label"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=42)
model.fit(X,y)
inp=r"C:\Users\user\Downloads\the_mountain-rock-rock-music-576586.mp3"
input_features=audiofeatures(inp)
input_feat=pd.DataFrame([input_features],columns=X.columns)
y_pred=model.predict(input_feat)
print(y_pred)