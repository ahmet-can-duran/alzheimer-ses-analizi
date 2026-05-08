import streamlit as st
from scipy.io import wavfile
from python_speech_features import mfcc
import numpy as np
import joblib
@st.cache_resource
def model_yukle():
    return joblib.load("alzheimer_modeli.pkl")

model = model_yukle()
def ozellik_cikar(dosya):
    sr,y=wavfile.read(dosya)
    if len(y.shape)>1:
        y=y[:,0]
        
    mfccs_veri=mfcc(y,sr,numcep=13)
    mfcc_ortalama=np.mean(mfccs_veri,axis=0)
    return mfcc_ortalama
st.title("Ses Analizi Ile Erken Teşhis!")
st.write("Lütfen analiz etmek istediğiniz ses dosyasını (wav) uzantılı olarak ekleyiniz.")
yuklenen_dosya=st.file_uploader("Ses Verisini seç",type=["wav"])
if yuklenen_dosya is not None:
    st.audio(yuklenen_dosya, format='audio/wav')
    if st.button("Yapay zekayı çalıştır."):
        with open("temp.wav","wb") as f:
            f.write(yuklenen_dosya.getbuffer())
        mfcc_degerleri=ozellik_cikar("temp.wav")
        test_verisi=mfcc_degerleri.reshape(1,-1)
        tahmin=model.predict(test_verisi)
        st.divider()
        if tahmin[0]==0:
           st.success("Bu veriler SAĞLIKLI olduğunuzu gösteriyor..")
           st.balloons()
        elif tahmin[0]==1:
           st.error("Maalesef bu sonuçlar ALZHEİMER RISKI taşyor")