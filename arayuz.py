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
    sr, y = wavfile.read(dosya)
    if len(y.shape) > 1:
        y = y[:, 0]
        
    mfccs_veri = mfcc(y, sr, numcep=13)
    mfcc_ortalama = np.mean(mfccs_veri, axis=0)
    return mfcc_ortalama

# Her iki sekmede de aynı işlemi yapacağımız için bunu bir fonksiyona bağladık
def analiz_yap(ses_verisi):
    with open("temp.wav", "wb") as f:
        f.write(ses_verisi.getbuffer())
        
    mfcc_degerleri = ozellik_cikar("temp.wav")
    test_verisi = mfcc_degerleri.reshape(1, -1)
    tahmin = model.predict(test_verisi)
    
    st.divider()
    if tahmin[0] == 0:
        st.success(" ANALİZ SONUCU: Bu veriler SAĞLIKLI olduğunuzu gösteriyor..")
        st.balloons()
    elif tahmin[0] == 1:
        st.error(" ANALİZ SONUCU: Maalesef bu sonuçlar ALZHEİMER RİSKİ taşıyor")

st.title(" Ses Analizi İle Erken Teşhis!")
st.write("Lütfen analiz yöntemini seçin:")


sekme1, sekme2 = st.tabs([" Dosya Yükle", " Mikrofonla Kaydet"])


with sekme1:
    yuklenen_dosya = st.file_uploader("Ses Verisini seç (wav)", type=["wav"])
    if yuklenen_dosya is not None:
        st.audio(yuklenen_dosya, format='audio/wav')
        if st.button("Yüklenen Sesi Analiz Et"):
            analiz_yap(yuklenen_dosya)


with sekme2:
    st.info(" Doğru sonuç için lütfen mikrofon simgesine basıp  ses çıkarın veya normal bir tonda konuşun.")
  
    kaydedilen_ses = st.audio_input("Sesinizi kaydetmek için tıklayın")
    
    if kaydedilen_ses is not None:
        if st.button("Kaydedilen Sesi Analiz Et"):
            analiz_yap(kaydedilen_ses)
            st.ballons
