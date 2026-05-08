import streamlit as st
from scipy.io import wavfile
from scipy.signal import resample
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
        
    if sr != 44100:
        y = resample(y, int(len(y) * 44100 / sr))
        sr = 44100
        
    if y.dtype != np.int16:
        if np.max(np.abs(y)) <= 1.5: 
            y = np.int16(y * 32767)
        else:
            y = np.int16(y)
            
    mfccs_veri = mfcc(y, sr, numcep=13)
    mfcc_ortalama = np.mean(mfccs_veri, axis=0)
    return mfcc_ortalama

def analiz_yap(ses_verisi):
    with open("temp.wav", "wb") as f:
        f.write(ses_verisi.getbuffer())
        
    st.write("Yapay zekamın analiz ettiği ses:")
    st.audio("temp.wav", format="audio/wav")
        
    mfcc_degerleri = ozellik_cikar("temp.wav")
    test_verisi = mfcc_degerleri.reshape(1, -1)
    tahmin = model.predict(test_verisi)
    
    st.divider()
    if tahmin[0] == 0:
        st.success("Analizimi tamamladım. Girdiğiniz ses verisi tamamen sağlıklı bir profile uyuyor.")
    elif tahmin[0] == 1:
        st.error("Analizimi tamamladım. Modelim bu ses profilinde maalesef Alzheimer risk faktörleri tespit etti.")

st.title("TÜBİTAK 2209-A: Ses Analizi İle Erken Teşhis")
st.write("Geliştirdiğim yapay zeka modelini test etmek için aşağıdaki analiz yöntemlerinden birini seçebilirsiniz.")

sekme1, sekme2 = st.tabs(["Dosya Yükle", "Mikrofonla Kaydet"])

with sekme1:
    yuklenen_dosya = st.file_uploader("Test edilecek .wav dosyasını yükleyebilirsiniz", type=["wav"])
    if yuklenen_dosya is not None:
        st.audio(yuklenen_dosya, format='audio/wav')
        if st.button("Sisteme Yükle ve Analizimi Başlat"):
            analiz_yap(yuklenen_dosya)

with sekme2:
    st.info("Sistemimin doğru bir analiz yapabilmesi için mikrofona basıp birkaç saniye konuşmanız yeterlidir.")
    kaydedilen_ses = st.audio_input("Ses kaydı almak için tıklayın")
    
    if kaydedilen_ses is not None:
        if st.button("Kaydedilen Sesi Analiz Et"):
            analiz_yap(kaydedilen_ses)
