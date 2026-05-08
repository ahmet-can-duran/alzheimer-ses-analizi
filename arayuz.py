import streamlit as st
from scipy.io import wavfile
from scipy.signal import resample
from python_speech_features import mfcc
import numpy as np
import joblib
import noisereduce as nr

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
        
    y_temiz = nr.reduce_noise(y=y, sr=sr)
        
    mfccs_veri = mfcc(y_temiz, sr, numcep=13)
    mfcc_ortalama = np.mean(mfccs_veri, axis=0)
    return mfcc_ortalama

def analiz_yap(ses_verisi):
    with open("temp.wav", "wb") as f:
        f.write(ses_verisi.getbuffer())
        
    st.write("Yapay Zekanın Analiz Ettiği Ses:")
    st.audio("temp.wav", format="audio/wav")
        
    mfcc_degerleri = ozellik_cikar("temp.wav")
    test_verisi = mfcc_degerleri.reshape(1, -1)
    tahmin = model.predict(test_verisi)
    
    st.divider()
    if tahmin[0] == 0:
        st.success("Analizimi tamamladım. Girdiğiniz ses verisi tamamen sağlıklı bir profile uyuyor.")
        st.balloons()
    elif tahmin[0] == 1:
        st.error("Analizimi tamamladım. Modelim bu ses profilinde maalesef Alzheimer risk faktörleri tespit etti.")

st.title("TÜBİTAK 2209-A: Ses Analizi İle Erken Teşhis")
st.write("Geliştirdiğim yapay zeka modelini test etmek için aşağıdaki analiz yöntemlerinden birini seçebilirsiniz.")

sekme1, sekme2 = st.tabs(["Dosya Yükle", "Mikrofonla Kaydet"])

with sekme1:
    yuklenen_dosya = st.file_uploader("Test edilecek .wav dosyasını yükleyebilirsiniz", type=["wav"])
    if yuklenen_dosya is not None:
        st.audio(yuklenen_dosya, format='audio/wav')
        if st.button("Sisteme Yükle ve Analizi Başlat"):
            analiz_yap(yuklenen_dosya)

with sekme2:
    st.info("Sistemimin doğru bir analiz yapabilmesi için mikrofona basıp birkaç saniye konuşmanız yeterlidir.")
    kaydedilen_ses = st.audio_input("Ses kaydı almak için tıklayın")
    
    if kaydedilen_ses is not None:
        if st.button("Kaydedilen Sesi Analiz Et"):
            analiz_yap(kaydedilen_ses)
