import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

st.set_page_config(page_title="Cat vs Dog Predictor", page_icon="🐾")

st.title("🐾 Cat vs Dog Prediction (KNN Custom)")
st.write("Menggunakan algoritma KNN buatan sendiri (From Scratch).")

# --- 1. DEFINISI CLASS KNN (Sesuai kodemu + Perbaikan Bug) ---
class KNN:
    def __init__(self, k, p):
        self.k = k
        self.p = p
        
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def distance(self, x1, x2):
        if self.p == 1:
            # Manhattan Distance
            return np.sum(np.abs(x1 - x2))
        elif self.p == 2:
            # Euclidean Distance (SUDAH DIPERBAIKI KURUNGNYA)
            return np.sqrt(np.sum((x1 - x2) ** 2))
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        dists = [self.distance(x, x_train) for x_train in self.X_train]
        # Ambil indeks tetangga terdekat
        best_k = np.argsort(dists)[:self.k]
        # Ambil label dari tetangga tersebut
        label_k = [self.y_train[i] for i in best_k]
        # Cari label terbanyak (Voting)
        winner = max(set(label_k), key=label_k.count)
        return winner

# --- 2. LOAD & PREPARE DATA ---
@st.cache_resource
def get_trained_model():
    try:
        # Load Data
        df = pd.read_csv('CatsAndDogs_v2.csv')
        
        # FILTER PENTING: Hapus class 2 sesuai notebook kamu
        df = df[df['Animal'] != 2]
        
        # Pisahkan Fitur dan Label
        # Drop kolom 'Animal' untuk X
        X = df.drop(columns=['Animal']).values
        y = df['Animal'].values
        
        # Inisialisasi & Train Model
        # Kita set k=3 dan p=2 (Euclidean) sebagai default
        model = KNN(k=3, p=2)
        model.train(X, y)
        
        return model
    except Exception as e:
        return None

# Panggil fungsi (Otomatis masuk cache, jadi gak training ulang tiap detik)
knn_model = get_trained_model()

if knn_model is None:
    st.error("Gagal load data! Pastikan file 'CatsAndDogs_v2.csv' ada.")
    st.stop()

# --- 3. INPUT USER (SIDEBAR) ---
with st.sidebar:
    st.header("📏 Masukkan Ciri Hewan")
    
    # Input sesuai kolom: Height, Weight, Length, FurLength, PawSize, EarShape
    h = st.number_input("Height (cm)", 10, 100, 25)
    w = st.number_input("Weight (kg)", 1, 50, 4)
    l = st.number_input("Length (cm)", 10, 100, 30)
    
    st.markdown("---")
    fur = st.slider("Fur Length (0=Short, 2=Long)", 0, 2, 0)
    paw = st.slider("Paw Size (0=Small, 2=Big)", 0, 2, 0)
    ear = st.slider("Ear Shape (0=Pointy, 2=Floppy)", 0, 2, 0)
    
    predict_btn = st.button("Tebak Hewan 🔮")

# --- 4. PREDIKSI ---
if predict_btn:
    # Gabungkan input jadi array numpy 1 baris
    data_baru = np.array([[h, w, l, fur, paw, ear]])
    
    # Lakukan prediksi
    hasil = knn_model.predict(data_baru)
    label_hasil = hasil[0] # Ambil elemen pertama karena outputnya list
    
    # Mapping Label (Sesuai notebook: 0=Cats, 1=Dogs)
    nama_hewan = "🐱 KUCING (Cat)" if label_hasil == 0 else "🐶 ANJING (Dog)"
    
    st.subheader("Hasil Prediksi:")
    if label_hasil == 0:
        st.success(f"Ini sepertinya: **{nama_hewan}**")
        st.balloons()
    else:
        st.info(f"Ini sepertinya: **{nama_hewan}**")