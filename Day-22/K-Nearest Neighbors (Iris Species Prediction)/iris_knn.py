import streamlit as st
import joblib
import numpy as np
import pandas as pd

class KNN:
    def __init__(self, k, p, weights):
        self.P = p
        self.K = k
        self.Weights = weights
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def distance(self, x1, x2):
        diff = np.abs(x1 - x2)
        axis_val = 1 if x1.ndim > 1 else 0
        if self.P == 1: return np.sum(diff, axis=axis_val)
        elif self.P == 2: return np.sqrt(np.sum(diff ** 2, axis=axis_val))
        else: return np.max(diff, axis=axis_val)

    def _predict(self, x):
        dists = self.distance(self.X_train, x)
        best_k = np.argsort(dists)[:self.K]
        label_k = self.y_train[best_k]
        
        if self.Weights == 'uniform':
            values, counts = np.unique(label_k, return_counts=True)
            return values[np.argmax(counts)]
        elif self.Weights == 'distance':
            weights = 1 / (dists[best_k] + 1e-6)
            vote_scores = {}
            for lbl, w in zip(label_k, weights):
                vote_scores[lbl] = vote_scores.get(lbl, 0) + w
            return max(vote_scores, key=vote_scores.get)

# --- B. Load Data Training (Kita cache biar ngebut) ---
@st.cache_resource
def load_saved_model():
    # Load file joblib
    data = joblib.load('model_iris_knn.joblib')
    return data['model_knn'], data['mean'], data['std']

# Panggil fungsinya
try:
    model, mean_val, std_val = load_saved_model()
    st.success("Model berhasil dimuat dari file Joblib! 🚀")
except:
    st.error("File model belum ada. Jalankan Notebook dulu buat bikin file .joblib")

# ==========================================
# 2. USER INTERFACE (Bagian "Wajah"-nya)
# ==========================================

st.set_page_config(page_title="Iris Predictor", page_icon="🌸")

# Header
st.title("🌸 Iris Species Prediction")
st.markdown("""
Aplikasi ini memprediksi spesies bunga Iris menggunakan algoritma **KNN**.
Silakan masukkan parameter ukuran bunga di bawah ini.
""")
st.divider() # Garis pemisah horizontal

# --- 4. FORM INPUT (LAYOUTING) ---
if model is not None:
    # Membagi layar jadi 2 kolom
    col1, col2 = st.columns(2)

    # Isi Kolom Kiri
    with col1:
        st.subheader("Ukuran Sepal")
        # st.number_input(Label, Min, Max, Default Value)
        sepallen = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
        sepalwid = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
        sepalare = st.number_input("Sepal Area (cm²)", 0.0, 50.0, 17.8)

    # Isi Kolom Kanan
    with col2:
        st.subheader("Ukuran Petal")
        petallen = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
        petalwid = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)
        petalare = st.number_input("Petal Area (cm²)", 0.0, 50.0, 0.28)

    st.divider()

    # --- 5. LOGIKA PREDIKSI ---
    # Tombol ditekan -> Proses dijalankan
    if st.button("🔍 Prediksi Spesies", type="primary", use_container_width=True):
        
        # A. Progress bar (biar terlihat canggih/loading)
        with st.spinner('Sedang memproses...'):
            
            # B. Susun Data Input
            raw_input = np.array([sepallen, sepalwid, petallen, petalwid, sepalare, petalare])
            
            # C. Scaling (StandardScaler formula)
            # Rumus: z = (x - u) / s
            mean_val = np.array(mean_val)
            std_val = np.array(std_val)
            input_scaled = (raw_input - mean_val) / std_val
            
            # D. Prediksi
            # Input harus 2D array untuk sklearn, makanya pakai [input_scaled]
            result_idx = model.predict([input_scaled])[0]
            
            # E. Mapping Hasil
            species_names = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
            final_name = species_names.get(result_idx, "Unknown")
            
            # F. Tampilkan Output Berwarna
            st.markdown("### Hasil Analisa:")
            
            if result_idx == 0:
                st.success(f"Spesies terdeteksi: **{final_name}**")
                st.balloons()
            elif result_idx == 1:
                st.warning(f"Spesies terdeteksi: **{final_name}**")
                st.balloons()
            else:
                st.info(f"Spesies terdeteksi: **{final_name}**")
                st.balloons()

else:
    st.error("Model gagal dimuat. Cek file csv atau pickle kamu.")

# --- 6. FOOTER ---

st.divider()
st.markdown("<p style='text-align: center;'>© Pemuda Sintaks | 2026</p>", unsafe_allow_html=True)