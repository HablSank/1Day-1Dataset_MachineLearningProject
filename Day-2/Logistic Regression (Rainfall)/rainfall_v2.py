import streamlit as st
import numpy as np
import pandas as pd

# 1. SETUP HALAMAN
st.set_page_config(page_title="Rainfall Prediction (LogReg)", layout="wide")

st.title("🌦️ Prediksi Hujan (Logistic Regression)")
st.markdown("""
Aplikasi ini menggunakan algoritma **Logistic Regression (Manual Implementation)** untuk memprediksi cuaca berdasarkan data curah hujan, suhu, kelembaban, dan angin.
""")

# --- 2. CLASS MODEL (Sesuai Kode Kamu) ---
class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.loss_history = []
        self.acc_history = []
        
    def _sigmoid(self, z):
        # Pengaman 1: Clip biar gak overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _log_loss(self, y_true, y_pred):
        # Pengaman 2: Epsilon biar gak log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # Pengaman 3: Reshape y jadi (N, 1)
        y = y.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Init bobot
        self.w = np.zeros((n_features, 1))
        self.b = 0
        
        for i in range(self.n_iters):
            # A. Forward
            z = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(z)
            
            # B. Hitung Gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # C. Update Parameter
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # D. Record History
            if i % 10 == 0:
                loss = self._log_loss(y, y_pred)
                self.loss_history.append(loss)
                
                cls_pred = (y_pred >= 0.5).astype(int)
                acc = np.mean(cls_pred == y)
                self.acc_history.append(acc)

    def predict_proba(self, X):
        X = np.array(X)
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# --- 3. TRAINING PROCESS (Cached) ---
@st.cache_resource
def train_model():
    try:
        # Load Data
        df = pd.read_csv('rainfall.csv')
        
        # Preprocessing (Sesuai Notebook)
        df.dropna(inplace=True)
        
        # Mapping: Sunny=0, Rainy=1
        weather_map = {"Sunny": 0, "Rainy": 1}
        df['target'] = df['weather_condition'].map(weather_map)
        
        # Hapus baris yang targetnya NaN (kalau ada label aneh)
        df.dropna(subset=['target'], inplace=True)
        
        # Definisikan X dan y
        feature_cols = ['rainfall', 'temperature', 'humidity', 'wind_speed']
        X = df[feature_cols].values
        y = df['target'].values
        
        # Inisialisasi Model Manual
        # Kita pakai iterasi agak banyak biar konvergen
        model = LogisticRegressionManual(learning_rate=0.01, n_iters=2000)
        model.fit(X, y)
        
        return model, df # Kembalikan df juga buat preview
    except Exception as e:
        return None, None

# Panggil fungsi training
model, df_clean = train_model()

if model is None:
    st.error("Gagal training! Pastikan file 'rainfall.csv' ada di folder yang sama.")
    st.stop()

# Tampilkan Status Training
st.sidebar.success("✅ Model Trained Successfully!")
st.sidebar.metric("Final Accuracy (Last Epoch)", f"{model.acc_history[-1]*100:.2f}%")

# --- 4. UI INPUT USER ---
with st.sidebar:
    st.header("🎛️ Masukkan Data Cuaca")
    
    # Inputan sesuai fitur X
    input_rain = st.number_input("Rainfall (mm)", min_value=0.0, value=10.5)
    input_temp = st.number_input("Temperature (°C)", min_value=-10.0, value=25.0)
    input_hum  = st.slider("Humidity (%)", 0, 100, 75)
    input_wind = st.number_input("Wind Speed (km/h)", min_value=0.0, value=5.5)
    
    btn_predict = st.button("Prediksi Cuaca ☀️/🌧️")

# --- 5. LOGIKA PREDIKSI SINGLE ---
if btn_predict:
    # Gabungkan input jadi array (1 baris, 4 kolom)
    X_new = [[input_rain, input_temp, input_hum, input_wind]]
    
    # Prediksi Probabilitas
    prob = model.predict_proba(X_new)[0][0] # Ambil nilai float-nya
    
    # Prediksi Kelas (0 atau 1)
    # Threshold 0.5 (Default)
    pred_class = 1 if prob >= 0.5 else 0
    
    # Tampilkan Hasil Utama
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if pred_class == 1:
            st.image("https://cdn-icons-png.flaticon.com/512/116/116251.png", width=150)
            status = "RAINY (Hujan)"
            color = "inverse" # Merah/Gelap
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=150)
            status = "SUNNY (Cerah)"
            color = "normal" # Hijau/Terang
            
    with col2:
        st.subheader(f"Hasil: {status}")
        st.write(f"Confidence Level (Probabilitas Hujan): **{prob*100:.2f}%**")
        
        # Progress bar probabilitas
        st.progress(float(prob), text="Peluang Hujan")
        
        if pred_class == 1:
            st.warning("Sedia payung sebelum hujan! ☂️")
        else:
            st.success("Cuaca cerah, waktunya jalan-jalan! 😎")

# --- 6. VISUALISASI LOSS (Bonus Pembelajaran) ---
# Biar kelihatan proses 'belajar' modelnya
with st.expander("📈 Lihat Grafik Training (Loss History)"):
    st.write("Grafik ini menunjukkan bagaimana error model menurun seiring waktu training.")
    loss_data = pd.DataFrame(model.loss_history, columns=['Log Loss'])
    st.line_chart(loss_data)

# --- 7. BATCH PREDICTION (Upload File) ---
st.divider()
st.header("📂 Batch Prediction")
st.write("Upload file CSV baru yang punya kolom: `rainfall`, `temperature`, `humidity`, `wind_speed`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_batch = pd.read_csv(uploaded_file)
    
    # Pastikan kolom yang dibutuhkan ada
    required_cols = ['rainfall', 'temperature', 'humidity', 'wind_speed']
    if all(col in df_batch.columns for col in required_cols):
        
        # Lakukan Prediksi
        X_batch = df_batch[required_cols].values
        probs = model.predict_proba(X_batch) # Hasilnya array (N, 1)
        
        # Masukkan hasil ke dataframe
        df_batch['Probability (Rainy)'] = probs
        df_batch['Prediction'] = ["Rainy" if p >= 0.5 else "Sunny" for p in probs]
        
        st.write("### Hasil Prediksi Batch:")
        st.dataframe(df_batch)
        
        # Visualisasi Simpel
        st.bar_chart(df_batch['Prediction'].value_counts())
        
    else:
        st.error(f"Kolom tidak lengkap! Wajib ada: {required_cols}")