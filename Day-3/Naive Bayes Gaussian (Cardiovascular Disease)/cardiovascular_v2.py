import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. PAGE CONFIG (Wajib paling atas)
st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENGINE (LOGIC) TETAP SAMA ---
class NaiveBayesGaussian:
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + 1e-9

    def predict_proba(self, X):
        X = np.array(X)
        return np.array([self._calculate_proba(x) for x in X])

    def _calculate_proba(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            pdf = self._pdf(idx, x)
            likelihood = np.sum(np.log(pdf + 1e-9))
            posteriors.append(prior + likelihood)
        sick_prob = 1 / (1 + np.exp(posteriors[0] - posteriors[1])) 
        return sick_prob

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

@st.cache_resource
def train_model():
    try:
        # Coba baca dengan delimiter berbeda buat jaga-jaga
        df = pd.read_csv('cardiovascular.csv', delimiter=';')
        if df.shape[1] < 2:
            df = pd.read_csv('cardiovascular.csv', delimiter=',')
            
        df.drop_duplicates(inplace=True)
        # Preprocessing sesuai logic kamu
        df['age'] = (df['age'] / 365.25).astype(int)
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
        df = df[(df['ap_hi'] < 250) & (df['ap_hi'] > 60)]
        df = df[(df['ap_lo'] < 150) & (df['ap_lo'] > 40)]
        df = df[df['ap_hi'] >= df['ap_lo']]
        
        X = df[['age', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']].values
        y = df['cardio']
        
        model = NaiveBayesGaussian()
        model.fit(X, y)
        return model
    except Exception as e:
        return None

# --- UI DIMULAI DI SINI ---

# Sidebar Title
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=80)
st.sidebar.title("Parameter Pasien")
st.sidebar.write("Isi data pasien di bawah ini:")

# Load Model dengan Status Spinner (Lebih elegan)
with st.spinner('Memuat Model AI...'):
    model = train_model()

if model is None:
    st.error("❌ Model gagal dimuat. Cek file 'cardiovascular.csv'")
    st.stop()

# --- INPUT SECTION (DI SIDEBAR & DIKELOMPOKKAN) ---
# Menggunakan Expander biar rapi
with st.sidebar:
    with st.expander("👤 Data Diri", expanded=True):
        age = st.number_input('Umur (Tahun)', 10, 100, 50, help="Usia pasien dalam tahun")
        gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Wanita' if x==0 else 'Pria')
        bmi = st.slider('BMI', 10.0, 50.0, 22.0, help="Body Mass Index")

    with st.expander("🏥 Tanda Vital", expanded=True):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            ap_hi = st.number_input('Systolic', 60, 240, 120, help="Tekanan Atas")
        with col_s2:
            ap_lo = st.number_input('Diastolic', 40, 180, 80, help="Tekanan Bawah")
        
        cholesterol = st.select_slider('Kolesterol', options=[1, 2, 3], format_func=lambda x: ['Normal', 'Agak Tinggi', 'Sangat Tinggi'][x-1])
        gluc = st.select_slider('Glukosa (Gula)', options=[1, 2, 3], format_func=lambda x: ['Normal', 'Agak Tinggi', 'Sangat Tinggi'][x-1])

    with st.expander("🍺 Gaya Hidup"):
        smoke = st.checkbox("Merokok?", value=False)
        alco = st.checkbox("Konsumsi Alkohol?", value=False)
        active = st.checkbox("Rajin Olahraga?", value=True)

    # Convert boolean ke int (0/1) untuk model
    smoke = 1 if smoke else 0
    alco = 1 if alco else 0
    active = 1 if active else 0
    
    predict_btn = st.button("🔍 Analisa Risiko", type="primary", use_container_width=True)

# --- MAIN CONTENT AREA ---
st.title("🫀 CardioGuard AI")
st.markdown("Sistem deteksi dini risiko penyakit kardiovaskular menggunakan **Gaussian Naive Bayes**.")

# Gunakan TABS untuk memisahkan mode
tab_single, tab_batch = st.tabs(["📊 Analisa Personal", "📂 Analisa Massal (Upload)"])

# === TAB 1: SINGLE PREDICTION ===
with tab_single:
    if predict_btn:
        # Prediksi
        input_data = [[age, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]
        prob = model.predict_proba(input_data)[0]
        
        # --- TAMPILAN HASIL (CARD STYLE) ---
        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            # Container dengan border biar kayak kartu
            with st.container(border=True):
                st.write("### Tingkat Risiko")
                
                # Logic Warna
                if prob > 0.7:
                    status_color = "red"
                    status_text = "SANGAT TINGGI"
                    icon = "🚨"
                elif prob > 0.5:
                    status_color = "orange"
                    status_text = "TINGGI"
                    icon = "⚠️"
                else:
                    status_color = "green"
                    status_text = "RENDAH"
                    icon = "✅"
                
                st.markdown(f"<h1 style='text-align: center; color: {status_color};'>{status_text}</h1>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-size: 50px;'>{icon}</div>", unsafe_allow_html=True)
                st.metric("Probabilitas", f"{prob*100:.1f}%")
                
        with col_res2:
            st.subheader("🔍 Visualisasi Profil")
            
            # Helper Plot (Updated Range)
            def plot_gaussian_mini(feature_idx, feature_name, user_val):
                mean0, var0 = model._mean[0][feature_idx], model._var[0][feature_idx]
                mean1, var1 = model._mean[1][feature_idx], model._var[1][feature_idx]
                std0, std1 = np.sqrt(var0), np.sqrt(var1)
                
                start = min(mean0-3*std0, mean1-3*std1, user_val-10)
                end = max(mean0+3*std0, mean1+3*std1, user_val+10)
                x = np.linspace(start, end, 200)
                y0 = (1/np.sqrt(2*np.pi*var0)) * np.exp(-((x-mean0)**2)/(2*var0))
                y1 = (1/np.sqrt(2*np.pi*var1)) * np.exp(-((x-mean1)**2)/(2*var1))
                
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.fill_between(x, y0, color='green', alpha=0.3, label='Sehat')
                ax.fill_between(x, y1, color='red', alpha=0.3, label='Berisiko')
                ax.axvline(user_val, color='black', linewidth=2, linestyle='--')
                ax.text(user_val, max(y0.max(), y1.max())*0.9, "ANDA", ha='center', fontweight='bold')
                ax.set_title(feature_name, fontsize=10)
                ax.axis('off')
                st.pyplot(fig)
            
            # Tampilkan 2 grafik dalam expander biar rapi
            with st.expander("Lihat Detail Grafik Distribusi", expanded=True):
                plot_gaussian_mini(3, "Tekanan Darah (Systolic)", ap_hi)
                plot_gaussian_mini(2, "Body Mass Index (BMI)", bmi)
    else:
        st.info("👈 Silakan isi data di Sidebar sebelah kiri dan klik tombol 'Analisa Risiko'.")

# === TAB 2: BATCH PREDICTION ===
with tab_batch:
    st.write("Upload file CSV untuk memproses banyak data sekaligus.")
    
    # Tombol download template (Biar user gak bingung formatnya)
    sample_data = pd.DataFrame({
        'age': [45, 60], 'gender': [1, 0], 'bmi': [22.5, 30.1], 
        'ap_hi': [120, 150], 'ap_lo': [80, 95], 'cholesterol': [1, 3],
        'gluc': [1, 1], 'smoke': [0, 1], 'alco': [0, 0], 'active': [1, 0]
    })
    csv_sample = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Template CSV", csv_sample, "template_cardio.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        cols = ['age', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
        if all(c in df_new.columns for c in cols):
            X_batch = df_new[cols].values
            probs = model.predict_proba(X_batch)
            
            df_new['Risk Probability'] = probs
            df_new['Risk Status'] = ['High' if p > 0.5 else 'Low' for p in probs]
            
            st.dataframe(df_new.style.background_gradient(subset=['Risk Probability'], cmap='Reds'))
            
            # Download Result
            res_csv = df_new.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Simpan Hasil Analisa", res_csv, "hasil_analisa.csv", "text/csv", type='primary')
        else:
            st.error("Kolom CSV tidak sesuai template!")