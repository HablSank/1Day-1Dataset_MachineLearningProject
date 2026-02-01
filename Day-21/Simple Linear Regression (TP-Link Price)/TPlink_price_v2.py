
import numpy as np
import pandas as pd
import streamlit as st

# Setup page biar lebih lebar (Widescreen mode)
st.set_page_config(page_title="TP-Link Predictor", layout="wide")

st.title('📶 TP-Link Price Prediction')
st.markdown("Dashboard prediksi harga **TWLR80N** berbasis Machine Learning.")

# --- MODEL TRAINING ---
@st.cache_resource
def train_model():
    try:
        df = pd.read_excel('TPlink_price.xlsx')
        X = df[['Discrepancy']].values
        y = df['Selling Price'].values
        X_b = np.c_[np.ones((len(X), 1)), X]
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return theta
    except:
        return None

theta = train_model()

# Jika model gagal load
if theta is None:
    st.error("File dataset tidak ditemukan!")
    st.stop()

intercept = theta[0]
slope = theta[1]

# --- FITUR 1: SIDEBAR (Pindah Input ke Samping) ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    st.write("Atur parameter di sini:")
    
    # Input ada di dalam sidebar
    discrepancy = st.number_input('Input Discrepancy', min_value=0.001, value=0.014, step=0.001)
    
    predict_btn = st.button('Hitung Harga 🚀')
    
    st.markdown("---")
    st.caption(f"Model Info:\nIntercept: {intercept:.2f}\nSlope: {slope:.2f}")

# --- LOGIKA PREDIKSI SINGLE ---
if predict_btn:
    tp_link_price = (slope * discrepancy) + intercept
    
    # --- FITUR 2: COLUMNS (Tampilan Berjajar) ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Input Discrepancy")
        st.write(f"**{discrepancy:.3f}**")
        
    with col2:
        st.success("Estimasi Harga (Predicted)")
        # Metric bikin angka jadi besar dan bold
        st.metric(label="Selling Price", value=f"Rp {tp_link_price:,.2f}")
    
    st.balloons()

# --- LOGIKA BATCH PREDICTION ---
st.divider()
st.header("📂 Batch Prediction (Banyak Data)")

# Gunakan Expander biar penjelasan tidak memenuhi layar
with st.expander("ℹ️ Klik untuk baca panduan upload"):
    st.write("Upload file Excel (.xlsx) yang memiliki kolom bernama **'Discrepancy'**.")

uploaded_file = st.file_uploader("Upload file Excel di sini", type=['xlsx'])

if uploaded_file is not None:
    df_new = pd.read_excel(uploaded_file)
    
    if 'Discrepancy' in df_new.columns:
        # Hitung Prediksi
        df_new['Predicted Price'] = (slope * df_new['Discrepancy']) + intercept
        
        # Tampilkan Dataframe
        st.write("### Hasil Prediksi:")
        st.dataframe(df_new, use_container_width=True) # Full width
        
        # --- FITUR 3: DOWNLOAD BUTTON ---
        # Convert dataframe ke CSV string
        csv_data = df_new.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download Hasil sebagai CSV",
            data=csv_data,
            file_name="hasil_prediksi_tplink.csv",
            mime="text/csv",
        )
        
        # Visualisasi (Tetap ada)
        st.divider()
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Scatter Plot")
            st.scatter_chart(data=df_new, x='Discrepancy', y='Predicted Price')
            
        with col_chart2:
            st.subheader("Trend Line")
            chart_data = df_new.sort_values(by='Discrepancy')
            st.line_chart(data=chart_data, x='Discrepancy', y='Predicted Price')

    else:
        st.error("Kolom 'Discrepancy' tidak ditemukan di file Excel!")