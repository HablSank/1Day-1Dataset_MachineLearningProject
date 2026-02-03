import streamlit as st
import pandas as pd
import numpy as np

st.title('🌦️ Rainfall Prediction')
st.write('Logistic Regression')

class LogisticRegressionManual:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.loss_history = []
        self.acc_history = []
        
    def _sigmoid(self, z):
        # Pengaman 1: Clip biar gak overflow (infinity)
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _log_loss(self, y_true, y_pred):
        # Pengaman 2: Epsilon biar gak log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        # Konversi ke Numpy Array biar aman
        X = np.array(X)
        y = np.array(y)
        
        # Pengaman 3 (CRITICAL): Reshape y jadi (N, 1)
        # Ini obat manjur biar akurasi gak nyangkut 50%
        y = y.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Init bobot
        self.w = np.zeros((n_features, 1))
        self.b = 0
        
        # Training Loop
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
            
            # D. Record History (Opsional, buat monitoring)
            if i % 10 == 0:
                loss = self._log_loss(y, y_pred)
                self.loss_history.append(loss)
                
                # Hitung akurasi on-the-fly
                cls_pred = (y_pred >= 0.5).astype(int)
                acc = np.mean(cls_pred == y)
                self.acc_history.append(acc)
                
                if i % 50 == 0:
                    print(f"Epoch {i}: Loss {loss:.4f} | Accuracy {acc:.4f}")

    def predict_proba(self, X):
        X = np.array(X)
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('Day-2/Logistic Regression (Rainfall)/rainfall.csv')
        df.dropna(inplace=True)
        
        df['target'] = df['weather_condition'].map({'Sunny' : 0, 'Rainy' : 1})
        df.dropna(subset=['target'], inplace=True)
        
        X = df[['rainfall', 'temperature' ,'humidity', 'wind_speed']].values
        y = df['target']

        model = LogisticRegressionManual(learning_rate=0.1, n_iters=1000)
        model.fit(X, y)
        
        return model        
    except Exception as e:
        return None
    
model = train_model()

if model is None:
    st.error('Model Tidak Ditemukan')
    st.stop()
    
st.success(('Berhasil Melatih Model (Bobot & Bias Didapatkan'))

col1, col2 = st.columns(2)
with col1:
    with st.expander("📉 Lihat Grafik Training (Loss History)"):
        st.write("Grafik ini menunjukkan bagaimana error model menurun seiring waktu training.")
        loss_data = pd.DataFrame(model.loss_history, columns=['Log Loss'])
        st.line_chart(loss_data)
with col2:
    with st.expander("📈 Lihat Grafik Accuracy (Accuracy History)"):
        st.write("Grafik ini menunjukkan bagaimana akurasi model naik seiring waktu training.")
        accuracy_data = pd.DataFrame(model.acc_history, columns=['Log Accuracy'])
        st.line_chart(accuracy_data)
    
col1, col2 = st.columns(2)
with col1:
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=5.0)
    temperature = st.number_input('Temperature (℃)', value=20.0)
    
with col2:
    humidity = st.number_input('Humidity (%)', min_value=0, value=50, max_value=100)
    wind = st.number_input('Wind Speed (Km/h)', min_value=0.0, value=5.0)
    
if st.button('Predict Weather'):
    input_data = [[rainfall, temperature, humidity, wind]]
    rain_proba = model.predict_proba(input_data)[0][0]
    st.divider()
    if rain_proba > 0.5:
        st.error('Predict: Rainy 🌧️')
        st.balloons()
    else:
        st.success('Predict: Sunny ☀️')
        
    st.metric('Rain Probability: ', f'{rain_proba*100:.2f}%')
    st.info('Logistic Regression Formula: 1 / (1 + e^-(w*x + b))')

st.divider()
st.header('Batch Prediction 📁')
uploaded_file = st.file_uploader('Upload CSV File', type=['csv'])

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    
    needed_col = ['rainfall', 'temperature' ,'humidity', 'wind_speed']
    if all(col in df_new.columns for col in needed_col):
        X_batch = df_new[needed_col].values
        probs = model.predict_proba(X_batch)
        
        df_new['Probability'] = probs
        df_new['Prediction'] = ['Rainy' if p > 0.5 else 'Sunny' for p in probs]
        
        st.write('Prediction Result:')
        st.dataframe(df_new)
        st.bar_chart(df_new['Prediction'].value_counts())
        
    else:
        st.error('Columns Not Found')
        