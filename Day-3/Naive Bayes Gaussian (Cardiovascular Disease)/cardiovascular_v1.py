import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title('Cardiovascular Prediction')
st.write('Naive Bayes Gaussian')

class NaiveBayesGaussian:
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Store Priors, Mean, Var
        self._priors = np.zeros(n_classes, dtype=np.float64)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            
            # Calculate Priors
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            
            # Calculate Statistics
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
        
        # Rumus Softmax untuk 2 kelas (biar jadi 0-1)
        # Probabilitas Kelas 1 (Sakit)
        sick_prob = 1 / (1 + np.exp(posteriors[0] - posteriors[1])) 
        return sick_prob

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    # Gaussian Function
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('cardiovascular.csv', delimiter=';')
        if df.shape[1] < 2:
            df = pd.read_csv('cardiovascular.csv', delimiter=',')
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        
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

model = train_model()

if model is None:
    st.error('Model Not Found')
    st.stop()

st.success('Model Loaded')


col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age (Years)', min_value=10, value=50)
    gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')
    bmi = st.number_input('BMI (Body Mass Index)', value=22.0)
    ap_hi = st.number_input('Upper Blood Pressure (Systolic)', value=120.0)
    ap_lo = st.number_input('Lower Blood Pressure (Diastolic)', value=80.0)
with col2:
    cholesterol = st.selectbox('Cholesterol', options=[1, 2, 3], format_func=lambda x: ['Normal', 'Above Normal', 'Well Above Normal'][x-1])
    gluc = st.selectbox('Glucose', options=[1, 2, 3], format_func=lambda x: ['Normal', 'Above Normal', 'Well Above Normal'][x-1])
    smoke = st.radio('Smoke?', [0, 1], format_func=lambda x: 'No' if x==0 else 'Yes', horizontal=True)
    alco = st.radio('Alcohol Intake?', [0, 1], format_func=lambda x: 'No' if x==0 else 'Yes', horizontal=True)
    active = st.radio('Physical Activity?', [0, 1], format_func=lambda x: 'No' if x==0 else 'Yes', horizontal=True)
    
    
if st.button('Predict Cardiovascular'):
    input_data = [[age, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]
    cardio_proba = model.predict_proba(input_data)[0]
    st.divider()
    

    st.subheader("Hasil Prediksi")
    if cardio_proba > 0.5:
        st.error(f"⚠️ HIGH RISK")
        st.write("It is recommended to consult a doctor.")
    else:
        st.success(f"✅ LOW RISK")
        st.write("Keep up your healthy lifestyle.")
                
    st.metric("Cardiovascular Probability: ", f"{cardio_proba*100:.1f}%")
    st.progress(cardio_proba)
        
st.divider()
st.header('Batch Prediction 📁')
uploaded_file = st.file_uploader('Upload CSV File', type=['csv'])

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    
    needed_col = ['age', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    if all(col in df_new.columns for col in needed_col):
        X_batch = df_new[needed_col].values
        probs = model.predict_proba(X_batch)
        
        df_new['Probability'] = probs
        df_new['Prediction'] = ['Positive' if p > 0.5 else 'Negative' for p in probs]
        
        st.write('Prediction Result:')
        st.dataframe(df_new)
        st.bar_chart(df_new['Prediction'].value_counts())
        
    else:
        st.error('Columns Not Found')