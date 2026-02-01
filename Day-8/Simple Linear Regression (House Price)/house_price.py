import numpy as np
import pandas as pd
import streamlit as st

st.title('House Price Prediction')

@st.cache_resource
def train_model():
    df = pd.read_csv('house_price.csv')
    X = df[['Square_Footage']].values
    y = df['House_Price'].values
    
    X_b = np.c_[np.ones((len(X), 1)), X]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    return theta

theta = train_model()
st.success('Load Model Success')

intercept = theta[0]
slope = theta[1]

house_sqft = st.number_input('Input House Square Footage', min_value=100, value=1500)

if st.button('Calculate Price'):
    price_predict = (slope * house_sqft) + intercept
    
    st.write(f'Estimated Price: Rp. {price_predict:,.0f}')
    st.info(f'Formula: y = ({slope:.2f} * {house_sqft:.2f}) + {intercept:.2f}')