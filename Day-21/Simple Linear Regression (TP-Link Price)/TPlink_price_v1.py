import numpy as np
import pandas as pd
import streamlit as st

st.title('TP-Link Price Prediction')
st.write('### TWLR80N')

@st.cache_resource
def train_model():
    df = pd.read_excel('TPlink_price.xlsx')
    
    X = df[['Discrepancy']].values
    y = df['Selling Price'].values
    
    X_b = np.c_[np.ones((len(X), 1)), X]
    
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    return theta

theta = train_model()
st.success('Succes Training Model')
intercept = theta[0]
slope = theta[1]

discrepancy = st.number_input('Input Discrepancy Value', min_value=0.001, value=0.014)
if st.button('Calculate Price 😝'):
    st.balloons()
    tp_link_price = (slope * discrepancy) + intercept
    st.metric(label='TP-Link Price:', value=f'Rp. {tp_link_price:,.2f}')
    st.info(f'Formula: y = ({slope:.2f} * {discrepancy:.2f}) + {intercept:.2f}')
    
    
st.divider()
st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload file Excel", type=['xlsx'])

if uploaded_file is not None:
    df_new = pd.read_excel(uploaded_file)
    if 'Discrepancy' in df_new.columns:
        df_new['Predicted Price'] = (slope * df_new['Discrepancy']) + intercept
        st.success("Yeay!.")
        st.dataframe(df_new)
        
        st.subheader("Prediction Scatter Visualization")
        st.scatter_chart(data=df_new, x='Discrepancy', y='Predicted Price')
        
        st.subheader("Linear Visualization")
        chart_data = df_new.sort_values(by='Discrepancy')
        st.line_chart(data=chart_data, x='Discrepancy', y='Predicted Price')
    else:
        st.error("Discrepancy Column Not Found!")