# app.py
# House Price Prediction Web App using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ------------------------------
# CUSTOM CSS
# ------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 50px;
    color: #1f4e79;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 20px;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

.prediction-box {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 30px;
    color: #155724;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# HEADER
# ------------------------------
st.markdown('<p class="title">🏠 House Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict house prices using Machine Learning</p>', unsafe_allow_html=True)

# ------------------------------
# SAMPLE DATASET
# ------------------------------
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2500],
    'Bedrooms': [2, 2, 3, 3, 4, 4],
    'Bathrooms': [1, 2, 2, 3, 3, 4],
    'Age': [10, 8, 5, 4, 3, 1],
    'Price': [3000000, 4000000, 5500000, 6500000, 7500000, 9500000]
}

df = pd.DataFrame(data)

# ------------------------------
# MODEL TRAINING
# ------------------------------
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

model = RandomForestRegressor()
model.fit(X, y)

# ------------------------------
# LAYOUT
# ------------------------------
col1, col2 = st.columns([1, 1])

# ------------------------------
# INPUT SECTION
# ------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Enter House Details")

    area = st.slider("Area (sq ft)", 500, 5000, 1500)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
    age = st.slider("House Age", 0, 30, 5)

    predict_button = st.button("Predict Price")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# PREDICTION SECTION
# ------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Prediction Result")

    if predict_button:

        input_data = np.array([[area, bedrooms, bathrooms, age]])

        prediction = model.predict(input_data)[0]

        st.markdown(
            f'''
            <div class="prediction-box">
                Estimated Price <br>
                ₹ {prediction:,.0f}
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.balloons()

    else:
        st.info("Enter house details and click Predict Price")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# DATA PREVIEW
# ------------------------------
st.markdown("## 📊 Sample Training Data")

st.dataframe(df, use_container_width=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("""
<hr>
<center>
</center>
""", unsafe_allow_html=True)
