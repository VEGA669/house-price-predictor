import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── Load and train model ──────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('housing.csv')
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, r2, mae, X.columns.tolist()

# ── UI ────────────────────────────────────────────────────────
st.title("🏠 House Price Predictor")
st.caption("Enter house details to predict the price")

model, r2, mae, features = train_model()

# Model stats
col1, col2 = st.columns(2)
col1.metric("Model Accuracy (R² Score)", f"{r2:.2f}")
col2.metric("Avg Error", f"${mae*100000:,.0f}")

st.markdown("---")

# Input sliders
st.subheader("Enter House Details")

medinc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 5.0)
house_age = st.slider("House Age (years)", 1, 52, 20)
ave_rooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
ave_bedrms = st.slider("Average Bedrooms", 1.0, 5.0, 2.0)
population = st.slider("Population in Area", 100, 10000, 1500)
ave_occup = st.slider("Average Occupants per House", 1.0, 6.0, 3.0)
latitude = st.slider("Latitude", 32.0, 42.0, 37.0)
longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

# Predict
if st.button("🔍 Predict Price", type="primary"):
    input_data = pd.DataFrame([[medinc, house_age, ave_rooms, ave_bedrms,
                                  population, ave_occup, latitude, longitude]],
                                columns=features)
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Predicted House Price: ${prediction * 100000:,.0f}")
    
    # Feature importance
    st.markdown("---")
    st.subheader("What affects the price most?")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(importance.set_index('Feature'))