import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --- Custom CSS for UI ---
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Load and Train Model (Cached for Speed) ---
@st.cache_resource
def train_model():
    # Load dataset
    try:
        df = pd.read_csv('house_data.csv')
        # Ensure columns match your CSV
        X = df[['Size', 'Rooms', 'Age']]
        y = df['Price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        accuracy = r2_score(y_test, model.predict(X_test))
        return model, accuracy
    except FileNotFoundError:
        return None, None

model, accuracy = train_model()

# --- 2. Sidebar Inputs ---
st.sidebar.header("🏡 Specify House Details")
st.sidebar.write("Adjust the sliders below:")

size = st.sidebar.slider("Size (Sq. Ft.)", min_value=500, max_value=5000, value=1500, step=50)
rooms = st.sidebar.slider("Number of Rooms", min_value=1, max_value=10, value=3, step=1)
age = st.sidebar.slider("Age of House (Years)", min_value=0, max_value=100, value=10, step=1)

# --- 3. Main Page Content ---
st.title("AI Real Estate Valuator")
st.markdown("### Predict the market value of a house instantly using Machine Learning.")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Result")
    if model:
        # Make Prediction
        input_data = [[size, rooms, age]]
        prediction = model.predict(input_data)[0]
        
        # Display with big bold metric
        st.metric(label="Estimated Price", value=f"${prediction:,.2f}", delta="Predicted just now")
        
        st.info(f"Based on a house with **{size} sqft**, **{rooms} rooms**, and **{age} years** old.")
    else:
        st.error("Error: 'house_data.csv' not found. Please upload it to your GitHub repo.")

with col2:
    st.subheader("Model Insights")
    if accuracy:
        st.write(f"Model Accuracy (R²): **{accuracy:.2%}**")
        st.progress(accuracy)
        st.caption("This bar shows how well the AI understands the housing market based on your data.")
    
    st.markdown("---")
    st.write("#### How it works")
    st.write("This model uses **Multiple Linear Regression** to analyze relationships between size, age, and room count.")
