import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
import os
from datetime import datetime, timedelta
import uuid
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Tomato Doctor & Market",
    page_icon="🌱",
    layout="centered"
)

# =============================================
# 1. Disease Detection Components (Original)
# =============================================
HISTORY_FILE = "farmer_history.json"
MAX_HISTORY_ENTRIES = 100

@st.cache_resource
def load_models():
    leaf_classifier = tf.keras.models.load_model('leaf_classifier_mobilenetv2.h5')
    model_path = 'Tomato_doctor_mblnetv2.h5'
    for path in [model_path] + ['./models/Tomato_doctor_mblnetv2.h5', './plant-doctor/Tomato_doctor_mblnetv2.h5']:
        if os.path.exists(path):
            disease_model = tf.keras.models.load_model(path)
            return leaf_classifier, disease_model
    raise FileNotFoundError("Model file not found")

@st.cache_data
def load_knowledge():
    with open('final_crop_disease_knowledge_base.json') as f:
        return json.load(f)['diseases']

def process_disease_detection(uploaded_file, device_id):
    try:
        leaf_classifier, disease_model = load_models()
        knowledge = load_knowledge()
        
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Leaf classification
        leaf_pred = leaf_classifier.predict(img_array, verbose=0)
        leaf_class = ['non_leaf', 'other_leaf', 'tomato_leaf'][np.argmax(leaf_pred)]
        
        if leaf_class != 'tomato_leaf':
            st.error("Please upload a clear tomato leaf image")
            return None
        
        # Disease detection
        output = disease_model.predict(img_array, verbose=0)[0]
        predicted_class = max(load_class_indices().items(), key=lambda x: output[int(x[0])])[1]
        confidence = float(output[np.argmax(output)])
        
        return format_results(predicted_class, knowledge[predicted_class], confidence)
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return None

# =============================================
# 2. Price Forecast Demo (New)
# =============================================
def generate_forecast_data():
    dates = [datetime.today() + timedelta(days=i) for i in range(30)]
    base_price = 36.0
    trends = np.sin(np.linspace(0, 3*np.pi, 30)) * 2 + np.random.normal(0, 0.3, 30)
    prices = (base_price + trends).round(2)
    
    return pd.DataFrame({
        "Date": dates,
        "Price (SZL/kg)": prices,
        "Lower Bound": (prices - 0.8).round(2),
        "Upper Bound": (prices + 0.8).round(2)
    })

def show_forecast_demo():
    st.header("🍅 Tomato Price Forecast")
    st.info("This demo shows simulated market trends - real data would come from your LSTM model")
    
    forecast_data = generate_forecast_data()
    
    # Visualization
    st.line_chart(
        forecast_data.set_index("Date"),
        y=["Price (SZL/kg)", "Lower Bound", "Upper Bound"],
        color=["#FF0000", "#888888", "#888888"]
    )
    
    # Data table
    with st.expander("View Forecast Data"):
        st.dataframe(
            forecast_data,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Price (SZL/kg)": st.column_config.NumberColumn("Price", format="%.2f")
            },
            hide_index=True
        )
    
    # Key metrics
    current_price = forecast_data["Price (SZL/kg)"].iloc[0]
    min_price = forecast_data["Price (SZL/kg)"].min()
    st.metric("Current Market Price", f"{current_price:.2f} SZL/kg", 
             f"{(current_price - min_price):.2f} above season low")

# =============================================
# 3. Integrated App Navigation
# =============================================
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode",
        ("Disease Diagnosis", "Market Forecast"),
        index=0
    )
    
    if app_mode == "Disease Diagnosis":
        st.title("🍅🌿 Tomato Disease Diagnosis")
        st.markdown("Upload a clear photo of a tomato leaf for analysis")
        
        uploaded_file = st.file_uploader(
            "Choose leaf image...", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            device_id = str(uuid.uuid4())  # Simplified for demo
            result = process_disease_detection(uploaded_file, device_id)
            if result:
                st.image(Image.open(uploaded_file), width=300)
                if result['status'] == "healthy":
                    st.success("✅ Healthy Leaf")
                else:
                    st.warning(f"⚠️ {result['disease_name']}")
                
                with st.expander("Treatment Recommendations"):
                    st.write(result.get('treatments', {}))
    
    else:  # Market Forecast
        show_forecast_demo()

if __name__ == "__main__":
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    main()
