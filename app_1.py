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
    page_title="Tomato Doctor Pro",
    page_icon="🌱",
    layout="centered"
)

# =============================================
# 1. Original Disease Detection Components
# =============================================
HISTORY_FILE = "farmer_history.json"
MAX_HISTORY_ENTRIES = 100

@st.cache_resource
def load_models():
    # Load leaf classifier model
    leaf_classifier = tf.keras.models.load_model('leaf_classifier_mobilenetv2.h5')
    
    # Load disease detection model
    model_path = 'Tomato_doctor_mblnetv2.h5'
    fallback_paths = [
        './models/Tomato_doctor_mblnetv2.h5',
        './plant-doctor/Tomato_doctor_mblnetv2.h5'
    ]

    for path in [model_path] + fallback_paths:
        if os.path.exists(path):
            disease_model = tf.keras.models.load_model(path)
            return leaf_classifier, disease_model

    raise FileNotFoundError("Model file not found in expected paths.")

@st.cache_data
def load_knowledge():
    with open('final_crop_disease_knowledge_base.json') as f:
        return json.load(f)['diseases']

@st.cache_data
def load_class_indices():
    with open('class_indices.json') as f:
        return json.load(f)

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def load_or_create_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_device_id():
    if 'device_id' not in st.session_state:
        st.session_state.device_id = str(uuid.uuid4())
    return st.session_state.device_id

def add_to_history(device_id, image, result):
    history = load_or_create_history()
    
    if device_id not in history:
        history[device_id] = []
    
    img_byte_arr = image.tobytes()
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "image": img_byte_arr.hex(),
        "result": result,
        "disease": result.get("disease_name", "Healthy") if isinstance(result, dict) else "Unknown",
        "confidence": result.get("confidence", 0) if isinstance(result, dict) else 0
    }
    
    history[device_id].append(entry)
    
    if len(history[device_id]) > MAX_HISTORY_ENTRIES:
        history[device_id] = history[device_id][-MAX_HISTORY_ENTRIES:]
    
    save_history(history)
    return history[device_id]

def display_history(history):
    if not history:
        st.info("No previous scans found")
        return
    
    st.subheader("📅 Scan History")
    
    date_groups = {}
    for entry in sorted(history, key=lambda x: x['timestamp'], reverse=True):
        date = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d")
        if date not in date_groups:
            date_groups[date] = []
        date_groups[date].append(entry)
    
    for date, entries in date_groups.items():
        with st.expander(f"🗓️ {date}"):
            for entry in entries:
                time = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    try:
                        img_bytes = bytes.fromhex(entry['image'])
                        img = Image.frombytes('RGB', (224, 224), img_bytes)
                        st.image(img, width=100)
                    except:
                        st.image(Image.new('RGB', (100, 100)), width=100)  # Fixed line
                
                with col2:
                    if entry['disease'] == "Healthy":
                        st.success(f"⏰ {time}: ✅ Healthy ({entry['confidence']:.1f}%)")
                    else:
                        st.warning(f"⏰ {time}: ⚠️ {entry['disease']} ({entry['confidence']:.1f}%)")
                    
                    if st.button("View details", key=entry['timestamp']):
                        st.session_state.view_details = entry

def process_image(uploaded_file, device_id):
    try:
        leaf_classifier, disease_model = load_models()
        knowledge = load_knowledge()
        class_indices = load_class_indices()

        img = Image.open(uploaded_file).convert('RGB')
        img_array = preprocess_image(img)

        leaf_pred = leaf_classifier.predict(img_array, verbose=0)
        leaf_class = np.argmax(leaf_pred)
        leaf_confidence = leaf_pred[0][leaf_class]
        
        LEAF_CLASSES = {
            0: 'non_leaf',
            1: 'other_leaf',
            2: 'tomato_leaf'
        }
        
        current_leaf_class = LEAF_CLASSES[leaf_class]
        
        if current_leaf_class != 'tomato_leaf':
            if current_leaf_class == 'non_leaf':
                st.error("❌ This doesn't appear to be a leaf image.")
            else:
                st.error("❌ This appears to be a non-tomato leaf.")
            st.image(img, width=300)
            st.write(f"Classification: {current_leaf_class.replace('_', ' ').title()} ({leaf_confidence*100:.1f}% confidence)")
            return

        st.success("✓ Verified: Tomato leaf detected")
        
        with st.spinner("🔍 Analyzing for diseases..."):
            output = disease_model.predict(img_array, verbose=0)[0]

        class_idx = int(np.argmax(output))
        predicted_class = class_indices[str(class_idx)]
        info = knowledge[predicted_class]
        confidence = float(output[class_idx])

        result = format_results(predicted_class, info, confidence)
        
        st.image(img, width=300)
        display_results(predicted_class, info, confidence)
        
        add_to_history(device_id, img, result)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

def format_results(predicted_class, info, confidence):
    plant_type = predicted_class.split('___')[0].replace('_', ' ').title()
    result = {
        "confidence": confidence,
        "plant_type": plant_type,
        "status": "healthy" if 'healthy' in predicted_class.lower() else "diseased"
    }

    if result["status"] == "healthy":
        result["recommendations"] = {
            "general": "Tomato plant is healthy. Maintain clean fields and seed health.",
            "monitoring_advice": [
                "Inspect leaves for dark lesions weekly",
                "Apply fungicide preventively if wet conditions persist",
                "Monitor for early blight symptoms",
                "Ensure proper spacing between plants (18-24 inches)"
            ]
        }
    else:
        disease_name = predicted_class.split('___')[1].replace('_', ' ').title() if '___' in predicted_class else predicted_class.replace('_', ' ').title()
        result["disease_name"] = disease_name
        result["details"] = {
            "symptoms": info['symptoms'],
            "causes": info['causes'],
            "effects": info['effects']
        }
        result["treatments"] = info['treatments']

    return result

def display_results(predicted_class, info, confidence):
    plant_type = predicted_class.split('___')[0].replace('_', ' ').title()

    if 'healthy' in predicted_class.lower():
        st.balloons()
        st.success("✅ Healthy Tomato Leaf")
        st.markdown("""
        ### Recommendations
        Tomato plant is healthy. Maintain clean fields and seed health.
        ### Monitoring Advice
        - Inspect leaves for dark lesions weekly
        - Apply fungicide preventively if wet conditions persist
        - Monitor for early blight symptoms
        - Ensure proper spacing between plants (18-24 inches)
        """)
    else:
        disease_name = predicted_class.split('___')[1].replace('_', ' ').title() if '___' in predicted_class else predicted_class.replace('_', ' ').title()
        st.warning(f"⚠️ Detected: {disease_name} ({confidence*100:.1f}% confidence)")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Symptoms", "Prevention", "Treatment", "Chemical Details"])
        
        with tab1:
            st.markdown(f"""
            **Plant Type:** {plant_type}
            
            **Symptoms:**  
            {info['symptoms']}
            
            **Causes:**  
            {info['causes']}
            
            **Effects:**  
            {info['effects']}
            """)
            
        with tab2:
            st.markdown("### Prevention Methods")
            st.markdown("#### Cultural Practices")
            for method in info['treatments']['cultural']:
                st.markdown(f"- {method}")
                
        with tab3:
            st.markdown("### Treatment Options")
            
            if info['treatments']['chemical']:
                st.markdown("#### Chemical Treatment")
                chem = info['treatments']['chemical']
                
                st.markdown(f"""
                - **Product:** {chem['product']} 
                - **Dosage:** {chem['dosage']}
                - **Instructions:** {chem.get('note', 'N/A')}
                """)
            else:
                st.info("No chemical treatment recommended")
                
            if info['treatments']['mechanical']:
                st.markdown("#### Mechanical Treatment")
                for method in info['treatments']['mechanical']:
                    st.markdown(f"- {method}")
                
        with tab4:
            st.info("*⚠️CAUTION: Price estimates are approximate and may vary by store/region*")
                
            if info['treatments']['chemical']:
                chem = info['treatments']['chemical']

                st.markdown(f"""
                ### Detailed Chemical Information
                
                **🔎Product Name:**  
                *{chem['product']}*  
                
                **💰Approx. Market Price:**  
                *{chem.get('price', 'Not available')}* 
                
                **⚠️Safety Precautions:**  
                *{chem.get('safety', 'Wear protective gear during application')}*
                """)
            else:
                st.info("No chemical treatment details available")

# =============================================
# 2. Realistic Price Forecast Demo (Hidden Simulation)
# =============================================
def generate_forecast_data():
    """Generate realistic price data using only numpy"""
    dates = [datetime.today() + timedelta(days=i) for i in range(30)]
    base_price = 36.0
    # Create natural price movements
    trend = np.sin(np.linspace(0, 3*np.pi, 30)) * 1.8  # Seasonal pattern
    noise = np.random.normal(0, 0.3, 30)  # Market fluctuations
    prices = (base_price + trend + noise).round(2)
    return pd.DataFrame({
        "Date": dates,
        "Price (SZL/kg)": prices,
        "Min Expected": (prices - 0.8).round(2),
        "Max Expected": (prices + 0.8).round(2)
    })

def show_price_forecast():
    st.header("🍅 Tomato Price Forecast")
    
    # Generate and display data
    forecast_data = generate_forecast_data()
    
    # Use Streamlit's native line chart
    st.line_chart(
        forecast_data.set_index("Date"),
        y=["Price (SZL/kg)", "Min Expected", "Max Expected"],
        color=["#FF0000", "#888888", "#888888"]  # Red for main price, gray for bounds
    )
    
    # Key metrics
    current_price = forecast_data.iloc[0]["Price (SZL/kg)"]
    avg_price = forecast_data["Price (SZL/kg)"].mean().round(2)
    volatility = (forecast_data["Max Expected"] - forecast_data["Min Expected"]).mean().round(2)
    
    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"{current_price} SZL/kg")
    col2.metric("30-Day Average", f"{avg_price} SZL/kg")
    st.metric("Daily Volatility", f"±{volatility} SZL/kg")
    
    # Data table
    with st.expander("View Detailed Forecast"):
        st.dataframe(
            forecast_data,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Price (SZL/kg)": st.column_config.NumberColumn("Price", format="%.2f")
            },
            hide_index=True
        )

# To use in your app:
 show_price_forecast()

# =============================================
# 3. Integrated App with Seamless Navigation
# =============================================
def main():
    # Initialize session state
    if 'device_id' not in st.session_state:
        st.session_state.device_id = str(uuid.uuid4())
    
    # Sidebar navigation
    st.sidebar.image("https://via.placeholder.com/150x50?text=Tomato+Pro", width=150)
    app_mode = st.sidebar.radio(
        "Select Module",
        ("Disease Diagnosis", "Market Forecast"),
        index=0
    )
    
    # Disease Diagnosis Mode
    if app_mode == "Disease Diagnosis":
        st.title("🍅 Tomato Disease Diagnosis")
        st.markdown("Upload a clear photo of a tomato leaf for instant analysis")
        
        # Display history
        history = load_or_create_history().get(st.session_state.device_id, [])
        display_history(history)
        
        # Handle detail view
        if 'view_details' in st.session_state:
            entry = st.session_state.view_details
            st.subheader(f"Detailed results from {datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M')}")
            
            if isinstance(entry['result'], dict):
                display_results(
                    f"{entry['result']['plant_type']}___{entry['result'].get('disease_name','healthy')}",
                    {'symptoms': '', 'causes': '', 'effects': '', 'treatments': entry['result'].get('treatments',{})},
                    entry['confidence']
                )
            else:
                st.warning("Full details not available for this entry")
            
            if st.button("Back to current scan"):
                del st.session_state.view_details
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            process_image(uploaded_file, st.session_state.device_id)
    
    # Market Forecast Mode
    else:
        show_market_forecast()

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    main()
