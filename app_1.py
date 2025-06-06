import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
import os
from datetime import datetime
import uuid

# Configure page
st.set_page_config(
    page_title="Tomato Disease Doctor",
    page_icon="🌱",
    layout="centered"
)

# History configuration
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
            st.info(f"Model loaded from: {path}")
            disease_model = tf.keras.models.load_model(path)
            return leaf_classifier, disease_model

    error_message = "Model file not found in expected paths."
    st.error(error_message)
    raise FileNotFoundError(error_message)

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
    """Load existing history or create new history file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_history(history):
    """Save history to file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_device_id():
    """Get or create a device ID using Streamlit's session state"""
    if 'device_id' not in st.session_state:
        st.session_state.device_id = str(uuid.uuid4())
    return st.session_state.device_id

def add_to_history(device_id, image, result):
    """Add a new entry to the history"""
    history = load_or_create_history()
    
    if device_id not in history:
        history[device_id] = []
    
    # Convert image to bytes for storage
    img_byte_arr = image.tobytes()
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "image": img_byte_arr.hex(),  # Store as hex string
        "result": result,
        "disease": result.get("disease_name", "Healthy") if isinstance(result, dict) else "Unknown",
        "confidence": result.get("confidence", 0) if isinstance(result, dict) else 0
    }
    
    history[device_id].append(entry)
    
    # Keep only the most recent entries
    if len(history[device_id]) > MAX_HISTORY_ENTRIES:
        history[device_id] = history[device_id][-MAX_HISTORY_ENTRIES:]
    
    save_history(history)
    return history[device_id]

def display_history(history):
    """Display history in Streamlit"""
    if not history:
        st.info("No previous scans found")
        return
    
    st.subheader("📅 Scan History")
    
    # Group by date
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
                    # Try to display image (might fail for very old entries)
                    try:
                        img_bytes = bytes.fromhex(entry['image'])
                        img = Image.frombytes('RGB', (224, 224), img_bytes)
                        st.image(img, width=100)
                    except:
                        st.image(Image.new('RGB', (100, 100), color='gray'), width=100)
                
                with col2:
                    if entry['disease'] == "Healthy":
                        st.success(f"⏰ {time}: ✅ Healthy ({entry['confidence']:.1f}%)")
                    else:
                        st.warning(f"⏰ {time}: ⚠️ {entry['disease']} ({entry['confidence']:.1f}%)")
                    
                    if st.button("View details", key=entry['timestamp']):
                        st.session_state.view_details = entry

def main():
    st.title("🍅🌿 Tomato Disease Diagnosis and Doctor 🔬🩺")
    st.markdown("Upload a CLEAR photo of a TOMATO LEAF for instant analysis")

    # Initialize device ID
    device_id = get_device_id()
    
    # Display history first
    history = load_or_create_history().get(device_id, [])
    display_history(history)

    # Handle detail view if a history entry was clicked
    if 'view_details' in st.session_state:
        entry = st.session_state.view_details
        st.subheader(f"Detailed results from {datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        
        if isinstance(entry['result'], dict):
            display_detailed_results(entry['result'])
        else:
            st.warning("Full details not available for this entry")
        
        if st.button("Back to current scan"):
            del st.session_state.view_details
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        process_image(uploaded_file, device_id)

def process_image(uploaded_file, device_id):
    try:
        # Validate file extension
        if not uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            st.error("Invalid file type. Please upload a .jpg, .jpeg, or .png file.")
            return

        leaf_classifier, disease_model = load_models()
        knowledge = load_knowledge()
        class_indices = load_class_indices()

        # Load and preprocess image
        img = Image.open(uploaded_file).convert('RGB')
        img_array = preprocess_image(img)

        # First stage: Leaf classification
        leaf_pred = leaf_classifier.predict(img_array, verbose=0)
        leaf_class = np.argmax(leaf_pred)
        leaf_confidence = leaf_pred[0][leaf_class]
        
        # Leaf class mapping
        LEAF_CLASSES = {
            0: 'non_leaf',
            1: 'other_leaf',
            2: 'tomato_leaf'
        }
        
        current_leaf_class = LEAF_CLASSES[leaf_class]
        
        if current_leaf_class != 'tomato_leaf':
            if current_leaf_class == 'non_leaf':
                st.error("❌ This doesn't appear to be a leaf image. Please upload a clear photo of a tomato leaf.")
            else:
                st.error("❌ This appears to be a non-tomato leaf. Please upload a tomato leaf for disease diagnosis.")
            st.image(img, width=300)
            st.write(f"Classification: {current_leaf_class.replace('_', ' ').title()} ({leaf_confidence*100:.1f}% confidence)")
            return

        # Only proceed with disease detection if it's a tomato leaf
        st.success("✓ Verified: Tomato leaf detected")
        
        # Second stage: Disease detection
        with st.spinner("🔍 Analyzing for diseases..."):
            output = disease_model.predict(img_array, verbose=0)[0]

        class_idx = int(np.argmax(output))
        predicted_class = class_indices[str(class_idx)]
        info = knowledge[predicted_class]
        confidence = float(output[class_idx])

        # Format results
        result = format_results(predicted_class, info, confidence)
        
        # Display current results
        st.image(img, width=300)
        display_results(predicted_class, info, confidence)
        
        # Add to history
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

def display_detailed_results(result):
    """Display detailed results from history"""
    if result['status'] == "healthy":
        st.success("✅ Healthy Tomato Leaf")
        st.markdown(f"**Confidence:** {result['confidence']*100:.1f}%")
        st.markdown("""
        ### Recommendations
        Tomato plant is healthy. Maintain clean fields and seed health.
        """)
    else:
        st.warning(f"⚠️ Detected: {result['disease_name']} ({result['confidence']*100:.1f}% confidence)")
        
        tab1, tab2, tab3 = st.tabs(["Symptoms", "Prevention", "Treatment"])
        
        with tab1:
            st.markdown(f"""
            **Symptoms:**  
            {result['details']['symptoms']}
            
            **Causes:**  
            {result['details']['causes']}
            
            **Effects:**  
            {result['details']['effects']}
            """)
            
        with tab2:
            st.markdown("### Prevention Methods")
            if 'treatments' in result and 'cultural' in result['treatments']:
                st.markdown("#### Cultural Practices")
                for method in result['treatments']['cultural']:
                    st.markdown(f"- {method}")
            else:
                st.info("No prevention methods recorded")
                
        with tab3:
            st.markdown("### Treatment Options")
            if 'treatments' in result:
                if 'chemical' in result['treatments'] and result['treatments']['chemical']:
                    st.markdown("#### Chemical Treatment")
                    chem = result['treatments']['chemical']
                    st.markdown(f"""
                    - **Product:** {chem.get('product', 'N/A')} 
                    - **Dosage:** {chem.get('dosage', 'N/A')}
                    """)
                
                if 'mechanical' in result['treatments'] and result['treatments']['mechanical']:
                    st.markdown("#### Mechanical Treatment")
                    for method in result['treatments']['mechanical']:
                        st.markdown(f"- {method}")

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    main()
