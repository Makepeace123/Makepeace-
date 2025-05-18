import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
import os

# Configure page
st.set_page_config(
    page_title="Tomato Disease Doctor",
    page_icon="🌱",
    layout="centered"
)

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

def main():
    st.title("🍅🌿 Tomato Disease Diagnosis and Doctor 🔬🩺")
    st.markdown("Upload a CLEAR photo of a TOMATO LEAF for instant analysis")

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        process_image(uploaded_file)

def process_image(uploaded_file):
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

        st.image(img, width=300)
        display_results(predicted_class, info, confidence)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

