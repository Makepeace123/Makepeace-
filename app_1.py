from PIL import Image
import numpy as np
import json
import tensorflow as tf
import os
import sys

def load_models():
    """Load both the leaf classifier and disease detection models"""
    try:
        # Load leaf classifier model
        leaf_classifier_path = 'leaf_classifier_mobilenetv2.h5'
        if not os.path.exists(leaf_classifier_path):
            raise FileNotFoundError(f"Leaf classifier model not found at {leaf_classifier_path}")
        leaf_classifier = tf.keras.models.load_model(leaf_classifier_path)
        
        # Load disease detection model as TFLite
        model_paths = [
            'model.tflite',
            './models/model.tflite',
            './plant-doctor/model.tflite'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"Loading TFLite model from: {path}")
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                return leaf_classifier, interpreter
        
        raise FileNotFoundError("TFLite model not found in any of the expected paths")
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def load_knowledge():
    """Load the disease knowledge base"""
    try:
        with open('final_crop_disease_knowledge_base.json') as f:
            return json.load(f)['diseases']
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        raise

def load_class_indices():
    """Load the class indices mapping"""
    try:
        with open('class_indices.json') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading class indices: {str(e)}")
        raise

def preprocess_image(image):
    """Preprocess the image for model input"""
    try:
        img = image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise

def process_image(image_path):
    """Process an image through both models"""
    try:
        # Load models and data
        leaf_classifier, disease_interpreter = load_models()
        knowledge = load_knowledge()
        class_indices = load_class_indices()

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
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
        
        current_leaf_class = LEAF_CLASSES.get(leaf_class, 'unknown')
        
        if current_leaf_class != 'tomato_leaf':
            if current_leaf_class == 'non_leaf':
                return {"error": "This doesn't appear to be a leaf image."}
            else:
                return {"error": "This appears to be a non-tomato leaf."}

        # Second stage: Disease detection with TFLite
        input_details = disease_interpreter.get_input_details()
        output_details = disease_interpreter.get_output_details()
        
        # Verify input shape
        if img_array.shape != tuple(input_details[0]['shape']):
            return {"error": f"Input shape mismatch. Expected {input_details[0]['shape']}, got {img_array.shape}"}
        
        # Set input tensor and run inference
        disease_interpreter.set_tensor(input_details[0]['index'], img_array)
        disease_interpreter.invoke()
        
        # Get output tensor
        output = disease_interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Process results
        class_idx = int(np.argmax(output))
        predicted_class = class_indices.get(str(class_idx), "unknown")
        if predicted_class == "unknown":
            return {"error": f"Unknown class index {class_idx} detected"}
            
        info = knowledge.get(predicted_class, {})
        confidence = float(output[class_idx])

        return format_results(predicted_class, info, confidence)

    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

def format_results(predicted_class, info, confidence):
    """Format the detection results into a structured output"""
    try:
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
                "symptoms": info.get('symptoms', 'Not available'),
                "causes": info.get('causes', 'Not available'),
                "effects": info.get('effects', 'Not available')
            }
            result["treatments"] = info.get('treatments', {})

        return result
    except Exception as e:
        return {"error": f"Formatting error: {str(e)}"}

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    if len(sys.argv) != 2:
        print("Usage: python tomato_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    
    results = process_image(image_path)
    print(json.dumps(results, indent=2))
