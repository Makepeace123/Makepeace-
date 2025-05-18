from PIL import Image
import numpy as np
import json
import tensorflow as tf
import os
import sys

def load_models():
    # Load leaf classifier model (still using .h5 as example)
    leaf_classifier = tf.keras.models.load_model('leaf_classifier_mobilenetv2.h5')
    
    # Load disease detection model as TFLite
    model_path = 'model.tflite'
    fallback_paths = [
        './models/model.tflite',
        './plant-doctor/model.tflite'
    ]

    for path in [model_path] + fallback_paths:
        if os.path.exists(path):
            print(f"TFLite model loaded from: {path}")
            
            # Load TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            
            return leaf_classifier, interpreter

    error_message = "Model file not found in expected paths."
    raise FileNotFoundError(error_message)

def load_knowledge():
    with open('final_crop_disease_knowledge_base.json') as f:
        return json.load(f)['diseases']

def load_class_indices():
    with open('class_indices.json') as f:
        return json.load(f)

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def process_image(image_path):
    try:
        leaf_classifier, disease_interpreter = load_models()
        knowledge = load_knowledge()
        class_indices = load_class_indices()

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_array = preprocess_image(img)

        # First stage: Leaf classification (still using Keras model)
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
                return {"error": "This doesn't appear to be a leaf image."}
            else:
                return {"error": "This appears to be a non-tomato leaf."}

        # Second stage: Disease detection with TFLite
        # Get input and output tensors
        input_details = disease_interpreter.get_input_details()
        output_details = disease_interpreter.get_output_details()
        
        # Set input tensor
        disease_interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        disease_interpreter.invoke()
        
        # Get output tensor
        output = disease_interpreter.get_tensor(output_details[0]['index'])[0]
        
        class_idx = int(np.argmax(output))
        predicted_class = class_indices[str(class_idx)]
        info = knowledge[predicted_class]
        confidence = float(output[class_idx])

        return format_results(predicted_class, info, confidence)

    except Exception as e:
        return {"error": str(e)}

def format_results(predicted_class, info, confidence):
    plant_type = predicted_class.split('___')[0].replace('_', ' ').title()
    result = {
        "confidence": confidence,
        "plant_type": plant_type
    }

    if 'healthy' in predicted_class.lower():
        result["status"] = "healthy"
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
        result["status"] = "diseased"
        result["disease_name"] = disease_name
        result["details"] = {
            "symptoms": info['symptoms'],
            "causes": info['causes'],
            "effects": info['effects']
        }
        result["treatments"] = info['treatments']

    return result

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    results = process_image(image_path)
    print(json.dumps(results, indent=2))
