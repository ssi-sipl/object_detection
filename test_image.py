import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# --- Load the TFLite model ---
interpreter = tflite.Interpreter(model_path="tflite_model/detect.tflite")
interpreter.allocate_tensors()

# --- Get model input and output details ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Load and preprocess the test image ---
image_path = "test.jpg"
image = cv2.imread(image_path)

# Resize image to match model's input shape
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]
image_resized = cv2.resize(image, (width, height))

# Normalize and prepare input data
input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)
# --- Run inference ---
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# --- Get detection results ---
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

# --- Load labels ---
def load_labels(path):
    with open(path, 'r') as f:
        
        return [line.strip() for line in f.readlines()]

labels = load_labels('tflite_model/labelmap1.txt')

# --- Draw boxes and labels on the image ---
for i in range(len(scores)):
    if scores[i] > 0.5:  # Confidence threshold
        y_min, x_min, y_max, x_max = boxes[i]
        x_min, x_max = int(x_min * width), int(x_max * width)
        y_min, y_max = int(y_min * height), int(y_max * height)
        
        class_id = int(classes[i])
        label = labels.get(class_id, "???")
        
        # Draw bounding box
        cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw label with confidence
        label_text = f"{label} ({scores[i] * 100:.2f}%)"
        cv2.putText(image_resized, label_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# --- Save and display the output image ---
output_path = "output.jpg"
cv2.imwrite(output_path, image_resized)
cv2.imshow("Result", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Detection complete! Check the output image: {output_path}")
