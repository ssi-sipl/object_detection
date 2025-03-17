from picamera2 import Picamera2
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

# Load TFLite model
interpreter = tflite.Interpreter(model_path="tflite_model/efficientdet_d0.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details: ", input_details)
print("Output Detaisl: ", output_details)
# Load labels file
with open("tflite_model/labelmap1.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_frame(frame):
    
    img = cv2.resize(frame, (1, 1))  # Resize to (1, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype(np.uint8)  # Convert to uint8
    return img

def run_inference(frame):
    """ Run inference on a single frame """
    img = preprocess_frame(frame)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])  # Bounding boxes
    class_probs = interpreter.get_tensor(output_details[3]['index'])  # Class probabilities
    num_detections = interpreter.get_tensor(output_details[2]['index'])  # Number of detections

    return np.squeeze(boxes), np.squeeze(class_probs), int(num_detections[0])

def draw_boxes(frame, boxes, class_probs, num_detections, threshold=0.5):
    """ Draw bounding boxes on frame """
    h, w, _ = frame.shape

    for i in range(num_detections):
        class_id = np.argmax(class_probs[i])  # Get highest probability class
        confidence = class_probs[i][class_id]  # Confidence score

        if confidence > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            
            # Convert normalized coordinates to pixel values
            xmin, xmax = int(xmin * w), int(xmax * w)
            ymin, ymax = int(ymin * h), int(ymax * h)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR

    boxes, class_probs, num_detections = run_inference(frame)
    frame = draw_boxes(frame, boxes, class_probs, num_detections)

    
    # Display the frame
    cv2.imshow("EfficientDet Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
