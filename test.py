from picamera2 import Picamera2
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

# Load TFLite model
interpreter = tflite.Interpreter(model_path="tflite_model/detect.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # Example: (1, 300, 300, 3)

# Load labels
with open("tflite_model/labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize PiCamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

    # Preprocess image
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img_array = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    # Draw bounding boxes
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.3:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            
            class_id = int(class_ids[i])
            if class_id < len(labels):  # Ensure valid class index
                label = f"{labels[class_id]}: {int(scores[i] * 100)}%"
            else:
                label = f"Unknown: {int(scores[i] * 100)}%"

            # Draw rectangle and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Object Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
