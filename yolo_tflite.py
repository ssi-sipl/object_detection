import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="../yolov5/best-fp16.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("../yolov5/labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 640)})
picam2.configure(config)
picam2.start()

# Get model input shape
input_shape = input_details[0]["shape"]
height, width = input_shape[1], input_shape[2]

print(f"Expected Input Shape: {input_shape}")


# Define minimum confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.5

print("📸 Starting Real-Time Object Detection...")

# Main Loop
while True:
    # Capture frame from PiCamera2
    # Capture frame and check dimensions
    # Get frame from PiCamera2
    frame = picam2.capture_array()

    # Convert to RGB if grayscale or RGBA
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Resize to model's input size
    resized_frame = cv2.resize(frame, (width, height))

    # Expand dimensions and convert to float32
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)

    # Normalize (if needed)
    input_data /= 255.0

    # Check shape before setting tensor
    print(f"Shape before setting tensor: {input_data.shape}")
    print(f"Type before setting tensor: {input_data.dtype}")

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()


    # Get results
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]["index"])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]["index"])[0]  # Confidence scores

    # Draw results on the frame
    for i in range(len(scores)):
        if scores[i] > CONFIDENCE_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])

            label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Object Detection", frame)

    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
picam2.stop()
print("🛑 Detection stopped. Bye!")
