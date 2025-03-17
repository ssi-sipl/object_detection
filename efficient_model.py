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
input_shape = input_details[0]['shape']

# Load labels file
with open("tflite_model/labelmap1.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR

    # Preprocess image for EfficientDet
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img_array = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get detection results
    boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))  # (num_detections, 4)
    class_ids = np.squeeze(interpreter.get_tensor(output_details[1]['index']))  # (num_detections,)
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))  # (num_detections,)

    # Ensure scores is an iterable list
    if scores.ndim == 0:  # If scores is a single value instead of an array
        scores = [scores]
        class_ids = [class_ids]
        boxes = [boxes]

    # Draw bounding boxes
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            label = f"{labels[int(class_ids[i])]}: {int(scores[i] * 100)}%"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("EfficientDet Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
