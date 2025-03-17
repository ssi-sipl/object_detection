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

    output_details = interpreter.get_output_details()

    # Extract output data
    boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))  # (N, 4)
    class_ids = np.squeeze(interpreter.get_tensor(output_details[1]['index']))  # (N,)
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))  # (N,)
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])  # Integer

    # Iterate over detections
    for i in range(num_detections):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            class_id = int(class_ids[i])
            confidence = scores[i]
    # Display the frame
    cv2.imshow("EfficientDet Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
