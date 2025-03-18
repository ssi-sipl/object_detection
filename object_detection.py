from picamera2 import Picamera2
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(model_path="tflite_model/detect.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print("OUTPUT: ", output_details)

# Load labels
with open("tflite_model/labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Total labels: {len(labels)}")

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())

# Start camera
picam2.start()

while True:
    frame = picam2.capture_array()  # Capture frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR

      # Preprocess image
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img_array = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    # print("boxes:", len(boxes))
    # print("classids: ", len(class_ids))
    # print("scores: ", len(scores))
    
    # Draw bounding boxes
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            label = f"{labels[int(class_ids[i])]}: {int(scores[i] * 100)}%"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          
            # print("class id: ",type(class_ids), "len: ", len(class_ids))
            # print("Class Id: ", class_id, "Label: ", label)

    cv2.imshow("Camera Feed", frame)  # Display
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv2.destroyAllWindows()
picam2.stop()

