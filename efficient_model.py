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

# Load labels file
with open("tflite_model/labelmap1.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image, input_shape):
    img_resized = cv2.resize(image, (input_shape[1], input_shape[2]))  # Resize
    img_array = np.expand_dims(img_resized, axis=0).astype(np.uint8)   # Expand dims
    return img_array

def run_inference(image):
    input_tensor = preprocess_image(image, input_details[0]['shape'])

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Extract output tensors
    boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    class_ids = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    print("boxes: ", len(boxes))
    print("class_ids: ", len(class_ids))
    print("scores: ", scores)
    # num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    return boxes, class_ids, scores

def draw_boxes(image, boxes, class_ids, scores, threshold=0.5):
    h, w, _ = image.shape

    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            label = f"{labels[int(class_ids[i])]}: {int(scores[i] * 100)}%"
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR

    boxes, class_ids, scores = run_inference(frame)
    frame = draw_boxes(frame, boxes, class_ids, scores)
    
    # Display the frame
    cv2.imshow("EfficientDet Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
