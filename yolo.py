import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('../yolov5/yolov5n.pt')

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Start Video Capture
while True:
    frame = picam2.capture_array()
    
    # Run YOLO on frame
    results = model.predict(frame, conf=0.5, verbose=False)
    
    # Draw bounding boxes on frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = box.conf[0]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("YOLOv8 Detection", frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
