import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import threading
import time

# Load YOLOv8 model
model = YOLO('../yolov5/yolov5n.pt')

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Global variables
latest_frame = None
processed_frame = None
lock = threading.Lock()

def capture_frames():
    global latest_frame
    while True:
        frame = picam2.capture_array()
        with lock:
            latest_frame = frame.copy()
        time.sleep(0.01)  # Small delay to reduce CPU load

def process_frames():
    global latest_frame, processed_frame
    while True:
        with lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()
        
        if frame_copy is not None:
            # Run YOLO inference
            results = model.predict(frame_copy, conf=0.5, verbose=False)
            
            # Draw bounding boxes on frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls[0])]
                    conf = box.conf[0]
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_copy, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update the processed frame
            with lock:
                processed_frame = frame_copy
        time.sleep(0.05)  # Slight delay to reduce CPU usage

# Start Threads
thread1 = threading.Thread(target=capture_frames, daemon=True)
thread2 = threading.Thread(target=process_frames, daemon=True)

thread1.start()
thread2.start()

# Display Loop
while True:
    with lock:
        if processed_frame is not None:
            cv2.imshow("YOLOv5 Detection", processed_frame)
        else:
            cv2.imshow("YOLOv5 Detection", latest_frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
thread1.join()
thread2.join()
