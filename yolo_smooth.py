import cv2
import threading
import queue
from picamera2 import Picamera2
from ultralytics import YOLO
import time

# Load YOLOv5n Model
model = YOLO('../yolov5/yolov5n.pt')
cv2.setNumThreads(4)

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Queue for frames
frame_queue = queue.Queue(maxsize=5)
processed_frame = None
lock = threading.Lock()


def capture_frames():
    """ Continuously capture frames and push to queue """
    while True:
        frame = picam2.capture_array()
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.01)  # Small delay to reduce CPU load

def process_frames():
    """ Continuously grab latest frame and run YOLO """
    global processed_frame
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # Run YOLO inference
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

            # Update the processed frame
            with lock:
                processed_frame = frame

        time.sleep(0.05)  # Slight delay to reduce CPU usage

def display_frames():
    """ Display frames smoothly even if boxes lag a bit """
    global processed_frame
    while True:
        with lock:
            if processed_frame is not None:
                cv2.imshow("YOLOv5 Detection", processed_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



# Create and start threads
thread1 = threading.Thread(target=capture_frames, daemon=True)
thread2 = threading.Thread(target=process_frames, daemon=True)
thread3 = threading.Thread(target=display_frames, daemon=True)

thread1.start()
thread2.start()
thread3.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
    cv2.destroyAllWindows()
    picam2.stop()
    exit()  