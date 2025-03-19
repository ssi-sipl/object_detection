import cv2
import time
import threading
import numpy as np
from queue import Queue
from picamera2 import Picamera2
from ultralytics import YOLO

# Configuration
INPUT_WIDTH = 640
INPUT_HEIGHT = 480
INFERENCE_WIDTH = 320
INFERENCE_HEIGHT = 240
PROCESS_EVERY_N_FRAMES = 2
CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = '../yolov5/yolov8n.pt'
ENABLE_THREADING = True
FRAME_RATE = 15

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (INPUT_WIDTH, INPUT_HEIGHT)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.set_controls({"FrameRate": FRAME_RATE})
picam2.start()

# Load YOLOv8 model
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
# Warm up the model
dummy_frame = np.zeros((INFERENCE_HEIGHT, INFERENCE_WIDTH, 3), dtype=np.uint8)
_ = model.predict(dummy_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
print("Model loaded successfully!")

# Global variables
frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)
processing_frame = False
fps_counter = []
current_fps = 0

def calculate_fps():
    global fps_counter, current_fps
    now = time.time()
    fps_counter.append(now)
    
    # Keep only the last 30 frames for FPS calculation
    if len(fps_counter) > 30:
        fps_counter.pop(0)
    
    if len(fps_counter) >= 2:
        current_fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
    return current_fps

def process_frames():
    global processing_frame
    frame_count = 0
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_count += 1
            
            # Only process every Nth frame
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                processing_frame = True
                
                # Resize for inference
                resized_frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
                
                # Run inference
                start_time = time.time()
                results = model.predict(resized_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                inference_time = time.time() - start_time
                
                # Scale bounding boxes back to original size
                scaled_results = []
                for result in results:
                    boxes = []
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        # Scale coordinates back to original size
                        x1 = int(x1 * (INPUT_WIDTH / INFERENCE_WIDTH))
                        y1 = int(y1 * (INPUT_HEIGHT / INFERENCE_HEIGHT))
                        x2 = int(x2 * (INPUT_WIDTH / INFERENCE_WIDTH))
                        y2 = int(y2 * (INPUT_HEIGHT / INFERENCE_HEIGHT))
                        
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        boxes.append((x1, y1, x2, y2, cls, conf))
                    
                    scaled_results.append({
                        'boxes': boxes,
                        'names': result.names
                    })
                
                # Put results in queue
                if not result_queue.full():
                    result_queue.put((scaled_results, inference_time))
                
                processing_frame = False

# Start processing thread if enabled
if ENABLE_THREADING:
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()
    print("Processing thread started")

# Main loop
try:
    frame_count = 0
    last_results = None
    
    while True:
        # Capture frame
        frame = picam2.capture_array()
        frame_count += 1
        
        # Put frame in queue for processing if threading is enabled
        if ENABLE_THREADING:
            if frame_queue.empty() and not processing_frame:
                frame_queue.put(frame.copy())
                
            # Get results if available
            if not result_queue.empty():
                last_results, inference_time = result_queue.get()
        else:
            # Process directly if threading is disabled
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                resized_frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
                start_time = time.time()
                results = model.predict(resized_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                inference_time = time.time() - start_time
                
                # Scale results
                last_results = []
                for result in results:
                    boxes = []
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        x1 = int(x1 * (INPUT_WIDTH / INFERENCE_WIDTH))
                        y1 = int(y1 * (INPUT_HEIGHT / INFERENCE_HEIGHT))
                        x2 = int(x2 * (INPUT_WIDTH / INFERENCE_WIDTH))
                        y2 = int(y2 * (INPUT_HEIGHT / INFERENCE_HEIGHT))
                        
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        boxes.append((x1, y1, x2, y2, cls, conf))
                    
                    last_results.append({
                        'boxes': boxes,
                        'names': result.names
                    })
        
        # Draw results on frame if available
        if last_results:
            for result in last_results:
                for x1, y1, x2, y2, cls, conf in result['boxes']:
                    label = result['names'][cls]
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS
        fps = calculate_fps()
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the video feed
        cv2.imshow("YOLOv8 Detection", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Resources released.")