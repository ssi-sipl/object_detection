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
INFERENCE_WIDTH = 256  # Reduced for faster processing
INFERENCE_HEIGHT = 192  # Reduced for faster processing
PROCESS_EVERY_N_FRAMES = 1  # Process every frame for smooth tracking
CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = '../yolov5/yolov8n.pt'
ENABLE_THREADING = True
FRAME_RATE = 15
ENABLE_TRACKER = True  # Enable OpenCV tracker for smoother tracking

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
model.conf = CONFIDENCE_THRESHOLD  # Confidence threshold
model.iou = 0.45  # IoU threshold for NMS

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
last_boxes = {}  # Dictionary to store last positions of objects
trackers = {}  # Dictionary to store object trackers
last_frame_count = 0  # Track which frame was last processed

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
    global processing_frame, last_frame_count
    frame_count = 0
    
    while True:
        if not frame_queue.empty():
            frame, frame_id = frame_queue.get()
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
                    result_queue.put((scaled_results, inference_time, frame_id))
                
                processing_frame = False

def create_tracker(tracker_type="KCF"):
    """Create a tracker based on the specified type."""
    if tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT_create()
    
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    else:
        return cv2.legacy.TrackerKCF_create()  # Default to KCF

def update_trackers(frame):
    """Update all trackers and draw their bounding boxes."""
    global trackers
    
    # List of trackers to remove (if tracking fails)
    to_remove = []
    
    for track_id, tracker in trackers.items():
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            cls_id = int(track_id.split('_')[0])
            label = track_id.split('_')[1]
            
            # Draw tracking box (blue)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}", (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Update last_boxes with current position
            last_boxes[track_id] = (x, y, x+w, y+h)
        else:
            to_remove.append(track_id)
    
    # Remove failed trackers
    for track_id in to_remove:
        trackers.pop(track_id, None)

def update_trackers_with_motion_prediction(frame):
    """Update trackers with motion prediction between frames."""
    global last_boxes
    
    for box_id, (x1, y1, x2, y2) in last_boxes.items():
        if box_id in trackers:
            continue  # Skip if we have an active tracker for this object
            
        # Calculate motion vector if we have historical data
        if box_id + "_prev" in last_boxes:
            prev_x1, prev_y1, prev_x2, prev_y2 = last_boxes[box_id + "_prev"]
            # Calculate motion vector
            dx = (x1 - prev_x1) * 1.2  # Adjust factor as needed
            dy = (y1 - prev_y1) * 1.2
            # Apply motion prediction
            x1 += int(dx)
            y1 += int(dy)
            x2 += int(dx)
            y2 += int(dy)
            
            # Draw predicted box (yellow)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Update box position
            last_boxes[box_id] = (x1, y1, x2, y2)
        
        # Store current position as previous for next frame
        last_boxes[box_id + "_prev"] = last_boxes[box_id]

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
    last_frame_with_detection = None
    tracker_refresh_interval = 10  # Refresh trackers every 10 frames with YOLO detections
    
    while True:
        # Capture frame
        frame = picam2.capture_array()
        frame_count += 1
        
        # Put frame in queue for processing if threading is enabled
        if ENABLE_THREADING:
            if frame_queue.empty() and not processing_frame:
                frame_queue.put((frame.copy(), frame_count))
                
            # Get results if available
            if not result_queue.empty():
                last_results, inference_time, processed_frame_count = result_queue.get()
                last_frame_count = processed_frame_count
                last_frame_with_detection = frame.copy()
                
                # Initialize trackers with new detections
                if ENABLE_TRACKER and last_results:
                    # Only completely reset trackers every few frames
                    if frame_count % tracker_refresh_interval == 0:
                        trackers = {}  # Clear existing trackers
                    
                    for result in last_results:
                        for x1, y1, x2, y2, cls, conf in result['boxes']:
                            label = result['names'][cls]
                            box_id = f"{cls}_{label}"
                            
                            # Create a new tracker for this object
                            if box_id not in trackers:
                                tracker = create_tracker(tracker_type="KCF")
                                w, h = x2 - x1, y2 - y1
                                try:
                                    tracker.init(frame, (x1, y1, w, h))
                                    trackers[box_id] = tracker
                                except:
                                    print(f"Failed to initialize tracker for {box_id}")
                            
                            # Store current position for motion prediction
                            last_boxes[box_id] = (x1, y1, x2, y2)
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
                
                last_frame_with_detection = frame.copy()
                
                # Initialize trackers with new detections
                if ENABLE_TRACKER and last_results:
                    # Only completely reset trackers every few frames
                    if frame_count % tracker_refresh_interval == 0:
                        trackers = {}  # Clear existing trackers
                    
                    for result in last_results:
                        for x1, y1, x2, y2, cls, conf in result['boxes']:
                            label = result['names'][cls]
                            box_id = f"{cls}_{label}"
                            
                            # Create a new tracker for this object
                            if box_id not in trackers:
                                tracker = create_tracker(tracker_type="KCF")
                                w, h = x2 - x1, y2 - y1
                                try:
                                    tracker.init(frame, (x1, y1, w, h))
                                    trackers[box_id] = tracker
                                except:
                                    print(f"Failed to initialize tracker for {box_id}")
                            
                            # Store current position for motion prediction
                            last_boxes[box_id] = (x1, y1, x2, y2)
        
        # First, update trackers if enabled
        if ENABLE_TRACKER:
            update_trackers(frame)
        
        # Update with motion prediction if needed
        if not ENABLE_TRACKER and last_boxes:
            update_trackers_with_motion_prediction(frame)
        
        # Then, draw YOLO detection results if available
        if last_results:
            for result in last_results:
                for x1, y1, x2, y2, cls, conf in result['boxes']:
                    label = result['names'][cls]
                    box_id = f"{cls}_{label}"
                    
                    # Skip drawing if we have an active tracker for this object
                    if ENABLE_TRACKER and box_id in trackers:
                        continue
                    
                    # Draw bounding box and label (green for YOLO detection)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS
        fps = calculate_fps()
        
        # Display FPS and other info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Trackers: {len(trackers)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the video feed
        cv2.imshow("YOLOv8 Detection", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Resources released.")