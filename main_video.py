import cv2
import os
import time
import threading
import numpy as np
from queue import Queue
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
VIDEO_DIRECTORY = './videos'  # Directory containing video files
OUTPUT_DIRECTORY = './output_videos'  # Directory to save processed videos

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

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

def process_video(video_path):
    global frame_queue, result_queue
    
    # Get the video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_DIRECTORY, f"{video_name}_processed.mp4")
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Start processing thread if enabled
    if ENABLE_THREADING:
        processing_thread = threading.Thread(target=process_frames)
        processing_thread.daemon = True
        processing_thread.start()
        print(f"Processing thread started for {video_name}")
    
    # Process video
    try:
        frame_count = 0
        last_results = None
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Display progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing {video_name}: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Resize frame to target dimensions if necessary
            if frame.shape[1] != INPUT_WIDTH or frame.shape[0] != INPUT_HEIGHT:
                frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
            
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
            
            # Write frame to output video
            out.write(frame)
            
            # Show the video feed
            cv2.imshow(f"Processing {video_name}", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        print(f"Finished processing {video_name}")
    
    except Exception as e:
        print(f"Error processing {video_name}: {e}")
    
    finally:
        # Release resources
        video.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    # Get all video files in the directory
    video_files = [os.path.join(VIDEO_DIRECTORY, f) for f in os.listdir(VIDEO_DIRECTORY) 
                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {VIDEO_DIRECTORY}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    for video_path in video_files:
        print(f"Processing video: {video_path}")
        process_video(video_path)

if __name__ == "__main__":
    main()