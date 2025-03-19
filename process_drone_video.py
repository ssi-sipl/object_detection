import cv2
import torch
from ultralytics import YOLO

# Load YOLOv5s model (small model for speed)
model = YOLO('yolov5s.pt')

# Input and output file paths
input_video_path = 'videos/02.mp4'
output_video_path = 'output_videos/output.mp4'

# Open video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference on the frame
    results = model.predict(frame, conf=0.5, verbose=False)
    print(results)

    # Draw bounding boxes and labels
    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = obj
        label = f"{model.names[int(cls)]} ({conf:.2f})"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Processed video saved as '{output_video_path}'")
