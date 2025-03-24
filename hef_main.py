import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, HailoRT, InferVStreams, VDevice)

# Load YOLOv8 HEF model
hef_path = "/home/panther/hailo-rpi5-examples/resources/yolov8m_h8l.hef"
hef = HEF(hef_path)

# Initialize Hailo device
device = VDevice()
network_group = device.create_network_group(hef)
vstreams = InferVStreams(network_group)

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

def preprocess(frame):
    """Preprocess the frame for inference."""
    frame_resized = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(frame_rgb, axis=0).astype(np.float32)

def postprocess(results, frame):
    """Post-process and display results."""
    for result in results[0]:
        x1, y1, x2, y2, conf, class_id = result[:6]
        if conf > 0.5:
            label = f"Class {int(class_id)}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

# Start video pipeline
while True:
    frame = picam2.capture_array()
    preprocessed_frame = preprocess(frame)

    # Run inference on Hailo AI Kit
    results = vstreams.infer(preprocessed_frame)

    # Post-process results
    output_frame = postprocess(results, frame)

    # Show output
    cv2.imshow("Hailo YOLOv8 Inference", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
