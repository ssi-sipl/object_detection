import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (HEF, InferVStreams, VDevice, VStreamsParams)

# Load YOLOv8 HEF model
hef_path = "/usr/share/hailo-models/yolov8s_h8l.hef"
hef = HEF(hef_path)

# Open VDevice and configure the network group
with VDevice() as device:
    network_groups = device.configure(hef)
    network_group = network_groups[0]  # Use the first network group
    network_group.activate()

    # Get input and output vstream info
    input_vstream_info = network_group.get_input_vstream_infos()[0]
    output_vstream_info = network_group.get_output_vstream_infos()[0]

    # Define VStreamsParams for input and output
    input_vstreams_params = VStreamsParams.from_vstream_infos(input_vstream_info)
    output_vstreams_params = VStreamsParams.from_vstream_infos(output_vstream_info)

    # Create input and output vstreams with params
    input_vstreams = InferVStreams(network_group, input_vstream_info, input_vstreams_params)
    output_vstreams = InferVStreams(network_group, output_vstream_info, output_vstreams_params)

    # Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    # Start Video Capture Loop
    while True:
        # Capture frame from Picamera2
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to YOLOv8 input size
        resized_frame = cv2.resize(frame_rgb, (640, 640))  # YOLOv8 expects 640x640
        input_frame = np.expand_dims(resized_frame, axis=0)

        # Run inference
        input_vstreams.write(input_frame)
        results = output_vstreams.read()

        # Post-process results
        detections = results[0].as_numpy().reshape((-1, 6))  # YOLOv8 output: [x, y, w, h, conf, class]
        for detection in detections:
            x, y, w, h, conf, class_id = detection
            if conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class: {int(class_id)}, Conf: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with bounding boxes
        cv2.imshow("YOLOv8 + Hailo on RPi5", frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cv2.destroyAllWindows()
    network_group.deactivate()
