import numpy as np
import cv2
from multiprocessing import Process
from picamera2 import Picamera2
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType
)

# Initialize Hailo Device (VDevice)
target = VDevice()

# Load compiled HEFs to device
model_name = '/usr/share/hailo-models/yolov8s_h8l.hef'
hef_path = '../hefs/{}.hef'.format(model_name)
hef = HEF(model_name)

# Configure network groups
configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

# Create input and output virtual streams params
input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)

# Define dataset params
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]
image_height, image_width, channels = input_vstream_info.shape

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

def preprocess(frame):
    """Preprocess the frame for inference."""
    frame_resized = cv2.resize(frame, (image_width, image_height))
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

# Create input and output virtual streams
input_vstreams = InputVStreams(network_group, list(input_vstreams_params))
output_vstreams = OutputVStreams(network_group, list(output_vstreams_params))

# Inference loop - process real-time frames
try:
    while True:
        frame = picam2.capture_array()
        preprocessed_frame = preprocess(frame)

        # Send frame to Hailo AI Kit
        input_vstreams.write(preprocessed_frame)

        # Get inference results
        results = output_vstreams.read()

        # Post-process results and display
        output_frame = postprocess(results, frame)
        cv2.imshow("Hailo YOLOv8 Inference", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Clean up and release resources
    cv2.destroyAllWindows()
    # input_vstreams.close()
    # output_vstreams.close()
    