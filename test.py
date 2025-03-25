import cv2
import numpy as np
import picamera2
import libcamera
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams, 
    ConfigureParams, InputVStreamParams, OutputVStreamParams, 
    InputVStreams, OutputVStreams, FormatType
)

class HailoYOLOInference:
    def __init__(self, hef_path='/usr/share/hailo-models/yolov8s_h8l.hef'):
        # Initialize Hailo device and model
        self.target = VDevice()
        self.hef = HEF(hef_path)
        
        # Configure network groups
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, 
            interface=HailoStreamInterface.PCIe
        )
        network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = network_groups[0]
        
        # Create input and output virtual stream parameters
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.UINT8
        )
        
        # Get input and output stream info
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        
        # Input image dimensions
        self.input_height, self.input_width, _ = self.input_vstream_info.shape

    def preprocess_frame(self, frame):
        # Resize frame to model input size
        resized_frame = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Normalize pixel values (assuming model expects float32)
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Ensure correct channel order and shape
        preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
        
        return preprocessed_frame

    def run_inference(self, preprocessed_frame):
        # Create input and output virtual streams
        with InputVStreams(self.network_group, self.input_vstreams_params) as input_vstreams, \
             OutputVStreams(self.network_group, self.output_vstreams_params) as output_vstreams:
            
            # Run inference
            input_vstreams[0].send(preprocessed_frame)
            outputs = output_vstreams[0].recv()
            
            return outputs

    def postprocess_results(self, outputs, original_frame):
        # Implement your specific postprocessing logic here
        # This depends on the exact output format of your YOLOv8 model
        # Typical steps include:
        # 1. Decode bounding boxes
        # 2. Apply confidence thresholds
        # 3. Draw bounding boxes on the frame
        
        # Example placeholder (replace with actual logic):
        print(f"Inference output shape: {outputs.shape}")
        
        return original_frame

def main():
    # Setup Picamera2
    picam2 = picamera2.Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080)},
        lores={"size": (640, 480)},
        display="lores"
    )
    picam2.configure(config)
    
    # Initialize Hailo inference
    hailo_inference = HailoYOLOInference(hef_path='/path/to/yolov8_h8l.hef')
    
    # Start camera
    picam2.start()
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Preprocess frame
            preprocessed_frame = hailo_inference.preprocess_frame(frame)
            
            # Run inference
            outputs = hailo_inference.run_inference(preprocessed_frame)
            
            # Postprocess and visualize results
            result_frame = hailo_inference.postprocess_results(outputs, frame)
            
            # Display frame
            cv2.imshow('Hailo YOLOv8 Inference', result_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()