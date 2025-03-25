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
    def __init__(self, hef_path):
        # Initialize Hailo device and model
        self.target = VDevice()
        self.hef = HEF(hef_path)
        
        # Configure network groups
        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, 
            interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        
        # Create network group params
        self.network_group_params = self.network_group.create_params()
        
        # Get input and output stream info
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        
        # Print input and output stream information for debugging
        print(f"Input stream shape: {self.input_vstream_info.shape}")
        print(f"Input stream name: {self.input_vstream_info.name}")
        print(f"Number of output streams: {len(self.output_vstream_infos)}")
        for i, out_stream in enumerate(self.output_vstream_infos):
            print(f"Output stream {i} name: {out_stream.name}, shape: {out_stream.shape}")
        
        # Create input and output virtual stream parameters
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.FLOAT32
        )
        
        # Input image dimensions
        self.input_height, self.input_width, self.input_channels = self.input_vstream_info.shape

    def preprocess_frame(self, frame):
        # Resize frame to model input size (640x640)
        resized_frame = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert color space if needed (BGR to RGB)
        preprocessed_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values (0-1)
        normalized_frame = preprocessed_frame.astype(np.float32) / 255.0
        
        # Ensure the frame matches the expected input shape
        assert normalized_frame.shape == (self.input_height, self.input_width, self.input_channels), \
            f"Input shape mismatch. Expected {(self.input_height, self.input_width, self.input_channels)}, got {normalized_frame.shape}"
        
        return normalized_frame

    def run_inference(self, preprocessed_frame):
        # Prepare input data dictionary
        # Ensure input is a 4D tensor with batch dimension
        input_data = {
            self.input_vstream_info.name: np.expand_dims(preprocessed_frame, axis=0)
        }
        
        # Use InferVStreams for inference
        with InferVStreams(
            self.network_group, 
            self.input_vstreams_params, 
            self.output_vstreams_params
        ) as infer_pipeline:
            # Activate network group
            with self.network_group.activate(self.network_group_params):
                # Run inference
                infer_results = infer_pipeline.infer(input_data)
                
                # Return outputs for all output streams
                return {
                    out_stream.name: infer_results[out_stream.name] 
                    for out_stream in self.output_vstream_infos
                }

    def postprocess_results(self, outputs, original_frame, confidence_threshold=0.5):
        result_frame = original_frame.copy()

        for stream_name, output in outputs.items():
            output_array = np.array(output)  # Convert list to NumPy array
            
            print(f"Output stream {stream_name} shape: {output_array.shape}")
            
            # YOLOv8 NMS postprocess output format: (num_classes, 5, num_boxes)
            # output_array.shape -> (80, 5, 100) => 80 classes, 5 attributes, 100 predictions
            num_classes, _, num_boxes = output_array.shape

            for class_idx in range(num_classes):
                for box_idx in range(num_boxes):
                    box_info = output_array[class_idx, :, box_idx]

                    # Extract box info
                    x, y, w, h, confidence = box_info
                    if confidence < confidence_threshold:
                        continue  # Skip low confidence predictions

                    # Calculate box coordinates
                    x1 = int((x - w / 2) * original_frame.shape[1])  # X-min
                    y1 = int((y - h / 2) * original_frame.shape[0])  # Y-min
                    x2 = int((x + w / 2) * original_frame.shape[1])  # X-max
                    y2 = int((y + h / 2) * original_frame.shape[0])  # Y-max

                    # Draw bounding box and label
                    label = f"Class {class_idx} ({confidence:.2f})"
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        result_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
        
        return result_frame



def main():
    # Specify the exact path to your HEF file
    HEF_PATH = '/usr/share/hailo-models/yolov8s_h8l.hef'

    # Setup Picamera2
    picam2 = picamera2.Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080)},
        lores={"size": (640, 480)},
        display="lores"
    )
    picam2.configure(config)
    
    # Initialize Hailo inference
    hailo_inference = HailoYOLOInference(hef_path=HEF_PATH)
    
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