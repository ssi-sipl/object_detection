from picamera2 import Picamera2
import cv2

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())

# Start camera
picam2.start()

while True:
    frame = picam2.capture_array()  # Capture frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR
    
    cv2.imshow("Camera Feed", frame)  # Display
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv2.destroyAllWindows()
picam2.stop()

