import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
else:
    print("Camera opened successfully!")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
            
        # Display the frame
        cv2.imshow('Camera Test', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed.") 