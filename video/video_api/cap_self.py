import cv2

def capture_selfie():
    # Start the default webcam
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(" Cannot access the webcam")
        return

    print(" Webcam is on. Press 's' to take a selfie, or 'q' to quit.")

    # Optional: Create a resizable window
    cv2.namedWindow("Live Selfie", cv2.WINDOW_NORMAL)

    while True:
        # Read one frame
        ret, frame = cap.read()
        if not ret:
            print(" Failed to read from webcam")
            break

        # Show the video frame
        cv2.imshow("Live Selfie", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save the frame as an image
            cv2.imwrite("selfie.jpg", frame)
            print(" Selfie saved as 'selfie.jpg'")
            break
        elif key == ord('q'):
            print(" Cancelled. No selfie taken.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Call the function
capture_selfie()
