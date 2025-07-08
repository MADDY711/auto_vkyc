import cv2
import time
import os

def capture_selfie_and_video(save_img="selfie.jpg", save_video="selfie_video.avi", duration=10):
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Cannot access the webcam")
        return

    print(" Webcam is on. Press 's' to take selfie and start video, or 'q' to quit.")
    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to read from webcam")
            break

        cv2.imshow("Live", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # === Save Selfie ===
            cv2.imwrite(save_img, frame)
            print(f" Selfie saved as '{save_img}'")

            # === Start Recording Video ===
            print(f" Recording {duration} second video...")

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_video, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

            start_time = time.time()
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    print(" Frame dropped during video recording")
                    break
                out.write(frame)
                cv2.imshow("Recording", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(" Recording interrupted.")
                    break

            out.release()
            print(f"Video saved as '{save_video}'")
            break

        elif key == ord('q'):
            print(" Cancelled. No selfie or video saved.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_img, save_video
