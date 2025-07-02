import cv2
import numpy as np

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load models
def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('F:\\clg\\internships\\clg_internship\\auto_vkyc\\models\\deploy_age.prototxt', 'F:\\clg\\internships\\clg_internship\\auto_vkyc\\models\\age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('F:\\clg\\internships\\clg_internship\\auto_vkyc\\models\\deploy_gender.prototxt', 'F:\\clg\\internships\\clg_internship\\auto_vkyc\\models\\gender_net.caffemodel')
    return age_net, gender_net

# Initialize face detector and webcam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

def detect_and_display(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Get face ROI
            face_img = frame[y:y + h, x:x + w].copy()
            if face_img.size == 0:
                continue  # Skip if ROI is empty

            # Create blob from face
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y - 10), font, 0.8, (0, 255, 0), 2)

        cv2.imshow("Age and Gender Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    age_net, gender_net = load_caffe_models()
    detect_and_display(age_net, gender_net)

if __name__ == "__main__":
    main()
