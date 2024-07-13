import cv2
from ultralytics import YOLO

# Load the YOLOv8 pose model
pose_model = YOLO("hand.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Perform pose estimation on the frame
        results = pose_model(frame, verbose=False, conf=0.7)

        # Display the annotated frame
        annotated_frame = results[0].plot()
        cv2.imshow("Pose Estimation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the webcam and destroy the windows
cap.release()
cv2.destroyAllWindows()