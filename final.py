import cv2
from djitellopy import Tello
from ultralytics import YOLO
import time
# Initialize Tello Drone
drone = Tello()
drone.connect()
print("Drone Battery: ", drone.get_battery())

# Initialize Models
general_model = YOLO('yolov8s.pt')
hand_model = YOLO("hand.pt")

# Start video stream
drone.streamon()
frame_skip = 3  # Process every 3rd frame
frame_count = 0

while True:
    frame = drone.get_frame_read().frame
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip processing for this frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    general_results = general_model.predict(source=frame_rgb, conf=0.5, show=False)[0]
    person_boxes = [box for box in general_results.boxes if general_model.names[int(box.cls[0])] == "person" and box.conf[0] > 0.5]

    if person_boxes:
        hand_results = hand_model.predict(source=frame_rgb, conf=0.5, show=False)[0]
        for box in hand_results.boxes:
            label = hand_model.names[int(box.cls[0])]
            confidence = box.conf[0]  # Retrieve confidence value of the detection
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract bounding box coordinates

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if label == "thumbup":
                print("Thumbs up detected")
                drone.takeoff()
                print("Takeoff command sent")
                time.sleep(5)  # Wait for drone to stabilize
                drone.hover()  # Command the drone to hover and stabilize
                time.sleep(2)  # Give some time to stabilize in hover mode
                continue  # Proceed to the next frame processing
            elif label == "thumbdown":
                print("Land command sent")
                drone.land()
                time.sleep(5)
                break  # Exit after handling gesture


    cv2.imshow("Drone View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

drone.streamoff()
cv2.destroyAllWindows()
drone.land()
print("Landing... Drone Battery: ", drone.get_battery())
