import os
import cv2
import numpy as np
import time
from djitellopy import Tello
from ultralytics import YOLO

# Setup directories and parameters
screenshot_dir = "screenshots"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# PI Controller Parameters
rifX, rifY = 960 / 2, 720 / 2  # Frame center
Kp_X, Ki_X = 0.1, 0.0  # Proportional-Integral gains for X
Kp_Y, Ki_Y = 0.2, 0.0  # Proportional-Integral gains for Y
Tc = 0.05  # Control frequency
movement_threshold = 20  # Threshold for movement detection
integral_X, integral_Y = 0, 0

# Load Models
general_model = YOLO("yolov8s.pt")  # Object detection model
hand_model = YOLO("hand.pt")  # Hand gesture detection model

# Initialize Drone
drone = Tello()
drone.connect()
print("Drone Battery:", drone.get_battery())
drone.streamon()
is_flying = False
frame_skip = 3
frame_count = 0

# Position tracking
prev_person_center_x, prev_person_center_y = None, None

while True:
    frame = drone.get_frame_read().frame
    frame_count += 1

    if frame_count % frame_skip != 0:
        continue  # Skip frames to reduce processing load

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.circle(frame, (int(rifX), int(rifY)), 10, (0, 0, 255), 2)

    # Gesture recognition
    hand_results = hand_model.predict(source=frame_rgb, conf=0.5, show=False)[0]
    for box in hand_results.boxes:
        label = hand_model.names[int(box.cls[0])]
        confidence = box.conf[0]
        if confidence > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            if label == "thumbup" and not is_flying:
                screenshot_path = os.path.join(screenshot_dir, f"screenshot_thumbup_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(screenshot_path, frame)
                drone.takeoff()
                is_flying = True
                print(f"Takeoff command sent, screenshot saved at {screenshot_path}.")
                time.sleep(5)
            elif label == "thumbdown" and is_flying:
                screenshot_path = os.path.join(screenshot_dir, f"screenshot_thumbdown_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(screenshot_path, frame)
                drone.land()
                is_flying = False
                print(f"Land command sent, screenshot saved at {screenshot_path}.")
                time.sleep(5)

    # Object Detection and Following
    if is_flying:
        general_results = general_model.predict(source=frame_rgb, conf=0.5, show=False)[0]
        person_detected = False
        for box in general_results.boxes:
            label = general_model.names[int(box.cls[0])]
            if label == "person" and box.conf[0] > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                person_center_x = (x1 + x2) / 2
                person_center_y = (y1 + y2) / 2
                person_detected = True

                # Movement detection
                if prev_person_center_x is not None and prev_person_center_y is not None:
                    dx = abs(person_center_x - prev_person_center_x)
                    dy = abs(person_center_y - prev_person_center_y)
                    if dx < movement_threshold and dy < movement_threshold:
                        continue

                # PI Control to follow the person
                error_X = rifX - person_center_x
                error_Y = rifY - person_center_y
                integral_X += error_X * Tc
                integral_Y += error_Y * Tc
                control_X = Kp_X * error_X + Ki_X * integral_X
                control_Y = Kp_Y * error_Y + Ki_Y * integral_Y
                drone.send_rc_control(0, int(control_X), int(control_Y), 0)

                prev_person_center_x, prev_person_center_y = person_center_x, person_center_y

    cv2.imshow("Drone View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

drone.streamoff()
cv2.destroyAllWindows()
drone.land()
print("Landing... Drone Battery:", drone.get_battery())
