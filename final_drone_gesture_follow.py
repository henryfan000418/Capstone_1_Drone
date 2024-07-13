import os
import cv2
from djitellopy import Tello
from ultralytics import YOLO
import numpy as np
import time

# Setup directories and parameters
screenshot_dir = "screenshots"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# Load YOLOv8 Pose and hand models
pose_model = YOLO("yolov8s-pose.pt")
hand_model = YOLO("hand.pt")

# Initialize and return a Tello drone object
def initialize_drone():
    drone = Tello()
    drone.connect()
    print(f"BATTERY: {drone.get_battery()}")
    drone.streamon()
    return drone

# Update and return the latest video frame from the drone
def update_frame(drone):
    frame = drone.get_frame_read().frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (640, 480))

# Get keypoints from the frame using the YOLO model
def get_keypoints(frame, model):
    results = model(frame)
    keypoints = None
    confidences = None

    if results:
        for res in results:
            if hasattr(res, 'keypoints') and res.keypoints:
                if res.keypoints.xy.numel() > 0:
                    keypoints = res.keypoints.xy[0]
                    if hasattr(res.keypoints, 'conf') and res.keypoints.conf.numel() > 0:
                        confidences = res.keypoints.conf[0]
                break

    return keypoints, confidences

# Check if the required keypoints are visible with sufficient confidence
def check_visible_with_confidence(keypoints, confidences, indices, threshold=0.1):
    return all(confidences[idx] > threshold for idx in indices if idx < len(keypoints))

# Check if the keypoints indicate a full body is visible
def check_full_body(keypoints, confidences):
    required_keypoints = [0, 5, 6, 15, 16]  # Head, left shoulder, right shoulder, left ankle, right ankle
    return check_visible_with_confidence(keypoints, confidences, required_keypoints)

def check_half_body(keypoints, confidences):
    head_keypoint = [0]  # Head index
    ankle_keypoints = [15, 16]  # Ankle indices

    is_head_visible = check_visible_with_confidence(keypoints, confidences, head_keypoint)
    are_ankles_visible = any(check_visible_with_confidence(keypoints, confidences, [ankle]) for ankle in ankle_keypoints)
    is_not_full_body = not check_full_body(keypoints, confidences)

    return (is_head_visible or are_ankles_visible) and is_not_full_body

# Adjust the drone's lateral position to center the shoulders in the frame
def adjust_lateral_position(drone, keypoints, confidences):
    shoulder_center_x = (keypoints[5][0] + keypoints[6][0]) / 2
    while abs(shoulder_center_x - 320) > 106:
        if shoulder_center_x < 320:
            drone.move_left(100)
        else:
            drone.move_right(100)
        time.sleep(0.2)
        frame = update_frame(drone)
        keypoints, confidences = get_keypoints(frame, pose_model)
        if keypoints is not None:
            shoulder_center_x = (keypoints[5][0] + keypoints[6][0]) / 2
        else:
            break

# Adjust the drone's vertical position to center the person in the frame
def adjust_vertical_position(drone, keypoints, max_attempts=10):
    head_y = keypoints[0][1]
    ankle_y = min(keypoints[15][1], keypoints[16][1])
    head_ankle_height = abs(head_y - ankle_y)
    attempt_count = 0

    while not (240 <= head_ankle_height <= 320) and attempt_count < max_attempts:
        try:
            if head_ankle_height < 240:
                drone.move_forward(100)
            elif head_ankle_height > 320:
                drone.move_back(100)
            time.sleep(0.2)
            frame = update_frame(drone)
            keypoints, confidences = get_keypoints(frame, pose_model)
            head_y, ankle_y = keypoints[0][1], min(keypoints[15][1], keypoints[16][1])
            head_ankle_height = abs(head_y - ankle_y)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            if "Auto land" in str(e):
                print("Auto-landing triggered. Stopping command execution.")
                break
        finally:
            attempt_count += 1

# Main function
def main():
    drone = initialize_drone()
    is_flying = False
    exit_flag = False

    try:
        while not exit_flag:
            frame = update_frame(drone)
            keypoints, confidences = get_keypoints(frame, pose_model)

            if not is_flying:
                # Gesture recognition
                hand_results = hand_model.predict(source=frame, conf=0.7, show=False)[0]
                for box in hand_results.boxes:
                    label = hand_model.names[int(box.cls[0])]
                    confidence = box.conf[0]
                    if confidence > 0.7:
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

            if is_flying and keypoints is not None and confidences is not None:
                if check_full_body(keypoints, confidences):
                    adjust_lateral_position(drone, keypoints, confidences)
                    adjust_vertical_position(drone, keypoints)
                    print("Target height achieved, drone is hovering.")
                elif check_half_body(keypoints, confidences):
                    drone.move_back(30)
                    time.sleep(1)
                    frame = update_frame(drone)
                    keypoints, confidences = get_keypoints(frame, pose_model)
                    if keypoints is not None and check_full_body(keypoints, confidences):
                        adjust_lateral_position(drone, keypoints, confidences)
                        adjust_vertical_position(drone, keypoints)
                        print("Target height achieved, drone is hovering.")

            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
                break

    finally:
        drone.land()
        drone.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
