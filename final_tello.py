from djitellopy import Tello
import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load YOLOv8 Pose model
model = YOLO("C:\\Users\\tyreke\\Desktop\\yolo\\yolov8s-pose.pt")

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

def get_keypoints(frame, model):
    results = model(frame)
    keypoints = None
    confidences = None

    if results:
        for res in results:
            if hasattr(res, 'keypoints') and res.keypoints:
                if res.keypoints.xy.numel() > 0:  # 确保关键点坐标张量不为空
                    keypoints = res.keypoints.xy[0]
                    # 检查关键点信心分数张量是否不为空
                    if hasattr(res.keypoints, 'conf') and res.keypoints.conf.numel() > 0:
                        confidences = res.keypoints.conf[0]
                break  # 使用第一个检测到的带有关键点的姿态

    return keypoints, confidences

# Check if the required keypoints are visible with sufficient confidence
def check_visible_with_confidence(keypoints, confidences, indices, threshold=0.1):
    return all(confidences[idx] > threshold for idx in indices if idx < len(keypoints))

# Check if the keypoints indicate a full body is visible
def check_full_body(keypoints, confidences):
    required_keypoints = [0, 5, 6, 15, 16]  # Head, left shoulder, right shoulder, left ankle, right ankle
    return check_visible_with_confidence(keypoints, confidences, required_keypoints)

def check_half_body(keypoints, confidences):
    # 单独检查头部和脚踝
    head_keypoint = [0]  # 头部索引
    ankle_keypoints = [15, 16]  # 脚踝索引

    # 检查头部或任一脚踝的置信度是否足够
    is_head_visible = check_visible_with_confidence(keypoints, confidences, head_keypoint)
    are_ankles_visible = any(check_visible_with_confidence(keypoints, confidences, [ankle]) for ankle in ankle_keypoints)

    # 确保不是全身被检测到
    is_not_full_body = not check_full_body(keypoints, confidences)

    # 只要头部或任一脚踝被检测到，且不是全身被检测到，返回 True
    return (is_head_visible or are_ankles_visible) and is_not_full_body

# Adjust the drone's lateral position to center the shoulders in the frame
def adjust_lateral_position(drone, keypoints, confidences):
    # 计算肩膀中心的 x 坐标
    shoulder_center_x = (keypoints[5][0] + keypoints[6][0]) / 2
    while abs(shoulder_center_x - 320) > 106:
        if shoulder_center_x < 320:
            drone.move_left(100)  # 向右移动，速度20（根据需要调整）
        else:
            drone.move_right(100)  # 向左移动，速度20（根据需要调整）
        time.sleep(0.5)  # 调整睡眠时间
        frame = update_frame(drone)
        keypoints, confidences = get_keypoints(frame, model)
        if keypoints is not None:
            shoulder_center_x = (keypoints[5][0] + keypoints[6][0]) / 2
        else:
            break  # 如果没有关键点检测结果，退出循环



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
            time.sleep(0.5)
            frame = update_frame(drone)
            keypoints, confidences = get_keypoints(frame, model)
            head_y, ankle_y = keypoints[0][1], min(keypoints[15][1], keypoints[16][1])
            head_ankle_height = abs(head_y - ankle_y)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            # Optionally, handle or retry the command
            if "Auto land" in str(e):
                print("Auto-landing triggered. Stopping command execution.")
                break  # Stop the loop if auto-landing was triggered
        finally:
            attempt_count += 1



def main():
    # 初始化无人机并起飞
    drone = initialize_drone()
    drone.takeoff()

    try:
        while True:
            # 更新帧并获取关键点及其置信度
            frame = update_frame(drone)
            keypoints, confidences = get_keypoints(frame, model)

            # 检查是否检测到关键点
            if keypoints is not None and confidences is not None:
                if check_full_body(keypoints, confidences):
                    adjust_lateral_position(drone, keypoints, confidences)
                    adjust_vertical_position(drone, keypoints)
                    print("Target height achieved, drone is hovering.")
                    
                elif check_half_body(keypoints, confidences):
                    drone.move_back(60)
                    time.sleep(1)  # 稍作等待再次更新帧和关键点
                    frame = update_frame(drone)
                    keypoints, confidences = get_keypoints(frame, model)
                    if keypoints is not None and check_full_body(keypoints, confidences):
                        adjust_lateral_position(drone, keypoints, confidences)
                        adjust_vertical_position(drone, keypoints)
                        print("Target height achieved, drone is hovering.")
                        


            # 显示当前帧以供实时查看
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'退出
                break

    finally:
        # 确保无人机降落并关闭视频流和窗口
        drone.land()
        drone.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
