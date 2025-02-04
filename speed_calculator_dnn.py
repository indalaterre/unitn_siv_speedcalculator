import cv2
import time
import torch
import numpy as np

from sort.sort import Sort
from ultralytics import YOLO  # For YOLOv8

from utils.utils import calculate_speed
from utils.video import (process_frame,
                         map_from_homography,
                         is_using_gpu,
                         draw_tutor_area,
                         draw_optical_flow,
                         calculate_farneback_optical_flow,
                         print_car_speed)

def select_points(event, click_x, click_y, ___, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([click_x, click_y])

        frame = param.copy()
        for p in points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)
        cv2.imshow("Select Homographic Points", frame)

points = []

force_cpu = False

red_line_y = 470
blue_line_y = 500

# Load the model
model = YOLO('yolov8n.pt')
if force_cpu is True:
    print('Forcing YOLO model to use the CPU')
    model.to('cpu')

tracker = Sort()


cap = cv2.VideoCapture('video/56310-479197605.mp4')
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

scale = (.5, .5)
height *= scale[1]

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(1)

first_frame, prev_gray = process_frame(cap, scale=scale)

cv2.imshow("Select Homographic Points", first_frame)
cv2.setMouseCallback("Select Homographic Points", select_points, first_frame)

if is_using_gpu() and force_cpu is False:
    print('Running Farneback with GPU')
else:
    print('Running Farneback with CPU')


all_points_selected = False
while not all_points_selected:
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        if len(points) >= 4:
            all_points_selected = True
            cv2.setMouseCallback("Select Homographic Points", lambda *args : None)

cv2.destroyAllWindows()

height, width = first_frame.shape[:2]
homography_width = int(width / 2)

output_points = np.array([
    [0, 0],          # Top-left
    [homography_width - 1, 0],  # Top-right
    [homography_width - 1, height - 1],  # Bottom-right
    [0, height - 1]  # Bottom-left
], dtype=np.float32)

H, status = cv2.findHomography(np.array(points), output_points)

_, red_line_y_r = map_from_homography((0, red_line_y), H)
_, blue_line_y_r = map_from_homography((0, blue_line_y), H)

# Based on the difference between line blue/red in the homographic image we can assign a real world distance
max_allowed_speed = 90
max_distance_in_m = 750
speed_tutor_length_in_m = width * (abs(red_line_y - blue_line_y)) / max_distance_in_m

offset = 6
prev_positions = dict()  # Store car IDs and their previous positions

while cap.isOpened():
    frame_start_time = time.time()
    last_frame, last_frame_gray = process_frame(cap, scale)

    # Run detection
    if last_frame_gray is None:
        break

    #Calculate optical flow to detect direction
    flow = calculate_farneback_optical_flow(prev_frame=prev_gray,
                                            frame_gray=last_frame_gray,
                                            pyr_scale=0.5,
                                            levels=3,
                                            win_size=15,
                                            iterations=5,
                                            poly_n=5,
                                            poly_sigma=1.2,
                                            flags=0,
                                            force_cpu=force_cpu)


    arrow_frame = last_frame.copy()
    draw_optical_flow(arrow_frame, flow)

    homographic_frame = cv2.warpPerspective(last_frame, H, (homography_width, height))
    results = model(last_frame)

    detenctions = []
    current_time = time.time()
    for result in results:
        for box, confidence, class_tensor in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if int(class_tensor) == 2 or int(class_tensor) == 7: # <-- Cars or Trucks
                x1, y1, x2, y2 = box
                detenctions.append([int(x1), int(y1), int(x2), int(y2), float(confidence)])

    tracked_objects = tracker.update(np.array(detenctions))
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj

        i_box = (int(x1), int(y1), int(x2), int(y2))
        i_x1, i_y1, i_x2, i_y2 = i_box

        car_id = int(obj_id)

        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        f_center_x, f_center_y = float(center_x), float(center_y)

        if car_id not in prev_positions:
            roi_flow = flow[i_y1:i_y2, i_x1:i_x2]
            mean_flow_y = np.mean(roi_flow[..., 1])

            """
            We check for positive flow because higher pixel positions are in the bottom of the image
            
                -1 for downward traveling,
                 1 for upward traveling
            """
            direction = -1 if mean_flow_y > 0 else 1

            prev_positions[car_id] = {
                'flow': mean_flow_y,
                'direction': direction
            }

        direction = prev_positions[car_id]['direction']

        if f_center_y - offset < red_line_y_r < f_center_y + offset:
            # Crossed the red line
            if direction == -1 and 'start_time' not in prev_positions[car_id]:
                # Going downwards! Start tracking
                prev_positions[car_id]['start_time'] = time.time()
            elif 'start_time' in prev_positions[car_id] and direction == 1 and 'speed' not in prev_positions[car_id]:
                # Going upwards and exiting tutor area. Time to check for fines! $$
                calculate_speed(prev_positions[car_id], speed_tutor_length_in_m)
        elif f_center_y - offset < blue_line_y_r < f_center_y + offset:
            # Crossed the blue line
            if direction == 1 and 'start_time' not in prev_positions[car_id]:
                # Going upwards! Start tracking
                prev_positions[car_id]['start_time'] = time.time()
            elif 'start_time' in prev_positions[car_id] and direction == -1 and 'speed' not in prev_positions[car_id]:
                # Going downwards and exiting tutor area. Time to check for fines! $$
                calculate_speed(prev_positions[car_id], speed_tutor_length_in_m)

        if 'speed' in prev_positions[car_id]:
            speed = prev_positions[car_id]['speed']
            print_car_speed(last_frame, speed, i_box, max_allowed_speed)

    frame_rate = 1 / (time.time() - frame_start_time)
    cv2.putText(last_frame, f"FPS: {frame_rate:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    draw_tutor_area(last_frame, width, red_line_y_r, blue_line_y_r)
    cv2.imshow('Speed Estimation', last_frame)

    draw_tutor_area(homographic_frame, homography_width, red_line_y, blue_line_y)
    cv2.imshow('Homographic Plane', homographic_frame)

    cv2.imshow('OpticalFlow', arrow_frame)

    prev_gray = last_frame_gray

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
