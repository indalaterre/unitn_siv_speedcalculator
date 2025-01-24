import time

import cv2
import math
import numpy as np

from utils.video import process_frame, map_to_homography, is_using_gpu, calculate_farneback_optical_flow, print_car_speed

points = []

def select_points(event, click_x, click_y, ___, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([click_x, click_y])

        frame = param.copy()
        for p in points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)
        cv2.imshow("Select Homographic Points", frame)

cap = cv2.VideoCapture('video/56310-479197605.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

scale = (.5, .5)
if scale is not None:
    height *= scale[1]

total_road_distance_in_m = 1000
m_per_pixel = total_road_distance_in_m / height

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(1)

first_frame, prev_gray = process_frame(cap, scale)

cv2.imshow("Select Homographic Points", first_frame)
cv2.setMouseCallback("Select Homographic Points", select_points, first_frame)

if is_using_gpu():
    print('Running Farneback with GPU')
else:
    print('Running Farneback with CPU')


all_points_selected = False
while not all_points_selected:
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        if len(points) >= 4:
            all_points_selected = True
        else:
            print("At least 4 points are required for homography. Continue selecting.")

cv2.destroyAllWindows()

height, width = first_frame.shape[:2]
width = int(width / 2)

output_points = np.array([
    [0, 0],          # Top-left
    [width - 1, 0],  # Top-right
    [width - 1, height - 1],  # Bottom-right
    [0, height - 1]  # Bottom-left
], dtype=np.float32)

H, status = cv2.findHomography(np.array(points), output_points)

centroids_index = dict()

prev_timestamp = time.time()
while cap.isOpened():
    last_frame, last_frame_gray = process_frame(cap, scale)
    if last_frame_gray is None:
        break

    homographic_frame = cv2.warpPerspective(last_frame, H, (width, height))


    # Compute optical flow using the Lucas-Kanade method
    flow = calculate_farneback_optical_flow(prev_frame=prev_gray,
                                            frame_gray=last_frame_gray,
                                            pyr_scale=0.5,
                                            levels=3,
                                            win_size=15,
                                            iterations=5,
                                            poly_n=5,
                                            poly_sigma=1.2,
                                            flags=0)

    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    median_mag = np.median(mag)

    motion_mask = (mag > max(2.5, median_mag * 1.5)).astype(np.uint8)

    # 6. (Optional) Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 7. Find contours of the moving regions
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    arrow_frame = last_frame.copy()

    h, w = flow.shape[:2]
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            fx, fy = flow[y, x]
            magnitude = np.sqrt(fx**2 + fy**2)

            # Only draw if magnitude exceeds threshold
            if magnitude > 2:
                x_end = int(x + fx)
                y_end = int(y + fy)
                cv2.arrowedLine(
                    arrow_frame,
                    (x, y),
                    (x_end, y_end),
                    (0,255, 0),
                    thickness=1,
                    line_type=cv2.LINE_AA,
                    tipLength=0.3
                )

    current_timestamp = time.time()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 700:
            bounding_rect = cv2.boundingRect(cnt)
            (x, y, w, h) = bounding_rect

            cx, cy = math.ceil(x + w / 2), math.ceil(y + h / 2)
            dx, dy = flow[cy, cx, 0], np.mean(flow[cy, cx, 1])
            prev_x, prex_y = math.ceil(abs(cx - dx)), math.ceil(abs(cy - dy))

            prex_x_h, prex_y_h = map_to_homography((prev_x, prex_y), H)
            x_h, y_h = map_to_homography((cx, cy), H)

            magnitude = math.ceil(np.sqrt((x_h - prex_x_h)**2 + (y_h - prex_y_h)**2))

            # We could apply further checks for roundness here
            # e.g. ratio of bounding box dimensions, or Hough circle detection
            # But let’s just draw a bounding rect for demonstration

            speed = int(magnitude * 3.6 / (current_timestamp - prev_timestamp))
            print_car_speed(last_frame, speed, bounding_rect)

    prev_gray = last_frame_gray

    prev_timestamp = current_timestamp

    cv2.imshow('OpticalFlow', arrow_frame)
    cv2.imshow('Homographic Plane', homographic_frame)
    cv2.imshow('Cars Speed', np.hstack([last_frame]))

# Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
