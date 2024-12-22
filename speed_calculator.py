
import cv2
import cvzone
# Install CUDA-enabled OpenCV
import cv2.cuda

from ultralytics import YOLO

import numpy as np

video_path = 'video/56310-479197605.mp4'

mask_resized = False
road_mask_gray = cv2.cvtColor(cv2.imread('video/road_mask.png'), cv2.COLOR_BGR2GRAY)

detection_mask = cv2.imread('video/detection_mask.png')
stop_detection_mask = cv2.cvtColor(cv2.imread('video/stop_detection_mask.png'), cv2.COLOR_BGR2GRAY)

detection_model = YOLO('detenction/yolov8l.pt')

def preprocess_frame(video_stream, scale_factor=0.33):
    is_over, frame = video_stream.read()
    if not is_over:
        return None, None

    '''
    frame = cv2.resize(src=frame,
                       dsize=None,
                       fx=scale_factor,
                       fy=scale_factor,
                       interpolation=cv2.INTER_LINEAR)
    '''

    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.bitwise_and(frame_gray, road_mask_gray)
    return frame, frame_gray


def detect_cars(frame):
    detection_frame = cv2.bitwise_and(frame, detection_mask)
    return detection_model(detection_frame, stream=True)

cars_dict = dict()

isOpened = True
while isOpened:

    cap = cv2.VideoCapture(video_path)

    if not mask_resized:
        mask_resized = True
        # We need the first frame
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        detection_mask = cv2.resize(detection_mask,
                                    dsize=(width, height),
                                    interpolation = cv2.INTER_LINEAR)
        stop_detection_mask = cv2.resize(stop_detection_mask,
                                         dsize=(width, height),
                                         interpolation = cv2.INTER_LINEAR)
        road_mask_gray = cv2.resize(road_mask_gray,
                                    dsize=(width, height),
                                    interpolation = cv2.INTER_LINEAR)


    _, prev_frame_gray = preprocess_frame(cap)
    if prev_frame_gray is None:
        print("Error reading video file!")
        exit()

    while cap.isOpened():
        last_frame, last_frame_gray = preprocess_frame(cap)
        if last_frame is None:
            break

        # We'll now filter only the cars getting the centroid of the objects
        detected_cars = detect_cars(last_frame)
        for det in detected_cars:
            boxes = det.boxes

            boxes = [box for box in boxes if int(box.cls) == 2]
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                # Calculating the center of the box
                center = (int(x1 + w / 2), int(y1 + h / 2))

                ## Extracting box features for tracking

                cars_dict.update({
                    len(cars_dict): {
                        'center': center,
                        'box': (center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5),
                        'shape': (10, 10)
                    }
                })

                cv2.circle(last_frame,
                           center,
                           radius=3,
                           color=(255, 0, 0))
                cv2.rectangle(last_frame,
                              pt1=(center[0] - 5, center[1] - 5),
                              pt2=(center[0] + 5, center[1] + 5),
                              color=(255,0,0))
                #cvzone.cornerRect(last_frame, (x1, y1, w, h))

        # Calculate optical flow for the keypoints
        optical_flow = cv2.calcOpticalFlowFarneback(prev_frame_gray,
                                                    last_frame_gray,
                                                    None,
                                                    0.5,
                                                    3,
                                                    15,
                                                    3,
                                                    5,
                                                    1.2,
                                                    0)

        for car_id, car in cars_dict.items():
            ## Checking if the box is in the death area

            car_x, car_y = car['box'][0], car['box'][1]
            car_x1, car_y1 = car['box'][2], car['box'][3]

            car_w, car_h = car['shape'][0], car['shape'][1]

            death_area = stop_detection_mask[car_x:car_y, car_x1:car_y1]
            non_zero_count = cv2.countNonZero(death_area)
            if non_zero_count > 0:
                cars_dict.pop(car_id)
            else:
                car_flow = optical_flow[car_x:car_y, car_x1:car_y1]

                dx = car_flow[..., 0]  # Horizontal flow
                dy = car_flow[..., 1]  # Vertical flow

                new_x = int(car_x + np.mean(dx))
                new_y = int(car_y + np.mean(dy))

                # Update ROI position based on average flow
                car['box'] = (new_x, new_y, new_x + car_w, new_y + car_h)

                cv2.circle(last_frame, (car['box'][0], car['box'][1]), 3, (255, 0, 0))
                cv2.rectangle(last_frame,
                              pt1=(car['box'][0], car['box'][1]),
                              pt2=(car['box'][2], car['box'][3]),
                              color=(255,0,0))
                #cvzone.cornerRect(last_frame, (car['box'][0], car['box'][1], car_w, car_h))

        # Display the frame with rectangles
        cv2.imshow("Car Detection", last_frame)

        # Update previous frame
        prev_frame_gray = last_frame_gray

        # Break on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            isOpened = False
            break

    cap.release()

cv2.destroyAllWindows()