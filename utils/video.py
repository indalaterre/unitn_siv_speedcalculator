import cv2
import numpy as np


def process_frame(capture, scale=None):
    ret, frame = capture.read()
    if ret:
        if scale is not None:
            frame = cv2.resize(frame, None, fx=scale[0], fy=scale[1], interpolation=cv2.INTER_AREA)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 3)
    else:
        frame_gray = None

    return frame, frame_gray

def map_to_homography(position, h_matrix):
    p_array = [position[0], position[1], 1]

    p_world = np.dot(h_matrix, p_array)
    p_world /= p_world[2]
    return p_world[0], p_world[1]


def map_from_homography(position, h_matrix):
    """
     We use the inverted matrix to map homographic point to real world points
    :param position: the tutor line position
    :param h_matrix: the homography matrix
    :return:
    """
    inv_h_matrix = np.linalg.inv(h_matrix)
    p_array = [position[0], position[1], 1]

    p_world = np.dot(inv_h_matrix, p_array)
    p_world /= p_world[2]
    return int(p_world[0]), int(p_world[1])


def draw_optical_flow(frame, flow):
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
                    frame,
                    (x, y),
                    (x_end, y_end),
                    (0,255, 0),
                    thickness=1,
                    line_type=cv2.LINE_AA,
                    tipLength=0.3
                )


def print_car_speed(frame, speed, rect_coords, max_allowed_speed):
    (x, y, x1, y1) = rect_coords

    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    text_color = (0, 0, 255)  # Red color in BGR

    speed_text = f'{speed:.2f} km/h'

    rect_color = (0, 255, 0) if speed <= max_allowed_speed else (0, 0, 255)

    # Drawing box rectangle
    cv2.rectangle(frame, (x, y), (x1, y1), rect_color, 2)
    cv2.putText(frame, speed_text, (x, y - 5), font, font_scale, text_color, font_thickness)


def draw_tutor_area(frame, width, upper_bound, lower_bound, alpha = .5):
    overlay = frame.copy()

    cv2.line(frame, (0, upper_bound), (width, upper_bound), (0, 0, 255), thickness=2)
    cv2.line(frame, (0, lower_bound), (width, lower_bound), (255, 0, ), thickness=2)

    cv2.rectangle(overlay, (0, upper_bound), (width, lower_bound), color=(255, 0, 255), thickness=-1)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)



def is_using_gpu():
    return cv2.cuda.getCudaEnabledDeviceCount() > 0

def calculate_farneback_optical_flow(
        prev_frame,
        frame_gray,
        pyr_scale,
        levels,
        win_size,
        iterations,
        poly_n,
        poly_sigma,
        flags,
        force_cpu=False
):

    if is_using_gpu() and force_cpu is False:
        return farneback_optical_flow_with_gpu(
            prev_frame,
            frame_gray,
            pyr_scale,
            levels,
            win_size,
            iterations,
            poly_n,
            poly_sigma,
            flags
        )
    else:
        return cv2.calcOpticalFlowFarneback(
            prev_frame,
            frame_gray,
            None,
            pyr_scale,
            levels,
            win_size,
            iterations,
            poly_n,
            poly_sigma,
            flags
        )


def farneback_optical_flow_with_gpu(
        prev_frame,
        frame_gray,
        pyr_scale,
        levels,
        win_size,
        iterations,
        poly_n,
        poly_sigma,
        flags
):
    gpu_prev_gray = cv2.cuda_GpuMat()
    gpu_last_gray = cv2.cuda_GpuMat()
    gpu_prev_gray.upload(prev_frame)
    gpu_last_gray.upload(frame_gray)

    cuda_farneback = cv2.cuda_FarnebackOpticalFlow.create(
        numLevels=levels,
        pyrScale=pyr_scale,
        fastPyramids=False,
        winSize=win_size,
        numIters=iterations,
        polyN=poly_n,
        polySigma=poly_sigma,
        flags=flags
    )

    gpu_flow = cuda_farneback.calc(gpu_prev_gray, gpu_last_gray, None)
    return gpu_flow.download()
