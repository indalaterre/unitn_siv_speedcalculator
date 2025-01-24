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



def print_car_speed(frame, speed, rect_coords):
    (x, y, w, h) = rect_coords

    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    text_color = (0, 0, 255)  # Red color in BGR

    speed_text = f'{speed} km/h'
    (text_width, text_height), baseline = cv2.getTextSize(speed_text, font, font_scale, font_thickness)
    text_x = x + (w - text_width) // 2  # Center horizontally
    text_y = y + text_height // 2  # Center vertically (adjust for baseline)

    # Drawing box rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Drawing speed text
    cv2.rectangle(frame, (x, y - 10), (x + w, y + 10), (0, 255, 0), -1)
    cv2.putText(frame, speed_text, (text_x, text_y - 5), font, font_scale, text_color, font_thickness)


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
        flags
):

    if is_using_gpu():
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
