# Tutor simulator

This project integrates optical flow, homography, and the SORT (Simple Online and Realtime Tracking) algorithm with object detection models, such as YOLOv8, to enable efficient object tracking, speed estimation, and direction detection across video frames. The use of optical flow is key to detecting whether vehicles travel upwards or downwards, while homography is utilized for mapping real-world coordinates to image coordinates.

## Features
- **Object Detection**: Detect objects in video frames using YOLOv8.
- **Object Tracking**: Assign unique IDs to objects and track them across frames using the SORT algorithm.
- **Optical Flow Analysis**: Determine whether vehicles are traveling upwards or downwards using motion analysis.
- **Homography for Coordinate Mapping**: Map real-world coordinates to the image plane for precise calculations.

---

## Requirements

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or later
- OpenCV
- NumPy
- A cloned version of the SORT repository

### Repository Setup
You need to clone the SORT repository to use the tracker in this project:

```bash
# Clone the SORT repository
git clone https://github.com/abewley/sort.git

# Navigate to the cloned repository
cd sort
```

Once cloned, you can integrate `sort.py` into your project or use it as a module by appending its path.

---

## Installation

1. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install numpy opencv-python ultralytics
   ```

2. **Verify SORT Integration**:
   Ensure that the cloned SORT repository is accessible in your project. You can manually add its path to your Python environment:
   ```python
   import sys
   sys.path.append("/path/to/sort")
   from sort import Sort
   ```

---

## Screenshots

### Object tracking + speed estimation (and fine for transgressor :( )
![speed_estimation](screenshots/speed_estimation.png)

### Homographic view to map real world distances
![homographic_view](screenshots/homographic_view.png)

### Optical flow detection
![optical flow](screenshots/optical_flow.png)
---

## Installing CUDA and Building OpenCV with CUDA

To leverage GPU acceleration for faster optical flow calculations and object detection, follow these steps to install CUDA and build OpenCV with CUDA support:

### Step 1: Install NVIDIA Drivers

1. Check your GPU model and download the appropriate drivers from the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx).
2. Install the drivers by following the instructions provided on the NVIDIA website.
3. Verify the installation:
   ```bash
   nvidia-smi
   ```
   This command should display your GPU details.

### Step 2: Install CUDA Toolkit

1. Download the CUDA Toolkit from the [CUDA Downloads Page](https://developer.nvidia.com/cuda-downloads).
2. Install the toolkit by following the instructions for your operating system.
3. Verify CUDA installation:
   ```bash
   nvcc --version
   ```
   This command should display the installed CUDA version.

### Step 3: Install cuDNN (CUDA Deep Neural Network Library)

1. Download cuDNN from the [NVIDIA Developer Website](https://developer.nvidia.com/cudnn).
2. Copy the cuDNN files to your CUDA installation directory:
   - `lib` files to `CUDA/lib64`
   - `include` files to `CUDA/include`
3. Verify cuDNN installation by running CUDA-enabled applications.

### Step 4: Build OpenCV with CUDA Support

1. Clone the OpenCV repository:
   ```bash
   git clone https://github.com/opencv/opencv.git
   git clone https://github.com/opencv/opencv_contrib.git
   ```

2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Configure the build with CUDA support:
   ```bash
   cmake -D CMAKE_BUILD_TYPE=Release \
         -D CMAKE_INSTALL_PREFIX=/usr/local \
         -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
         -D WITH_CUDA=ON \
         -D ENABLE_FAST_MATH=1 \
         -D CUDA_FAST_MATH=1 \
         -D WITH_CUDNN=ON \
         -D OPENCV_DNN_CUDA=ON \
         ../opencv
   ```

4. Compile and install OpenCV:
   ```bash
   make -j$(nproc)
   sudo make install
   ```

5. Verify OpenCV installation with CUDA support:
   ```python
   import cv2
   print(cv2.getBuildInformation())
   ```
   Look for `CUDA` and `cuDNN` support in the output.

   To use the built version of openCV it's preferred to inject the following env variable
   ```bash
   PYTHONPATH=[opencv_folder]/build/python_loader
   ```

---

## License
This project uses the SORT repository under its respective license. Refer to the [SORT repository](https://github.com/abewley/sort) for license details.

---

## Credits
- **SORT Algorithm**: Developed by Alex Bewley ([GitHub Repository](https://github.com/abewley/sort))
- **YOLOv8**: Developed by Ultralytics ([GitHub Repository](https://github.com/ultralytics/ultralytics))

