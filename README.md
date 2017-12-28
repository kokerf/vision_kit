# vision_kit

A computer vision kit for algorithm verification. ***Far from stable***.

## Required
* [OpenCV 3.1.0](https://github.com/opencv/opencv/tree/3.1.0)
* [Eigen3](https://github.com/RLovelett/eigen/tree/3.3.3)

## Modules

### 1. **Base**
Some universal functions and some definitions.

### 2. **Optical Flow**
Algorithms for Optical flow

* **Pyramidal Lucas-Kanada Algorithm**

### 3. **Epipolar Geometry**
Function about Fundamental Matrix. There are two methods to find the fundamental matrix
* **8-Points Algorithm**. Normalized 8-point algorithm 

* **RANSAC**. Self-adaptive sample by the inliers number of current best model.  Solve the fundamental matrix by 8-Points algorithm.

### 4. **Image patch Alignment**
Use **Inverse Compositional** and **Efficient Second-order Minimization** algorithm to align image patch in reference image to patch in current image. The model contains:

* **pixel 2D drift** (IV)
$$I_c(\mathbf x + \mathbf u) = I_r(\mathbf x)$$

* **pixel 2D drift with bias(illumination or exposure differences)** (IV, ESM)
$$I_c(\mathbf x + \mathbf u) = I_r(\mathbf x) + \beta$$

## Usage
First of all, build the code.
```
mkdir build && cd build
cmake ..
make -j
```
Then run the demos
```
# Base
./test_base ../data/desk1.png ../data/desk2.png

# Optical Flow
./test_opticalflow ../data/floor1.png ../data/floor2.png

# Epipolar Geometry
./test_fundamental ../data/desk1.png ../data/desk2.png

# Image Alignment
./test_align2D ../data/floor1.png ../data/floor2.png
./test_align1D ../data/floor1.png ../data/floor2.png
```