# vision_kit

A computor vision kit based on [OpenCV 3.1.0](https://github.com/opencv/opencv/tree/3.1.0). ***Far from stable***.



## Modules

### **Base**
  Some universal functions and some definitions.

### **Optical Flow**
  Algorithms for Optical flow

* **Pyramidal  Lucas-Kanada Algorithm**

### **Epipolar Geometry**
  Function about Fundamental Matrix. There are two methods to find the fundamental matrix
* **8-Points Algorithm**

  Normalized 8-point algorithm 

* **RANSAC**

  Self-adaptive sample by the inliers number of current best model.  Slove the fundamental matrix by 8-Points algorithm.