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
* **8-Points Algorithm**. Normalized 8-point algorithm 

* **RANSAC**. Self-adaptive sample by the inliers number of current best model.  Slove the fundamental matrix by 8-Points algorithm.

### **Image patch Alignment**
Use inverse compositional algorithm to align image patch in reference image to patch in current image. The model contains:

* **pixel 2D drift with bias(illumination or exposure differences)** 
$$I_c(\mathbf x + \mathbf u) = I_r(\mathbf x) + \beta$$