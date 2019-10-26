# 3D Reconstruction (Microsoft Kinetic v1)
Use camera intrinsics, extrinsics and camera pose to reconstruct the 3D scene from Microsoft Kinetic sensor.

## Dependencies
Eigen
FreeImage

## Get Started
1. Clone the repository and navigate to this folder
```
git clone https://github.com/YangLiu14/deep-learning-examples.git
cd deep-learning-examples/computer_vision/3D_Scanning_and_Motion_Capture/3D_reconstruction/
```

2. Download data
The data is provided by TUM (Technical University of Munich) Computer Vision Group. [Download link](https://vision.in.tum.de/data/datasets/rgbd-dataset/download). Here we only need the `fr1/xyz`. Download and unzip it into the `data` folder.

3. Dependencies
[Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html). It's suggested to download and extract the source code and put it in the `/libs/` folder
FreeImage

4. Build
```
mkdir build && cd build
cmake ../src/
make 
```


## Background Knowledge

### Pinhole model
https://www.ics.uci.edu/~majumder/vispercep/cameracalib.pdf

### Intrinsics and Extrinsic
- Intrinsics:

<img src=\"img/reconstruct_snapshot.png\" style=\"width: 400px;\"/>


