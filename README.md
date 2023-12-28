# Hybrid-Images-and-Convolution
A basic image convolution function, used to create hybrid images for COMP3204 Coursework 1 - Hybrid Image

## Task Description

- Write a basic image convolution function
- Use that written function to create Hybrid Images, using a simplified version of the [SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns](https://dl.acm.org/doi/10.1145/1179352.1141919).



## Hybrid Images
Hybrid images are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available, but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances. An example of this is shown below:
<p align="center">
  <img src=https://github.com/daverajSingh/Hybrid-Images-and-Convolution/assets/92389554/6afbcf21-2274-4acc-9af6-f68fc1e52861 alt="Hybrid Image">
</p>

## Project Contents
- This project contains two python files:
  - `MyConvolution.py` which is used to perform template convolution on a given image
  - `MyHybridImages.py` which is used to create a hybrid image from two different images using the process outlined above.
### Running the Project
**This project is not runnable**. You will have to use the methods outlined in the Python scripts to run each function individually. 
