# Custom Gaussian Blur Function with NumPy

This project demonstrates my custom implementation of the Gaussian Blur operation using **NumPy**. The functionality replicates the widely used `cv2.GaussianBlur` function from the OpenCV library, providing an alternative for scenarios where OpenCV might not be available or required.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [References](#references)

---

## Overview
Gaussian Blur is a widely used image processing technique that smooths an image by reducing noise and detail. This project demonstrates how to:
- Generate a Gaussian kernel manually.
- Convolve an image with the Gaussian kernel to achieve the blurring effect.

---

## Features
- Implements Gaussian Blur using **NumPy** without relying on OpenCV.
- Allows customization of kernel size and standard deviation (sigma).
- Produces results comparable to OpenCV's `cv2.GaussianBlur` function.

---

## Requirements
Ensure you have the following installed before running the code:

- **Python 3.7+**
- **NumPy** (tested with version 1.20+)

Install NumPy using pip if it’s not already installed:
```bash
pip install numpy
```


## References
- OpenCV documentation: [cv2.GaussianBlur](https://docs.opencv.org/)
- Gaussian Blur algorithm: [Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur)
