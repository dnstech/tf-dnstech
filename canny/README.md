# Canny edge

Hough transform is a feature extraction technique used in image processing. The technique can be used to detect shapes in images, e.g., lines and circles.

## References

This module was built using the following resources as reference materials:

1. [Moonlight Optical Music Recognition](https://github.com/tensorflow/moonlight/blob/master/moonlight/vision/hough.py)
2. [Wikipedia](https://en.wikipedia.org/wiki/Hough_transform)

## Usage

```python
import cv2
import numpy as np
import tensorflow as tf

import canny

# get the image
img = cv2.imread('sudoku.png' , cv2.IMREAD_GRAYSCALE)
img_tf = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), axis=0)

# Mode 1: Manual threshold
# ---------------------------
# attributes
weak_threshold = tf.constant(100, dtype=tf.float32)
strong_threshold = tf.constant(200, dtype=tf.float32)

# get Canny edge
canny_manual = canny.CannyEdge(
    weak_threshold=weak_threshold, 
    strong_threshold=strong_threshold
)
edges_tf_manual = canny_manual(img_tf)


# Mode 2: Automatic threshold
# ---------------------------
# get Canny edge
canny_auto = canny.CannyEdge()
edges_tf_auto = canny_auto(img_tf)
```

## Results

### Image

![sudoku_edges](sudoku_edges.png)

### Time

#### Environment
```
CPU: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
GPU: GeForce RTX 2080 Ti
TensorFlow: 2.4.1
Python: 3.8.10
```
The python packages and their versions used during the development of this module can be found in `requirements.txt`.

#### Results

```
OpenCV: 0.0304 seconds for 100 reps of sudoku.png
Tensorflow (Manual Thresholding): 0.9486 seconds for a batch of 100 sudoku.png
Tensorflow (Auto Thresholding): 0.9317 seconds for a batch of 100 sudoku.png
```

