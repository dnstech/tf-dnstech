# Hough lines

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

import hough

# get the image and perform edge detection
img = cv2.imread('sudoku.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200, L2gradient=True)

# attributes
thetas = tf.range(start=0, limit=180, delta=1, dtype=tf.float32) / 180.0 * np.pi
threshold = tf.constant(200, dtype=tf.int32)

# get Hough lines
rhos, thetas = hough.HoughLines(thetas, threshold)(tf.convert_to_tensor(edges, dtype=tf.bool))
```

## Results

### Image

![sudoku_lines](sudoku_lines.png)

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
OpenCV: 4.7812 seconds for 1000 reps of sudoku.png
Tensorflow: 7.7971 seconds for 1000 reps of sudoku.png
```

