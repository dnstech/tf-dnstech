# MIT License
#
# Copyright (c) 2021 DNS Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Hough transform for line detection.

Note on optimising TensorFlow GPU performance:
 1. https://www.tensorflow.org/guide/gpu_performance_analysis
 2. https://github.com/NVIDIA/DeepLearningExamples/issues/57
"""

import tensorflow as tf
import os

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


class HoughLines:
    """Hough transform for line detection.

    Transforms a boolean 2D image to the Hough space, where each entry corresponds to a
    line parameterized by the angle `theta` clockwise from horizontal (in radians),
    and the distance `rho` (in pixels; the distance from coordinate `(0, 0)` in the
    image (matrix coordinate system, hence, top left corner) to the closest point in the line).
    """

    def __init__(self, thetas, threshold):
        """Initialise class attributes

        Args:
            thetas: 1D tensor of possible clockwise angles from top horizontal line (float32).
            threshold: Minimum vote count for a Hough bin to be considered a line (int32).
        """
        if not tf.is_tensor(thetas):
            raise TypeError("thetas must be a tensor")
        if not tf.is_tensor(threshold):
            raise TypeError("threshold must be a tensor")

        self.thetas = tf.cast(thetas, dtype=tf.float32)
        self.threshold = tf.cast(threshold, dtype=tf.int32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.bool)])
    def __call__(self, image):
        """Finds hough lines on the image.

        Args:
            image: 2D boolean tensor (bool). Should be sparse (mostly false), normally the output from an edge detection
            algorithm, e.g., Sobel, Canny, etc.

        Returns:
            hough_rhos: 1D tensor of rho values rounded to the nearest integer (int32).
            hough_thetas: 1D tensor of theta values in radians (float32).
        """
        # calculate rho values
        with tf.name_scope("rhos"):
            rhos = self._rhos(image)

        # perform 2-dimensional bincount
        with tf.name_scope("bincount_2d"):
            rhos, bins = self._bincount_2d(rhos)

        # get peaks based on threshold value
        with tf.name_scope("find_peaks"):
            peaks = tf.where(tf.greater_equal(bins, self.threshold))

        # get rhos and thetas corresponding to the peaks
        with tf.name_scope("gather_rhos"):
            hough_rhos = tf.gather(
                tf.range(tf.reduce_min(rhos), tf.reduce_max(rhos) + 1, dtype=self.threshold.dtype),
                peaks[:, 1]
            )

        with tf.name_scope("gather_thetas"):
            hough_thetas = tf.gather(
                self.thetas,
                peaks[:, 0]
            )

        return hough_rhos, hough_thetas

    def _bincount_2d(self, rhos):
        """2D bincount using 1D bincount function.

        Args:
            rhos: The rho values (float32). 2D tensor for all positives in image across all possible thetas.

        Returns:
            bins: The 2D bincount (int32). 2D tensor for intersections in Hough space.
        """
        # 2-dimensional bincount
        num_rows = tf.cast(tf.shape(rhos)[0], dtype=self.threshold.dtype)

        # round and typecast rho values
        rhos = tf.cast(tf.math.round(rhos), dtype=self.threshold.dtype)

        # convert the values in each row to a consecutive range of ids that will not
        # overlap with the other rows.
        row_values = rhos - tf.reduce_min(rhos) + \
                     (tf.expand_dims(tf.range(num_rows, dtype=self.threshold.dtype), axis=1)) * \
                     (tf.reduce_max(rhos) - tf.reduce_min(rhos) + 1)

        # flatten the tensor
        values_flat = tf.reshape(row_values, [-1])

        # bincount
        bins_length = tf.multiply(num_rows, tf.reduce_max(rhos) - tf.reduce_min(rhos) + 1)
        bins = tf.reshape(
            tf.math.bincount(values_flat, minlength=bins_length, maxlength=bins_length, dtype=self.threshold.dtype),
            [num_rows, -1]
        )

        return rhos, bins

    def _rhos(self, image):
        """Gets rho values based on rho = x cos theta + y sin theta.

        Args:
            image: 2D boolean tensor (bool). Should be sparse (mostly false), normally the output from an edge detection
            algorithm, e.g., Sobel, Canny, etc.

        Returns:
            rhos: The rho values (float32). 2D tensor for all positives in image across all possible thetas.
        """
        # coordinates of pixels where the edges is detected
        coordinates = tf.cast(tf.where(image), dtype=self.thetas.dtype)

        # get the rho values for the theta values
        # rho = x cos theta + y sin theta
        # x and y => unit coordinates for distance
        rhos = tf.expand_dims(coordinates[:, 1], axis=0) * tf.expand_dims(tf.cos(self.thetas), axis=1) + \
               tf.expand_dims(coordinates[:, 0], axis=0) * tf.expand_dims(tf.sin(self.thetas), axis=1)

        return rhos
