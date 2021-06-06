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
"""Canny edge detection.

Note on optimising TensorFlow GPU performance:
 1. https://www.tensorflow.org/guide/gpu_performance_analysis
 2. https://github.com/NVIDIA/DeepLearningExamples/issues/57
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


class CannyEdge:
    """Canny edge detector.

    <description here>
    """

    def __init__(self, thetas, threshold):
        """Initialise class attributes

        Args:
        """
        self.pi = tf.constant(np.pi, dtype=tf.float32)

        # filters for canny edge detection
        filter_0, filter_90 = np.zeros(shape=(3, 3, 1, 2)), np.zeros(shape=(3, 3, 1, 2))
        filter_45, filter_135 = np.zeros(shape=(3, 3, 1, 2)), np.zeros(shape=(3, 3, 1, 2))

        filter_0[1, 0, 0, 0], filter_0[1, 2, 0, 1] = 1, 1
        self.filter_0 = tf.constant(filter_0, tf.float32)

        filter_90[0, 1, 0, 0], filter_90[2, 1, 0, 1] = 1, 1
        self.filter_90 = tf.constant(filter_90, tf.float32)

        filter_45[0, 0, 0, 0], filter_45[2, 2, 0, 1] = 1, 1
        self.filter_45 = tf.constant(filter_45, tf.float32)

        filter_135[0, 2, 0, 0], filter_135[2, 0, 0, 1] = 1, 1
        self.filter_135 = tf.constant(filter_135, tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def __call__(self, image):
        """Finds hough lines on the image

        Args:
            image: 2D tensor (float32). Should be grayscale image.

        Returns:
        """
        # Step 1: Noise reduction. Apply 5x5 gaussian filter with sigma 1.0
        gaus_image = tfa.image.gaussian_filter2d(image, filter_shape=[5, 5], sigma=1.0)

        # Step 2: Finding intensity gradient of the image using Sobel edge detection
        # Sobel edge detection
        sobel_edges = tf.image.sobel_edges(gaus_image)

        # Calculate gradient and angle
        gradients = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(sobel_edges), axis=4))
        angles = (tf.math.atan2(sobel_edges[:, :, :, :, 0], sobel_edges[:, :, :, :, 1]) * 180 / self.pi) % 180

        # Step 3: Non-maximum Suppression
        # Rounded angle masks
        angles_0 = tf.cast(tf.math.logical_or(tf.greater_equal(angles, 157.5), tf.less(angles, 22.5)), tf.float32)
        angles_45 = tf.cast(tf.math.logical_and(tf.greater_equal(angles, 22.5), tf.less(angles, 67.5)), tf.float32)
        angles_90 = tf.cast(tf.math.logical_and(tf.greater_equal(angles, 67.5), tf.less(angles, 112.5)), tf.float32)
        angles_135 = tf.cast(tf.math.logical_and(tf.greater_equal(angles, 112.5), tf.less(angles, 157.5)), tf.float32)

        # Perform non-maximum suppression and get the raw edges
        target_pixel_0 = tf.nn.convolution(gradients, self.filter_0, padding='SAME')
        mask_0 = tf.cast(tf.greater(tf.math.multiply(gradients, angles_0), target_pixel_0), tf.float32)
        is_max_0 = tf.math.multiply(mask_0[:, :, :, 0:1], mask_0[:, :, :, 1:2])

        target_pixel_90 = tf.nn.convolution(gradients, self.filter_90, padding='SAME')
        mask_90 = tf.cast(tf.greater(tf.math.multiply(gradients, angles_90), target_pixel_90), tf.float32)
        is_max_90 = tf.math.multiply(mask_90[:, :, :, 0:1], mask_90[:, :, :, 1:2])

        target_pixel_45 = tf.nn.convolution(gradients, self.filter_45, padding='SAME')
        mask_45 = tf.cast(tf.greater(tf.math.multiply(gradients, angles_45), target_pixel_45), tf.float32)
        is_max_45 = tf.math.multiply(mask_45[:, :, :, 0:1], mask_45[:, :, :, 1:2])

        target_pixel_135 = tf.nn.convolution(gradients, self.filter_135, padding='SAME')
        mask_135 = tf.cast(tf.greater(tf.math.multiply(gradients, angles_135), target_pixel_135), tf.float32)
        is_max_135 = tf.math.multiply(mask_135[:, :, :, 0:1], mask_135[:, :, :, 1:2])

        raw_edges = tf.math.multiply(gradients, tf.math.add_n([is_max_0, is_max_90, is_max_45, is_max_135]))
