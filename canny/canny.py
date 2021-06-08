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
import tensorflow_probability as tfp
import numpy as np
import os

from connected_components import ConnectedComponents


os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


class CannyEdge:
    """Canny edge detector.

    <description here>
    """

    def __init__(self, weak_threshold=None, strong_threshold=None):
        """Initialise class attributes

        Args:
        """
        if (weak_threshold is not None) and (not tf.is_tensor(weak_threshold)):
            raise TypeError("weak_threshold must be a tensor")
        if (strong_threshold is not None) and (not tf.is_tensor(strong_threshold)):
            raise TypeError("strong_threshold must be a tensor")

        self.weak_threshold = tf.constant(weak_threshold, dtype=tf.float32) if weak_threshold is not None else None
        self.strong_threshold = tf.constant(strong_threshold, dtype=tf.float32) if weak_threshold is not None else None
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

        self.canny_strong = tf.constant(1.33, dtype=tf.float32)
        self.canny_weak = tf.constant(0.67, dtype=tf.float32)

        # connected components
        self.connected_components = ConnectedComponents()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
    def __call__(self, images):
        """Finds hough lines on the image

        Args:
            image: 2D tensor (float32). Should be grayscale image.

        Returns:
        """
        # get dimension of the images
        dims = tf.shape(images)

        # finding intensity gradient of the image using Sobel edge detection
        sobel_edges = tf.image.sobel_edges(tf.expand_dims(images, axis=-1))

        # calculate gradient and angle
        gradients = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(sobel_edges), axis=-1))
        angles = (tf.math.atan2(sobel_edges[:, :, :, :, 0], sobel_edges[:, :, :, :, 1]) * 180.0 / self.pi) % 180.0

        # non-maximum suppression
        angles_0 = tf.math.logical_or(tf.greater_equal(angles, 157.5), tf.less(angles, 22.5))
        angles_45 = tf.math.logical_and(tf.greater_equal(angles, 22.5), tf.less(angles, 67.5))
        angles_90 = tf.math.logical_and(tf.greater_equal(angles, 67.5), tf.less(angles, 112.5))
        angles_135 = tf.math.logical_and(tf.greater_equal(angles, 112.5), tf.less(angles, 157.5))

        target_pixel_0 = tf.nn.convolution(gradients, self.filter_0, padding='SAME')
        mask_0 = tf.greater_equal(tf.where(angles_0, gradients, 0), target_pixel_0)
        is_max_0 = tf.math.logical_and(mask_0[:, :, :, 0:1], mask_0[:, :, :, 1:2])

        target_pixel_90 = tf.nn.convolution(gradients, self.filter_90, padding='SAME')
        mask_90 = tf.greater_equal(tf.where(angles_90, gradients, 0), target_pixel_90)
        is_max_90 = tf.math.logical_and(mask_90[:, :, :, 0:1], mask_90[:, :, :, 1:2])

        target_pixel_45 = tf.nn.convolution(gradients, self.filter_45, padding='SAME')
        mask_45 = tf.greater_equal(tf.where(angles_45, gradients, 0), target_pixel_45)
        is_max_45 = tf.math.logical_and(mask_45[:, :, :, 0:1], mask_45[:, :, :, 1:2])

        target_pixel_135 = tf.nn.convolution(gradients, self.filter_135, padding='SAME')
        mask_135 = tf.greater_equal(tf.where(angles_135, gradients, 0), target_pixel_135)
        is_max_135 = tf.math.logical_and(mask_135[:, :, :, 0:1], mask_135[:, :, :, 1:2])

        raw_edges = tf.where(
            tf.reduce_any(
                tf.concat(values=[is_max_0, is_max_90, is_max_45, is_max_135], axis=-1), axis=-1,
                keepdims=True
            ),
            gradients,
            tf.reshape(tf.repeat(tf.constant(0, dtype=tf.float32), repeats=dims[0]), shape=[dims[0], 1, 1, 1])
        )

        # get the strong and weak edges
        if (self.weak_threshold is None) or (self.strong_threshold is None):
            medians = tfp.stats.percentile(
                tf.reshape(images, shape=[dims[0], dims[1] * dims[2]]), 50.0, axis=1,
                interpolation='midpoint'
            )
            strong_thres = tf.math.minimum(255.0, tf.math.scalar_mul(self.canny_strong, medians))
            weak_thres = tf.math.maximum(0.0, tf.math.scalar_mul(self.canny_weak, medians))
        else:
            strong_thres = tf.repeat(self.strong_threshold, repeats=dims[0])
            weak_thres = tf.repeat(self.weak_threshold, repeats=dims[0])

        strong_edges = tf.math.greater_equal(raw_edges, tf.reshape(strong_thres, shape=[dims[0], 1, 1, 1]))
        weak_edges = tf.math.greater_equal(raw_edges, tf.reshape(weak_thres, shape=[dims[0], 1, 1, 1]))

        # connected components of weak edges
        segment_ids = tf.expand_dims(
            self.connected_components(tf.reshape(weak_edges, shape=[dims[0], dims[1], dims[2]])),
            axis=-1
        )

        # fix segment ids
        segment_ids_reshaped = tf.reshape(segment_ids, shape=[dims[0], dims[1] * dims[2]])
        segment_ids_max = tf.reduce_max(segment_ids_reshaped, axis=1)
        starting_ids = tf.concat([[0], segment_ids_max[:-1]], axis=0)

        segment_ids = segment_ids - tf.reshape(starting_ids, shape=[dims[0], 1, 1, 1])
        segment_ids = tf.where(tf.math.less(segment_ids, 0), 0, segment_ids)

        # get the final edges from the connected segments and strong edges
        final_edges = tf.map_fn(
            lambda xx: _final_edge(xx[0], xx[1], dims[1:]),
            (segment_ids, strong_edges),
            fn_output_signature=tf.bool
        )

        return final_edges


def _final_edge(segment_ids, strong_edges, dim):
    strong_edges = tf.cast(strong_edges, tf.int32)
    num_segments = tf.reduce_max(segment_ids) + 1

    # number of pixels for each segment id
    segment_sums_strong = tf.math.unsorted_segment_sum(strong_edges, segment_ids, num_segments)

    # valid segments in the weak edges. in other words, pixels/edges worth paying attention to
    mask_strong = tf.math.greater(segment_sums_strong, 0)

    # correct for sorted y value as unsorted_segment_sum sorts output
    y, idx = tf.unique(tf.reshape(segment_ids, shape=[-1]))
    idx_corrected = tf.gather(tf.argsort(tf.argsort(y)), idx)

    # get the boolean mask
    edge = tf.reshape(tf.gather(mask_strong, idx_corrected), shape=dim)

    return edge
