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
"""Connected components.

Note on optimising TensorFlow GPU performance:
 1. https://www.tensorflow.org/guide/gpu_performance_analysis
 2. https://github.com/NVIDIA/DeepLearningExamples/issues/57
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


class ConnectedComponents:
    """Connected component labelling.

    Identifies 8-way connectivity in 2D boolean images using tensorflow addons' connected components operation, which
    identifies 4-way connectivity (neighbors above, below, left, and right).
    """

    def __init__(self):
        """Initialise class attributes"""
        # kernel initialisation
        kernel = np.zeros(shape=(3, 3, 1, 2))
        kernel[1, 2, 0, 0], kernel[2, 1, 0, 0] = 1, 1
        kernel[1, 0, 0, 1], kernel[2, 1, 0, 1] = 1, 1

        self.kernel = tf.constant(kernel, tf.int32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.bool)])
    def __call__(self, images):
        """Returns the connected component ids.

        Args:
            images: 3D (N, H, W) boolean tensor of images (bool).

        Returns:
            segment_ids: Components with the same shape as images (int32). entries that evaluate to False in images have
            value 0, and all other entries map to a component id > 0.
        """
        # apply filter and get pixel positions that should be augmented
        conv = tf.where(
            tf.math.equal(
                tf.nn.convolution(tf.cast(tf.expand_dims(images, axis=-1), tf.int32), self.kernel, padding='SAME'), 2
            ),
            True, False
        )

        # augment original image
        image_cc = tf.math.logical_or(tf.math.logical_or(conv[:, :, :, 0], conv[:, :, :, 1]), images)

        # tensorflow addons' connected component ops
        segment_ids_cc = tfa.image.connected_components(image_cc)

        # correct the segment ids using original image as the mask
        segment_ids = tf.where(images, segment_ids_cc, 0)

        return segment_ids
