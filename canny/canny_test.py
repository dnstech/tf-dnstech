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
"""Test for Hough lines."""

import tensorflow as tf
import numpy as np

from canny import HoughLines


class HoughTest(tf.test.TestCase):
    def test_lines(self):
        """Tests for Hough lines.

        Inputs:
            image: [[0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0]]
            thetas: [0,  45,  90,  135]
            threshold: 3

        Outputs:
            rhos: [3]
            thetas: [45]
        """
        image = tf.squeeze(tf.image.flip_left_right(tf.eye(5, dtype=tf.bool)[:, :, None]))
        thetas = tf.range(start=0, limit=180, delta=45, dtype=tf.float32) / 180.0 * np.pi
        threshold = tf.constant(2, dtype=tf.int32)

        hough_lines = HoughLines(thetas, threshold)
        rhos, thetas = hough_lines(image)

        self.assertAllEqual(rhos, tf.constant([3], dtype=tf.int32))
        self.assertAllEqual(thetas, tf.constant([0.7853982], dtype=tf.float32))


if __name__ == '__main__':
    tf.test.main()
