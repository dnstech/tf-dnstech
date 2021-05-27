# Copyright 2021 DNS Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Hough lines."""

import tensorflow as tf
import numpy as np

from hough import HoughLines


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
