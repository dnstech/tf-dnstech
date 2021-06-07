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
"""Test for connected components."""

import tensorflow as tf
import numpy as np

from connected_components import ConnectedComponents


class ConnectedComponentsTest(tf.test.TestCase):
    def test_connected_components(self):
        """Test for connected components.

        Inputs:
            image: [[True, False, True, False, True],
                    [False, True, True, True, False],
                    [False, False, True, False, False],
                    [False, True, True, True, False],
                    [True, False, True, False, True]]

        Outputs:
            segment_ids: [[1, 0, 1, 0, 1],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [1, 0, 1, 0, 1]]
        """
        img = np.eye(5)
        img = np.fliplr(img) + img
        img[:, 2] = 1.0

        image = tf.convert_to_tensor(img[None, :], dtype=tf.bool)

        connected_components = ConnectedComponents()
        segment_ids = connected_components(image)

        self.assertAllEqual(segment_ids, tf.convert_to_tensor(img[None, :], dtype=tf.int32))


if __name__ == '__main__':
    tf.test.main()
