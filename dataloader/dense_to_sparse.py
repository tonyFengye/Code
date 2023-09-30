# @Time     : 2022/4/16 15:04
# @Author   : Chen nengzhen
# @FileName : dense_to_sparse.py
# @Software : PyCharm
import numpy as np
import cv2


class DenseToSparse:
    def __init__(self):
        pass

    def dense_to_sparse(self, rgb, depth):
        pass

    def __repr__(self):
        pass


class UniformSampling(DenseToSparse):
    name = "uar"

    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d, md=%f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, depth):
        """
        Sample pixels with "num_samples" / pixels probability in 'depth'.
        Only pixels with a maximum depth of 'max_depth' are considered.
        If no 'max_depth' is given, samples in all pixels.
        :param depth:
        :return:
        """
        mask_keep = depth > 0
        if self.max_depth is not np.inf:
            mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
        n_keep = np.count_nonzero(mask_keep)
        print("n_keep: ", n_keep)
        if n_keep == 0:
            return mask_keep
        else:
            prob = float(self.num_samples) / n_keep
            return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
