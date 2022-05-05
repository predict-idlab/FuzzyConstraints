import numpy as np


class GenericBatchLoader:

    def __init__(self, batch_size=128, neg_ratio=1.0):
        self.batch_size = batch_size
        self.neg_ratio = int(neg_ratio)

        # empty arrays to contain individual batches (both 0 and 1 values)
        self.new_triples_indexes = np.empty((self.batch_size * (self.neg_ratio + 1), 3)).astype(np.int32)
        self.new_triples_values = np.empty((self.batch_size * (self.neg_ratio + 1), 1)).astype(np.float32)

        self.batches = list()

    def __call__(self):
        return next(self.batches)
