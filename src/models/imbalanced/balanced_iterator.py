import os
import threading
import numpy as np
from keras.preprocessing.image import Iterator
from .balanced_generator import BalancedGenerator
class BalancedIterator(Iterator):
    def __init__(self, n, batch_size, shuffle, seed, y = None, balanced_type = ""):
        self.y = y
        self.n = n
        self.balanced_type = balanced_type
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()
        self.balancedGenerator = BalancedGenerator()
    def _set_index_array(self):
        #custom function
        if self.y is not None:
            self.index_array = self.balancedGenerator._balance_index(self.y, self.batch_size, self.balanced_type)
        else:
            self.index_array = np.arange(self.n)
            if self.shuffle:
                self.index_array = np.random.permutation(self.n)
        #end costom function
        '''
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)
        '''
    def _get_index_array(self):
        return self.index_array
