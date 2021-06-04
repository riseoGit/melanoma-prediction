from keras.preprocessing.image import ImageDataGenerator
from .balanced_numpy_array_iterator import BalancedNumpyArrayIterator

class BalancedImageDataGenerator(ImageDataGenerator):
    def __init__(self,
                 balanced_type = "",
                 **kwargs):
        self.balanced_type = balanced_type
        super(BalancedImageDataGenerator, self).__init__(**kwargs)
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):
        return BalancedNumpyArrayIterator(
            x,
            y,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            balanced_type = self.balanced_type
        )
