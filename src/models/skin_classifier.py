import os
from .imbalanced.balanced_binary_classifier import BalancedBinaryClassifier

class SkinClassifier(BalancedBinaryClassifier):
    def __init__(self,
                 **kwargs
        ):
        super(SkinClassifier, self).__init__(**kwargs)