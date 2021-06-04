from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.python.util.tf_export import tf_export
import keras.losses
'''
CUSTOM METRICS
'''
def bacc(y_true, y_pred):
    sen = sensitivity(y_true, y_pred)
    spe = specificity(y_true, y_pred)
    return (sen + spe) / 2.0
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

keras.metrics.sensitivity = sensitivity
keras.metrics.specificity = specificity
keras.metrics.sen = sensitivity
keras.metrics.spe = specificity
keras.metrics.bacc = bacc
'''
CUSTOM LOSS functions
'''
#core loss function
def _binary_core_focal(y_true, y_pred, alpha = .25, gamma = 2.0):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    fpe = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    fne = -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return [fpe, fne]
def _binary_core_mse(y_true, y_pred):
    mask1 = tf.equal(y_true, 1)
    mask0 = tf.equal(y_true, 0)
  
    y_true_1 = tf.boolean_mask(y_true,mask1)
    y_pred_1 = tf.boolean_mask(y_pred,mask1)
    y_true_0 = tf.boolean_mask(y_true,mask0)
    y_pred_0 = tf.boolean_mask(y_pred,mask0)
    
    fpe = mean_squared_error(y_true_0,y_pred_0)
    fne = mean_squared_error(y_true_1,y_pred_1)
    
    return [fpe, fne]
def _binary_core_bce(y_true, y_pred):
    mask1 = tf.equal(y_true, 1)
    mask0 = tf.equal(y_true, 0)
  
    y_true_1 = tf.boolean_mask(y_true,mask1)
    y_pred_1 = tf.boolean_mask(y_pred,mask1)
    y_true_0 = tf.boolean_mask(y_true,mask0)
    y_pred_0 = tf.boolean_mask(y_pred,mask0)
    
    fpe = binary_crossentropy(y_true_0,y_pred_0)
    fne = binary_crossentropy(y_true_1,y_pred_1)
    
    return [fpe, fne]
def _binary_core_loss(fpe, fne, a=1.0, b=2.0, c=2.0, p = 1):
    x = fpe + fne
    y = K.abs(fpe - fne)
    if p > 1:
        x = K.pow((fpe + fne), p)
        y = K.pow((fpe - fne), p)
    else:
        x = fpe + fne
        y = K.abs(fpe - fne)
    return (x * a + y * b) / c    
def _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=0.0, c=2.0, p = 1, type = ""):
    
    if type == "focal":
        fpe, fne = _binary_core_focal(y_true, y_pred)
    elif type == "bce":
        fpe, fne = _binary_core_bce(y_true, y_pred)
    else:
        fpe, fne = _binary_core_mse(y_true, y_pred)
    return _binary_core_loss(fpe, fne, a, b, c, p)

'''
balanced =  x + y
balanced power = ((x + y) * (x + y) + b * (x - y) * (x - y)) / 2.0 with b = 2
balanced 4 power = ((x + y) * (x + y) + 4 * (x - y) * (x - y)) / 2.0 with b = 2
'''    
def binary_balanced_focal(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=0.0, c=1.0, p=1, type = "focal")
def binary_balanced_pfocal(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=2.0, c=1.0, p=2, type = "focal")
def binary_balanced_1pfocal(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=1.0, c=1.0, p=2, type = "focal")
def binary_balanced_4pfocal(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=4.0, c=1.0, p=2, type = "focal")
def binary_balanced_mse(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=0.0, c=1.0, p=1, type = "mse")
def binary_balanced_pmse(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=2.0, c=1.0, p=2, type = "mse")
def binary_balanced_1pmse(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=1.0, c=2.0, p=2, type = "mse")
def binary_balanced_4pmse(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=4.0, c=1.0, p=2, type = "mse")

def binary_balanced_bce(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=0.0, c=1.0, p=1, type = "bce")
def binary_balanced_pbce(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=2.0, c=1.0, p=2, type = "bce")
def binary_balanced_1pbce(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=1.0, c=1.0, p=2, type = "bce")
def binary_balanced_4pbce(y_true, y_pred):
    return _binary_core_loss_by_type(y_true, y_pred, a=1.0, b=4.0, c=1.0, p=2, type = "bce")
  
keras.losses.binary_balanced_focal = binary_balanced_focal
keras.losses.binary_balanced_pfocal = binary_balanced_pfocal
keras.losses.binary_balanced_1pfocal = binary_balanced_1pfocal
keras.losses.binary_balanced_4pfocal = binary_balanced_4pfocal
keras.losses.binary_balanced_mse = binary_balanced_mse
keras.losses.binary_balanced_pmse = binary_balanced_pmse
keras.losses.binary_balanced_1pmse = binary_balanced_1pmse
keras.losses.binary_balanced_4pmse = binary_balanced_4pmse
keras.losses.binary_balanced_bce = binary_balanced_bce
keras.losses.binary_balanced_pbce = binary_balanced_pbce
keras.losses.binary_balanced_1pbce = binary_balanced_1pbce
keras.losses.binary_balanced_4pbce = binary_balanced_4pbce