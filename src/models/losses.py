from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.python.util.tf_export import tf_export
import keras.losses
'''
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
'''
LOSS functions
'''
def binary_balance_focal(y_true, y_pred):
    return binary_balance_focal_base(y_true, y_pred, alpha = .25, gamma = 2.0, a=1.0, b=2.0, c=2.0)
def skin_focal_loss(y_true, y_pred):
    return binary_balance_focal_base(y_true, y_pred, alpha = .25, gamma = 2.0, a=1.0, b=0, c=2.0)
def binary_balance_focal_base(y_true, y_pred, alpha = .25, gamma = 2.0, a=1.0, b=0.5, c=2.0):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    fpe = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    fne = -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    l = ((fne + fpe) * a + K.abs(fne - fpe) * b) / c
    return l
def binary_custom_loss(y_true, y_pred):
    return binary_balance_mse_base(y_true, y_pred, a=1.0, b=0.5, c=2.0)
def binary_balance_mse_adv(y_true, y_pred):
    return binary_balance_mse_base(y_true, y_pred, a=1.0, b=2.0, c=2.0)
def binary_balance_mse(y_true, y_pred):
    return binary_balance_mse_base(y_true, y_pred, a=1.0, b=0.0, c=2.0)
def mask_size(ts):
    ret = K.int_shape(ts)
    if ret is None:
        return 0
    ret = ret[0]
    if ret is None:
        return 0
    return ret

def binary_balance_mse_base(y_true, y_pred, a=1.0, b=2.0, c=2.0):
    mask1 = tf.equal(y_true, 1)
    mask0 = tf.equal(y_true, 0)
    
    #len1 = mask_size(mask1)
    #len0 = mask_size(mask0)
    #print("binary_balance_mse_base->len1=",len1,",len0=", len0, flush=True)
    
    y_true_1 = tf.boolean_mask(y_true,mask1)
    y_pred_1 = tf.boolean_mask(y_pred,mask1)
    fne = mean_squared_error(y_true_1,y_pred_1)
        
    
    y_true_0 = tf.boolean_mask(y_true,mask0)
    y_pred_0 = tf.boolean_mask(y_pred,mask0)
    fpe = mean_squared_error(y_true_0,y_pred_0)
    
    l = ((fne + fpe) * a + K.abs(fne - fpe) * b) / c
    return l
def binary_balance_pmse(y_true, y_pred):
    return binary_balance_pmse_base(y_true, y_pred, a=1.0, b=2.0, c=2.0)
def binary_balance_pmse_base(y_true, y_pred, a=1.0, b=2.0, c=2.0):
    mask1 = tf.equal(y_true, 1)
    mask0 = tf.equal(y_true, 0)
  
    y_true_1 = tf.boolean_mask(y_true,mask1)
    y_pred_1 = tf.boolean_mask(y_pred,mask1)
    y_true_0 = tf.boolean_mask(y_true,mask0)
    y_pred_0 = tf.boolean_mask(y_pred,mask0)
    
    fpe = mean_squared_error(y_true_0,y_pred_0)
    fne = mean_squared_error(y_true_1,y_pred_1)
    l = ((fne + fpe) * (fne + fpe) * a + (fne - fpe) * (fne - fpe) * b) / c
    return l
def binary_balance_bce_adv(y_true, y_pred):
    return binary_balance_bce_base(y_true, y_pred, a=1.0, b=2.0, c=2.0)
def binary_balance_bce(y_true, y_pred):
    return binary_balance_bce_base(y_true, y_pred, a=1.0, b=0.0, c=2.0)
def binary_balance_bce_base(y_true, y_pred, a=1.0, b=2.0, c=2.0):
    mask1 = tf.equal(y_true, 1)
    mask0 = tf.equal(y_true, 0)
  
    y_true_1 = tf.boolean_mask(y_true,mask1)
    y_pred_1 = tf.boolean_mask(y_pred,mask1)
    y_true_0 = tf.boolean_mask(y_true,mask0)
    y_pred_0 = tf.boolean_mask(y_pred,mask0)
    
    fpe = binary_crossentropy(y_true_0,y_pred_0)
    fne = binary_crossentropy(y_true_1,y_pred_1)
    l = ((fne + fpe) * a + K.abs(fne - fpe) * b) / c
    return l
def binary_balance_err(y_true, y_pred):
    mask1 = tf.equal(y_true, 1)
    mask0 = tf.equal(y_true, 0)
  
    y_true_1 = tf.boolean_mask(y_true,mask1)
    y_pred_1 = tf.boolean_mask(y_pred,mask1)
    y_true_0 = tf.boolean_mask(y_true,mask0)
    y_pred_0 = tf.boolean_mask(y_pred,mask0)
    
    fpe = mean_squared_error(y_true_0,y_pred_0)
    fne = mean_squared_error(y_true_1,y_pred_1)
    l = (fne + fpe) * (fne + fpe) + (fne - fpe) * (fne - fpe)
    return l    
def binary_balance_loss(y_true, y_pred):
    return binary_balance_mse(y_true, y_pred)
def binary_focal_loss(y_true, y_pred):
    alpha = .25
    gamma = 2.0
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    fpe = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    fne = -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return fne + fpe

keras.losses.binary_balance_err = binary_balance_err
keras.losses.binary_focal_loss = binary_focal_loss
keras.losses.binary_balance_loss = binary_balance_loss
keras.losses.binary_harmonic_loss = binary_custom_loss
keras.losses.binary_custom_loss = binary_custom_loss
keras.losses.skin_focal_loss = skin_focal_loss
keras.losses.binary_balance_mse = binary_balance_mse
keras.losses.binary_balance_mse_adv = binary_balance_mse_adv
keras.losses.binary_balance_bce = binary_balance_bce
keras.losses.binary_balance_bce_adv = binary_balance_bce_adv
keras.losses.binary_balance_focal = binary_balance_focal

keras.losses.binary_balance_pmse = binary_balance_pmse

keras.metrics.sensitivity = sensitivity
keras.metrics.specificity = specificity
keras.metrics.bacc = bacc
keras.metrics.binary_focal_loss = binary_focal_loss