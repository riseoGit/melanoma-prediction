import argparse
import os
import numpy as np

from butil import dataset as ds
from butil.dataset import _ham_get_data
from models.skin_classifier import SkinClassifier
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def main(FLAGS):
    print("****** START TRAINING SKIN CLASSIFIER ******\n\n")
    
    data_dir = FLAGS.data_dir
    type = FLAGS.type
    epochs_fast = FLAGS.epochs_fast
    epochs_slow = FLAGS.epochs_slow
    batch_size = FLAGS.batch_size
    balance = FLAGS.balance
    retrain = FLAGS.retrain
    lossfunc = FLAGS.lossfunc
    optimizer = FLAGS.optimizer
    output_dir = FLAGS.output_dir
    val_batch_size = FLAGS.val_batch_size
    bn_type = FLAGS.bn_type
    #bn_top = FLAGS.bn_top
    lr = FLAGS.lr
    lr_min = FLAGS.lr_min
    lr_fast = FLAGS.lr_fast
    lr_fast_min = FLAGS.lr_fast_min
    
    top = FLAGS.top
    custom_layer_name = FLAGS.custom_layer_name
    bn_one = True
    bn_two = True
    bn_top = False
    if FLAGS.bn_top == "bn":
        bn_top = True
    if bn_type == "only_first" :
        bn_two = False
    elif bn_type == "no" :
        bn_one = False
        bn_two = False
    top_layers = [{"num":1024,"batchnormalization":bn_one,"activation":"relu","dropout":0.5},
                 {"num":512,"batchnormalization":bn_two,"activation":"relu","dropout":0.5}]
    
    if top == 1:
        top_layers = [{"num":1024,"batchnormalization":bn_two,"activation":"relu","dropout":0.5}]
    cls = SkinClassifier(
            type = type,
            retrain = retrain,
            epochs_fast = epochs_fast,
            epochs_slow = epochs_slow,
            batch_size = batch_size,
            val_batch_size = val_batch_size,
            balance = balance,
            lossfunc = lossfunc,
            optimizer = optimizer,
            output_dir = output_dir,
            lr = lr,
            lr_min = lr_min,
            lr_fast = lr_fast,
            lr_fast_min = lr_fast_min,
            top_layers = top_layers,
            bn_top = bn_top,
            custom_layer_name = custom_layer_name
            )
    cls.init_model()
    
    x_train = _ham_get_data(data_dir, "x_train")
    y_train = _ham_get_data(data_dir, "y_train")
    
    x_val = _ham_get_data(data_dir, "x_val")
    y_val = _ham_get_data(data_dir, "y_val")
    
    cls.train(x_train, y_train, x_val, y_val)
    cls.save(output_dir)
    x_test = _ham_get_data(data_dir, "x_test")
    y_test = _ham_get_data(data_dir, "y_test")
    cls.evaluate (x_val, y_val, x_test, y_test)
    
    print("****** END TRAINING DEEP CNN ******\n\n")
    
if __name__ == "__main__":
    
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/binary',
        help='input data directory'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='InceptionV3',
        help='CNN Model: InceptionV3,DenseNet201'
    )
    parser.add_argument(
        '--epochs_fast',
        type=int,
        default=20,
        help='epochs_fast'
    )
    parser.add_argument(
        '--epochs_slow',
        type=int,
        default=150,
        help='epochs_slow'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Please input batch_size'
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=0,
        help='Please input validation batch_size'
    )
    parser.add_argument(
        '--balance',
        type=int,
        default=0,
        help='Please input balance: 0->normal, 1:stratified, 2:balanced'
    )
    parser.add_argument(
        '--retrain',
        type=bool,
        default=False,
        help='Please input retrain'
    )
    parser.add_argument(
        '--lossfunc',
        type=str,
        default='',
        help='loss function name'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        help='optimizer: adam, sgd'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='result/models/v2',
        help='output directory of model'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate'
    )
    parser.add_argument(
        '--lr_min',
        type=float,
        default=0.0000001,
        help='lr_min'
    )
    parser.add_argument(
        '--lr_fast',
        type=float,
        default=0.0001,
        help='lr_fast'
    )
    parser.add_argument(
        '--lr_fast_min',
        type=float,
        default=0.00001,
        help='lr_fast_min'
    )
    parser.add_argument(
        '--bn_type',
        type=str,
        default='',
        help='Batch Normalization Type'
    )
    parser.add_argument(
        '--bn_top',
        type=str,
        default='',
        help='Batch Normalization Top'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=2,
        help='top hidden layer'
    )
    parser.add_argument(
        '--custom_layer_name',
        type=str,
        default='',
        help='Custom layer name'
    )
    FLAGS = parser.parse_args()
    print ("data_dir=",FLAGS.data_dir)
    print ("type=",FLAGS.type)
    print ("epochs_fast=",FLAGS.epochs_fast)
    print ("epochs_slow=",FLAGS.epochs_slow)
    print ("batch_size=",FLAGS.batch_size)
    print ("val_batch_size=",FLAGS.val_batch_size)
    print ("balance=",FLAGS.balance)
    print ("retrain=",FLAGS.retrain)
    print ("lossfunc=",FLAGS.lossfunc)
    print ("optimizer=",FLAGS.optimizer)
    print ("output_dir=",FLAGS.output_dir)
    print ("lr=",FLAGS.lr)
    print ("lr_min=",FLAGS.lr_min)
    print ("lr_fast=",FLAGS.lr_fast)
    print ("lr_fast_min=",FLAGS.lr_fast_min)
    print ("bn_type=",FLAGS.bn_type)
    print ("bn_top=",FLAGS.bn_top)
    print ("top=",FLAGS.top)
    print ("custom_layer_name=",FLAGS.custom_layer_name)
    main(FLAGS)
    