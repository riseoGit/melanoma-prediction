import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adamax
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.layers import BatchNormalization
from keras.layers import Activation

import tensorflow as tf

from .cyclical_learning_rate import CyclicLR
from .balanced_losses import *
from .balanced_image_data_generator import BalancedImageDataGenerator
from .balanced_generator import _balanced_batch_apply

class BalancedBinaryClassifier(object):
    def __init__(self,
                 type="InceptionV3",
                 retrain=False,
                 epochs_fast = 0,
                 epochs_slow = 10,
                 batch_size = 64,
                 val_batch_size = 0,
                 balance = 0,
                 lossfunc = "",
                 optimizer = "adam",
                 output_dir = "cnn_model",
                 lr = 0.00001,
                 lr_min = 0.00000001,
                 lr_fast = 0.00001,
                 lr_fast_min = 0.000001,
                 top_layers = [{"num":1024,"batchnormalization":True,"activation":"relu","dropout":0.5},
                 {"num":512,"batchnormalization":True,"activation":"relu","dropout":0.5}],
                 bn_top = "",
                 custom_layer_name = "",
                 input_shape = (192, 256, 3)
        ):
        self.type = type
        self.retrain = retrain
        self.epochs_fast = epochs_fast
        self.epochs_slow = epochs_slow
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.output_dir = output_dir
        self.cnn_model = None
        self.lr = lr
        self.lr_min = lr_min
        self.lr_fast = lr_fast
        self.lr_fast_min = lr_fast_min
        self.top_layers = top_layers
        self.bn_top = bn_top
        self.custom_layer_name = custom_layer_name
        self.last_layer_name = ""
        self.balance = balance
        self.balanced_type = ""
        if self.balance == 1:
            self.balanced_type = ""
        elif self.balance == 2:
            self.balanced_type = "balanced"
    def _get_pre_trained_model(self):
        #pre_trained_model = None
        self.last_layer_name = ""
        if self.type == "InceptionV3":
            self.last_layer_name = "mixed10"
            pre_trained_model = InceptionV3(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "VGG16":
            self.last_layer_name = "block5_pool"
            pre_trained_model = VGG16(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "InceptionResNetV2":
            self.last_layer_name = "conv_7b_ac"
            pre_trained_model = InceptionResNetV2(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "ResNet50":
            self.last_layer_name = "activation_49"
            pre_trained_model = ResNet50(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "EfficientNetB0":
            from efficientnet.keras import EfficientNetB0
            self.last_layer_name = "top_conv" 
            pre_trained_model = EfficientNetB0(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "EfficientNetB1":
            from efficientnet.keras import EfficientNetB1
            self.last_layer_name = "top_conv" 
            pre_trained_model = EfficientNetB1(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "EfficientNetB2":
            from efficientnet.keras import EfficientNetB2
            self.last_layer_name = "top_conv" 
            pre_trained_model = EfficientNetB2(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "EfficientNetB3":
            from efficientnet.keras import EfficientNetB3
            self.last_layer_name = "top_conv" 
            pre_trained_model = EfficientNetB3(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "EfficientNetB4":
            from efficientnet.keras import EfficientNetB4
            self.last_layer_name = "top_conv" 
            pre_trained_model = EfficientNetB4(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "EfficientNetB5":
            from efficientnet.keras import EfficientNetB5
            self.last_layer_name = "top_conv" 
            pre_trained_model = EfficientNetB5(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "EfficientNetB6":
            from efficientnet.keras import EfficientNetB6
            self.last_layer_name = "top_conv" 
            pre_trained_model = EfficientNetB6(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "EfficientNetB7":
            from efficientnet.keras import EfficientNetB7
            self.last_layer_name = "top_conv" 
            pre_trained_model = EfficientNetB7(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "DenseNet121":
            self.last_layer_name = "relu" 
            pre_trained_model = DenseNet121(input_shape = self.input_shape, include_top=False, weights="imagenet")
        elif self.type == "DenseNet169":
            self.last_layer_name = "relu" 
            pre_trained_model = DenseNet169(input_shape = self.input_shape, include_top=False, weights="imagenet")
        else:
            self.last_layer_name = "relu" 
            pre_trained_model = DenseNet201(input_shape = self.input_shape, include_top=False, weights="imagenet")
        pre_trained_model.summary()
        if self.retrain : 
            for layer in pre_trained_model.layers:
                layer.trainable = True
        return pre_trained_model
    def _get_metrics(self):
        return ["sen", "spe", "bacc", "acc"]
    def _get_activation(self):
        return 'sigmoid'
    def _get_lossfunc(self):
        if self.lossfunc == "":
            return "binary_crossentropy"
        return self.lossfunc
    def _get_last_layer_name(self):
        if self.custom_layer_name != "":
            return self.custom_layer_name
        return self.last_layer_name
    def _get_optimizer(self):
        opt = self.optimizer.lower()
        #
        lr = self.lr
        if self.epochs_fast > 0:
            lr = self.lr_fast
        if opt == "sgd":
            print("_get_optimizer->SGD")
            return SGD(
                lr=lr, 
                decay=0.0, 
                momentum=0.9, 
                nesterov=True)
        elif opt == "rmsprop":
            print("_get_optimizer->RMSprop")
            return RMSprop(
                lr=lr, 
                rho=0.9, 
                decay=0.0, 
                epsilon=None,
                )
        elif opt == "adadelta":
            print("_get_optimizer->Adadelta")
            return Adadelta(
                lr=lr, 
                rho=0.9, 
                decay=0.0, 
                epsilon=None,
                )
        elif opt == "adamax":
            print("_get_optimizer->Adamax")
            return Adamax(
                lr=lr, 
                beta_1=0.9, 
                beta_2=0.999,
                decay=0.0, 
                epsilon=None,
                )
        print("_get_optimizer->Adam")
        return Adam(
                lr=lr, 
                beta_1=0.9, 
                beta_2=0.999, 
                decay=0.0, 
                epsilon=None,
                amsgrad=False)
    def _get_balanced_image_data_generator(self):
        return BalancedImageDataGenerator(
                rotation_range=5, 
                width_shift_range=0.05, 
                height_shift_range=0.05,
                shear_range=0.05, 
                zoom_range=0.05, 
                fill_mode='nearest',
                balanced_type = self.balanced_type
                )
    def _get_image_data_generator(self):
        return ImageDataGenerator(
                rotation_range=5, 
                width_shift_range=0.05, 
                height_shift_range=0.05,
                shear_range=0.05, 
                zoom_range=0.05, 
                fill_mode='nearest'
                )
    def create_top_model(self, input_layer, output_layer,activation):
        x = layers.GlobalMaxPooling2D()(output_layer)
        if self.bn_top:
            x = BatchNormalization()(x)
        for ldata in self.top_layers:
            #x = layers.Dense(ldata["num"] * 1, activation=ldata["activation"])(x)
            x = layers.Dense(ldata["num"])(x)
            if ldata["batchnormalization"]:
                x = BatchNormalization()(x)
            x = Activation(ldata["activation"])(x)
            if ldata["dropout"] > 0:
                x = layers.Dropout(ldata["dropout"] * 1)(x) #0.5
        x = layers.Dense(1, activation=activation)(x)
        model = Model(input_layer, x)
        return model
    def summary(self):
        return self.cnn_model.summary()
    def init_model(self):
        print("bbc.init_model->type", self.type)
        print("bbc.init_model->retrain", self.retrain)        
        pre_trained_model = self._get_pre_trained_model()
        lossfunc = self._get_lossfunc()
        metrics = self._get_metrics()
        activation = self._get_activation()
        layer_name = self._get_last_layer_name()
        last_layer = pre_trained_model.get_layer(layer_name)
        last_output = last_layer.output
        model = self.create_top_model(pre_trained_model.input,last_output,activation)
        optimizer = self._get_optimizer()        
        print("bbc.init_model->loss=", lossfunc)
        model.compile(
            loss=lossfunc,
            optimizer=optimizer,
            metrics=metrics
        )
        self.cnn_model = model
        print("bbc.init_model->end")
    def _get_train_callbacks(self, steps_per_epoch):
        clr_fast, clr_slow = self._get_train_callbacks_lrs(steps_per_epoch)
        checkpoint_fast = self._get_train_fast_callbacks_save_file()
        checkpoint = self._get_train_callbacks_save_file()
        
        checkpoint_slow_best_loss = self._get_train_callbacks_save_best_file(pre="slow", monitor = "val_loss", mode="min")
        checkpoint_fast_best_loss = self._get_train_callbacks_save_best_file(pre="fast", monitor = "val_loss", mode="min")
        
        checkpoint_slow_best_sensitivity = self._get_train_callbacks_save_best_file(pre="slow", monitor = "val_sensitivity", mode="max")
        checkpoint_fast_best_sensitivity = self._get_train_callbacks_save_best_file(pre="fast", monitor = "val_sensitivity", mode="max")
        
        
        #return [[checkpoint, clr_fast],[checkpoint, clr_slow]]
        return [[checkpoint_fast, checkpoint_fast_best_loss, checkpoint_fast_best_sensitivity, clr_fast],[checkpoint,checkpoint_slow_best_loss, checkpoint_slow_best_sensitivity, clr_slow]]
    def _get_train_callbacks_lrs(self, steps_per_epoch):
        step_size = 2.0 * steps_per_epoch
        
        clr_fast = CyclicLR(base_lr=self.lr_fast_min, max_lr=self.lr_fast,
                            step_size=step_size,
                            mode='triangular2')
                            
        clr_slow = CyclicLR(base_lr=self.lr_min, max_lr=self.lr,
                            step_size=step_size * 2,
                            mode='triangular2')
        return [clr_fast, clr_slow]
    def _get_train_callbacks_save_best_file(self, pre="slow", monitor = "val_loss", mode="min"):
        #checkpoint
        
        fname = self._get_prefix_name()
        output_dir = self.output_dir + "_best"
        self.ensure_dir(output_dir)
        filepath = os.path.join(output_dir, pre + "_" + monitor + "_" + mode + "_" + fname + ".h5")
        return ModelCheckpoint(filepath, monitor = monitor, mode = mode, save_best_only = True)
    def _get_train_fast_callbacks_save_file(self):
        #checkpoint
        fname = self._get_prefix_name()
        output_dir = self.output_dir + "_fast"
        self.ensure_dir(output_dir)
        filepath = os.path.join(output_dir, fname + "_epochs{epoch:03d}_timer.h5")
        print("bbc.train->filepath=",filepath)
        return ModelCheckpoint(filepath)
    def _get_train_callbacks_save_file(self):
        #checkpoint
        fname = self._get_prefix_name()
        output_dir = self.output_dir
        self.ensure_dir(output_dir)
        filepath = os.path.join(output_dir, fname + "_epochs{epoch:03d}_timer.h5")
        print("bbc.train->filepath=",filepath)
        return ModelCheckpoint(filepath)
    def _get_train_epochs(self):
        return [self.epochs_fast, self.epochs_slow]
    def train(self, x_train, y_train, x_val, y_val):
        print("bbc.train->start")
        self.steps_per_epoch = 0
        val_batch_size = self.val_batch_size
        if val_batch_size <= 0:
            val_batch_size = x_val.shape[0]
        if self.balance > 0 :
            print("bbc.train->balance")
            train_datagen = self._get_balanced_image_data_generator()
            step, _, _ = _balanced_batch_apply(y_train, batch_size = self.batch_size, balanced_type = self.balanced_type)
            self.steps_per_epoch = step
        else:
            print("bbc.train->normal")
            train_datagen = self._get_image_data_generator()
            self.steps_per_epoch = x_train.shape[0] // self.batch_size
            
        
        val_datagen = ImageDataGenerator()
        validation_steps = x_val.shape[0] // val_batch_size
        
        train_datagen.fit(x_train)
        val_datagen.fit(x_val)
        
        cb_fast, cb_slow = self._get_train_callbacks(self.steps_per_epoch)
        
        epoch_fast, epoch_slow = self._get_train_epochs()
        
        if epoch_fast > 0:                     
            print("bbc.train->fit clr_fast")
            self._history = self.cnn_model.fit_generator(
                train_datagen.flow(
                    x_train,
                    y_train, 
                    batch_size = self.batch_size), 
                epochs = epoch_fast, 
                validation_data = val_datagen.flow(x_val, y_val, batch_size = val_batch_size), 
                verbose = 1, 
                steps_per_epoch= self.steps_per_epoch, 
                validation_steps= validation_steps,
                callbacks=cb_fast
                )
            self.save(suffix="_fast")
        if epoch_slow > 0 :
            print("bbc.train->fit clr_slow ")
            self._history = self.cnn_model.fit_generator(
                train_datagen.flow(
                    x_train,
                    y_train, 
                    batch_size = self.batch_size), 
                epochs = epoch_slow, 
                validation_data = val_datagen.flow(x_val, y_val, batch_size = val_batch_size), 
                verbose = 1, 
                steps_per_epoch= self.steps_per_epoch, 
                validation_steps= validation_steps,
                callbacks=cb_slow
                )
        print("bbc.train->end")
    def save (self, output_dir = "", fname = None, suffix=""):
        if output_dir == "":
            output_dir = self.output_dir
        if fname is None:
            fname = self._get_prefix()
        print("bbc.save->fname=",fname)   
        mf = os.path.join(output_dir + suffix, fname + "_model.h5")
        self.ensure_dir_for_file(mf)
        self.cnn_model.save(mf)
        
        hist_df = pd.DataFrame(self._history.history)
        hf = os.path.join(output_dir, fname + "_history.json")
        self.ensure_dir_for_file(hf)
        with open(hf, mode='w') as f:
            hist_df.to_json(f)
        print("bbc.save->end,fname=",fname) 
    def _get_prefix(self):
        ret = self._get_prefix_name()
        ret += "_epoch" + str(self.epochs_slow)
        return ret
    def _get_prefix_name(self):
        ret = ""
        ret += self.type
        if self.balance > 0:
            if self.balanced_type != "":
                ret += self.balanced_type
            else:
                ret += "stratified"
        if self.lossfunc:
            ret += self.lossfunc
        return ret
    def evaluate (self, x_test, y_test, x_val = None, y_val = None):
        f = self._get_prefix()
        if x_val is not None:
            y_proba = self.cnn_model.predict(x_val, verbose=1)
            print("==== REPORT START VALIDATION OF ", f, "=====")
            self.show(y_val, y_proba, title=f)
            print("==== REPORT END VALIDATION OF ", f, "=====")
        y_proba = self.cnn_model.predict(x_test, verbose=1)
        print("==== REPORT START AUC OF ", f, "=====")
        self.show(y_test, y_proba, title=f)
        print("==== REPORT END AUC OF ", f, "=====")
    def show(self, y_test, y_proba, title=""):
        y_pred = (y_proba > 0.5).astype('int32')
        
        all_test = len(y_test)
        all_p = len([e for e in y_test if e == 1])
        all_n = all_test - all_p
        true_p = len([i for i, j in zip(y_pred, y_test) if (i == j) and (i == 1) ])
        true_n = len([i for i, j in zip(y_pred, y_test) if (i == j) and (i == 0) ])
        pre = ""
        if title != "":
            pre = title + "->"
        sen = 1.0
        spe = 1.0
        if all_p > 0:
            sen = float(true_p) / all_p
        if all_n > 0:
            spe = float(true_n) / all_n 
        #print("y_pred: ", y_pred)
        #print("y_test: ", y_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        print(pre + "all_test: ", all_test)
        print(pre + "all_p: ", all_p)
        print(pre + "all_n: ", all_n)
        print(pre + "predict->true_p: ", true_p)
        print(pre + "predict->true_n: ", true_n)
        print(pre + "final->result")
        print(pre + "final->acc=",acc)
        print(pre + "final->auc=",auc)
        print(pre + "final->sen=",sen)
        print(pre + "final->spe=",spe, flush=True)
    def ensure_dir_for_file(self, file_path):
        directory = os.path.dirname(file_path)
        self.ensure_dir(directory)
    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)