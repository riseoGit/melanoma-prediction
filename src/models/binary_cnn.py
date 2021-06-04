import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
#from keras.applications.resnext import ResNeXt50, ResNeXt101
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from .losses import *
from .cyclical_learning_rate import CyclicLR
from .image_data_generator import BalanceImageDataGenerator
from .binary_inception_resnet import BalanceBinaryInceptionResnet
from .balanced_generator import _balanced_batch_apply

class BalanceBinaryCnn(object):
    def __init__(self,
                 type,
                 retrain=False,
                 epochs = 5,
                 batch_size = 64,
                 val_batch_size = 0,
                 balance = 0,
                 lossfunc = "",
                 output_dir = "cnn_model",
                 lr = 0.00001,
                 dropout = 0.5,
                 input_shape = (192, 256, 3)
        ):
        self.type = type
        self.retrain = retrain
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.lossfunc = lossfunc
        self.input_shape = input_shape
        self.output_dir = output_dir
        self.cnn_model = None
        self.lr = lr
        self.dropout = dropout
        self.last_layer_name = ""
        self.balance = False
        self.balanced_type = ""
        if balance == 1:
            self.balance = True
            self.balanced_type = ""
        elif balance == 2:
            self.balance = True
            self.balanced_type = "balanced"
        
    def init_model (self):
        print("binary_cnn.init_model->type", self.type)
        print("binary_cnn.init_model->retrain", self.retrain)
        #_include_loss_and_metric_func()
        lossfunc = self.lossfunc
        
        if self.lossfunc == "":
            lossfunc = "binary_crossentropy"
            
        metrics = [sensitivity, specificity, bacc, "acc"]
        
            
        activation='sigmoid'
           
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
            pre_trained_model.summary()
        else:
            self.last_layer_name = "relu" 
            pre_trained_model = DenseNet201(input_shape = self.input_shape, include_top=False, weights="imagenet")
        if self.retrain : 
            for layer in pre_trained_model.layers:
                #print(layer.name)
                layer.trainable = True
        last_layer = pre_trained_model.get_layer(self.last_layer_name)
        last_output = last_layer.output
        x = layers.GlobalMaxPooling2D()(last_output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x) #0.5
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x) #0.5
        x = layers.Dense(1, activation=activation)(x)

        model = Model(pre_trained_model.input, x)
        optimizer = Adam(
            lr=self.lr, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=None, 
            decay=0.0, 
            amsgrad=False)
        
        print("binary_cnn.init_model->loss=", lossfunc)
        model.compile(
            loss=lossfunc,
            optimizer=optimizer,
            metrics=metrics
        )
        self.cnn_model = model
        print("binary_cnn.init_model->end")
    def train(self, x_train, y_train, x_val, y_val):
        print("binary_cnn.train->start")
        
        val_batch_size = self.val_batch_size
        if val_batch_size <= 0:
            val_batch_size = x_val.shape[0]
        if self.balance :
            print("binary_cnn.train->balance")
            train_datagen = BalanceImageDataGenerator(
                rotation_range=15, 
                width_shift_range=0.2, 
                height_shift_range=0.2,
                shear_range=0.2, 
                zoom_range=0.2, 
                fill_mode='nearest',
                balanced_type = self.balanced_type
                )
            
            #steps_per_epoch = self._get_steps_per_epoch(y_train, self.batch_size)
            steps_per_epoch, _, _ = _balanced_batch_apply(y_train, batch_size = self.batch_size, balanced_type = self.balanced_type)
        
        else:
            print("binary_cnn.train->normal")
            train_datagen = ImageDataGenerator(
                rotation_range=5, 
                width_shift_range=0.1, 
                height_shift_range=0.1,
                shear_range=0.1, 
                zoom_range=0.1, 
                fill_mode='nearest'
                )
            steps_per_epoch = x_train.shape[0] // self.batch_size
            
        
        val_datagen = ImageDataGenerator()
        validation_steps = x_val.shape[0] // val_batch_size
        
        train_datagen.fit(x_train)
        val_datagen.fit(x_val)
        
        #checkpoint
        fname = self._get_prefix_name()
        output_dir = self.output_dir
        self.ensure_dir(output_dir)
        filepath = os.path.join(output_dir, fname + "_epochs{epoch:03d}_timer.h5")
        print("binary_cnn.train->filepath=",filepath)
        checkpoint = ModelCheckpoint(filepath)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.9, min_lr = self.lr / 1000.0, cooldown=1)
        
        step_size = 2.0 * steps_per_epoch
        
        clr_fast = CyclicLR(base_lr=self.lr / 10.0, max_lr=self.lr,
                            step_size=step_size,
                            mode='triangular2')
                            
        clr_slow = CyclicLR(base_lr=self.lr / 10000.0, max_lr=self.lr,
                            step_size=step_size * 2.0,
                            mode='triangular2')
                                
        print("binary_cnn.train->fit clr_fast")
        self._history = self.cnn_model.fit_generator(
            train_datagen.flow(
                x_train,
                y_train, 
                batch_size = self.batch_size), 
            epochs = 20, 
            validation_data = val_datagen.flow(x_val, y_val, batch_size = val_batch_size), 
            verbose = 1, 
            steps_per_epoch= steps_per_epoch, 
            validation_steps= validation_steps,
            #callbacks=[checkpoint,learning_rate_reduction]
            callbacks=[checkpoint,clr_fast]
            )
        self.save()
        print("binary_cnn.train->fit clr_slow ")
        self._history = self.cnn_model.fit_generator(
            train_datagen.flow(
                x_train,
                y_train, 
                batch_size = self.batch_size), 
            epochs = self.epochs, 
            validation_data = val_datagen.flow(x_val, y_val, batch_size = val_batch_size), 
            verbose = 1, 
            steps_per_epoch= steps_per_epoch, 
            validation_steps= validation_steps,
            #callbacks=[checkpoint,learning_rate_reduction]
            callbacks=[checkpoint,clr_slow]
            )
        print("binary_cnn.train->end")
    def save (self, output_dir = "", fname = None):
        if output_dir == "":
            output_dir = self.output_dir
        if fname is None:
            fname = self._get_prefix()
        print("binary_cnn.save->fname=",fname)   
        mf = os.path.join(output_dir, fname + "_model.h5")
        self.ensure_dir_for_file(mf)
        self.cnn_model.save(mf)
        
        hist_df = pd.DataFrame(self._history.history)
        hf = os.path.join(output_dir, fname + "_history.json")
        self.ensure_dir_for_file(hf)
        with open(hf, mode='w') as f:
            hist_df.to_json(f)
        print("binary_cnn.save->end,fname=",fname) 
    def _get_prefix(self):
        ret = self._get_prefix_name()
        ret += "_epoch" + str(self.epochs)
        return ret
    def _get_prefix_name(self):
        ret = ""
        ret += self.type
        if self.balance:
           ret += "Balance"
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