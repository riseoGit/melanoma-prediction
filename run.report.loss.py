import argparse
import os
import glob
import numpy as np
from sklearn.metrics import classification_report, accuracy_score,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from keras import backend as K
import efficientnet.keras
from keras.models import load_model
from butil.dataset import _ham_get_data, ensure_dir_for_file
import keras.losses
from models.imbalanced.balanced_losses import *

import functools
print = functools.partial(print, flush=True)
def _loss_model_files(core="DenseNet169", suf="bnb50"):
    return {
            "ORI":_loss_model_file(type="ORI",core=core,suf=suf)
            ,"BON":_loss_model_file(type="BON",core=core,suf=suf)
            ,"BLF":_loss_model_file(type="BLF",core=core,suf=suf)
            }
def _loss_model_file(core="DenseNet169", suf="bnb50", type="ORI", best="val_loss_min"):
    pattern = ""
    if type == "ORI":
        pattern = "{core}-adam-normal-binary_crossentropy-{suf}_best/slow_{best}_{core}.h5"
    elif type == "BON":
        #pattern = "{core}-adam-stratified-binary_crossentropy-{suf}-2_best/slow_{best}_{core}stratified.h5"
        pattern = "{core}-adam-stratified-binary_crossentropy-{suf}_best/slow_{best}_{core}stratified.h5"
    else:
        pattern = "{core}-adam-stratified-binary_balanced_pmse-{suf}_best/slow_{best}_{core}stratifiedbinary_balanced_pmse.h5"
    return pattern.format(core=core, suf=suf, best=best)
def show(y_test, y_proba, title=""):
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
    print(pre + "true_p: ", true_p)
    print(pre + "true_n: ", true_n)
    print(pre + "final->result")
    print(pre + "final->acc=",acc)
    print(pre + "final->auc=",auc)
    print(pre + "final->sen=",sen)
    print(pre + "final->spe=",spe, flush=True)
    return [acc,auc,sen,spe]

def main(FLAGS):
    core = FLAGS.core
    if core == "":
        core = "DenseNet169"
    data_dir = FLAGS.data_dir
    model_dir = FLAGS.model_dir
    output_dir = FLAGS.output_dir
    output_dir = os.path.join(output_dir, core)
    
    out_file = os.path.basename(model_dir)
    out_file = os.path.join(output_dir, "final_loss_" + out_file + ".npy")
    ensure_dir_for_file(out_file)
    
    print("out_file=",out_file)        

    x_val = _ham_get_data(data_dir, "x_val")
    y_val = _ham_get_data(data_dir, "y_val")
    
    x_test = _ham_get_data(data_dir, "x_test")
    y_test = _ham_get_data(data_dir, "y_test")
    
    x_dermoscopic = _ham_get_data(data_dir, "x_test.dermoscopic")
    y_dermoscopic = _ham_get_data(data_dir, "y_test.dermoscopic")
    
    
    fdatas = []
    files = _loss_model_files(core=core, suf="bnb50")
    print (files)
    for key,val in files.items():
        #print (key, "=>", val)
        f = os.path.join(model_dir, val)
        model = load_model(f,compile=True)
        print("==== REPORT VALIDATION OF ", key , "=====")
        
        y_val_proba = model.predict(x_val, verbose=0, batch_size=64)
        show(y_val, y_val_proba, title="VALIDATION->" + key)
        
        y_test_proba = model.predict(x_test, verbose=0, batch_size=64)
        show(y_test, y_test_proba, title="TEST->" + key)
        
        y_dermoscopic_proba = model.predict(x_dermoscopic, verbose=0, batch_size=64)
        show(y_dermoscopic, y_dermoscopic_proba, title="DERMOSCOPIC->" + key)
        
        
        data = {'type':key
                ,'val':{'y_true':y_val, 'y_pred':y_val_proba}
                ,'test':{'y_true':y_test, 'y_pred':y_test_proba}
                ,'dermoscopic':{'y_true':y_dermoscopic, 'y_pred':y_dermoscopic_proba}
                }
        fdatas.append(data)
        print ("main->save to file=", out_file)
        np.save(out_file, fdatas)
        #break
        K.clear_session()
if __name__ == "__main__":
    
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--core',
        type=str,
        default='DenseNet169',
        help='core'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/binary',
        help='input data directory'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='result/models/scientific-reports',
        help='model_dir'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='result/best_loss',
        help='model_dir'
    )
    
    FLAGS = parser.parse_args()
    print ("core=",FLAGS.core)
    print ("data_dir=",FLAGS.data_dir)
    print ("model_dir=",FLAGS.model_dir)
    print ("output_dir=",FLAGS.output_dir)
    main(FLAGS)
                        