import argparse
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import math
from resizeimage import resizeimage

def _test_doctor_performance_clinical():
    test_dir = "./data/testnd"
    csv_result = "./data/ResultsClinical.csv"
    csv_name = "./data/ClinicalNameSource.csv"
    return _test_doctor_performance(test_dir, csv_result, csv_name)
def _test_doctor_performance_dermoscopic():
    test_dir = "./data/test"
    csv_result = "./data/ResultsDermoscopic.csv"
    csv_name = "./data/DermoscopicNameSource.csv"
    return _test_doctor_performance(test_dir, csv_result, csv_name)
def _test_doctor_image_id(name):
    id = os.path.splitext(os.path.basename(name))[0]
    id = id.replace("ISIC_000000","")
    id = id.replace("ISIC_00000","")
    id = id.replace("ISIC_0000","")
    id = id.replace("ISIC_000","")
    id = id.replace("ISIC_00","")
    id = id.replace("ISIC_0","")
    id = id.replace("ISIC_","")
    #id = id.sub('^0+','')
    return id
def _test_doctor_performance(test_dir, csv_result, csv_name):
    print("_test_doctor_performance->test_dir=", test_dir)
    print("_test_doctor_performance->csv_result=", csv_result)
    print("_test_doctor_performance->csv_name=", csv_name)
    data = pd.read_csv(csv_name)
    #print(data)
    labels = {}
    for fin in sorted(glob(test_dir + "/*/*.jpg")):
        x = _test_doctor_image_id(fin)
        y = 0
        if fin.find("malign") > 0:
            y = 1
        labels[x] = y
    #return labels
    #print(labels)
    y_true = np.zeros(100)
    #data['y'] = data["file name"].map(labels.get)
    for index, row in data.iterrows():
        
        y_true[index] = labels[str(row["file name"])]
    data_preds = pd.read_csv(csv_result)
    y_preds = []
    #print(data_preds)
    for index, data_pred in data_preds.iterrows():
        #print (data_pred)
        #print ("Doctor index=",index)
        y_data = []
        for i in range(100):
            y_data.append(data_pred["Answer " + str(i+1)])
        y_pred = _test_doctor_pred(y_data)
        y_preds.append(y_pred)
    return [y_true, y_preds]

def _test_doctor_pred(y_data):
    y_pred = []
    #print(y_data)
    for y in y_data:
        y = _test_doctor_pred_y(y)
        y_pred.append(y)
    return y_pred
def _test_doctor_pred_y(y):
    #print("_test_doctor_pred_y->y=",y)
    if y.find("biopsy") != -1:
        return 1
    return 0
def _test_image_ids(test_dir, inclsampled = True):
    itest = []
    for fin in sorted(glob(test_dir + "/*/*.jpg")):
        id = os.path.splitext(os.path.basename(fin))[0]
        itest.append(id)
        if inclsampled:
            itest.append(id + "_downsampled")
    return itest
def _test_convert(data_dir, output_dir, suff="dermoscopic"):
    print("_test_convert->data_dir=", data_dir)
    print("_test_convert->output_dir=", output_dir)
    #_test_image_crop(data_dir, output_dir)
    w = 256
    h = 192
    idatas = []
    ydatas = []
    xdatas = []
    for fin in sorted(glob(data_dir + "/*/*.jpg")):
        idatas.append(fin)
        y = 0
        if fin.find("malign") > 0:
            y = 1
        ydatas.append(y)
        x = fin
        xdatas.append(np.asarray(Image.open(x).resize((w, h), resample=Image.LANCZOS).convert("RGB")))
    #print(idatas)
    print(ydatas)
    
    #xdatas = _ham_images_load(idatas, w, h)
    xdatas = np.stack(xdatas, 0)
    xdatas = xdatas.astype("float32")
    xdatas /= 255
    ilen = len(ydatas)
    _ham_save_data(output_dir, "x_test." + suff, xdatas)
    _ham_save_data(output_dir, "y_test." + suff, ydatas)
def _test_image_crop(data_dir, output_dir):
    print("_test_image_crop->data_dir=", data_dir)
    print("_test_image_crop->output_dir=", output_dir)
    for fin in sorted(glob(data_dir + "/*/*.jpg")):
        fout = fin.replace(data_dir, output_dir)
        _image_crop(fin, fout)
def _image_ratio(w,h):
    return math.floor(w * 10000.0 / h) / 10000
def _image_crop(fin,fout, w=4.0, h=3.0):
    ratio = _image_ratio(w,h)
    ret = 0
    if os.path.isfile(fout):
        return ret
    with open(fin, 'r+b') as f:
        with Image.open(f) as image:
            ensure_dir_for_file(fout)
            width, height = image.size
            nratio = _image_ratio(width, height)
            if nratio == ratio :
                image.save(fout)
                return ret
            elif nratio > ratio:
                #height
                nh = height
                nw = nh * w / h
            else:
                nw = width
                nh = nw * h / w
            print("_image_crop->width=",width)
            print("_image_crop->height=",height)
            print("_image_crop->nw=",nw)
            print("_image_crop->nh=",nh)
            print("_image_crop->fin=",fin)
            print("_image_crop->ratio=",ratio)
            img = resizeimage.resize_crop(image, [nw, nh])
            img.save(fout, image.format)
    return ret
def _get_image_id(x):
    return x.replace("_downsampled","")
def _ham_image_crop(data_dir, output_dir):
    print("_ham_image_crop->data_dir=", data_dir)
    print("_ham_image_crop->output_dir=", output_dir)
    for fin in sorted(glob(data_dir + "/*/*.jpg")):
        fout = fin.replace(data_dir, output_dir)
        _image_crop(fin, fout)
def _ham_convert(data_dir, output_dir, test_dir=""):
    print("_ham_convert->data_dir=", data_dir)
    print("_ham_convert->output_dir=", output_dir)
    ham_datas = _ham_metadata(data_dir)
    #print(ham_datas)
    ham_images = _ham_scan_images(data_dir)
    #print(ham_images)
    #return 
    ham_dict = _ham_malignant_dict()
    #print(ham_datas)
    test_ids = _test_image_ids(test_dir)
    #print(test_ids)
    
    ham_datas = ham_datas[~ham_datas["image"].isin(test_ids)]
    #print(ham_datas)
    #return ham_datas
    ham_datas = _ham_data_convert(ham_datas, ham_images, ham_dict)
    #print(ham_datas)
    #return
    w = 256
    h = 192
    #ydatas = ham_datas["labels"].values
    ydatas = ham_datas["MEL"].values
    xdatas = _ham_images_load(ham_datas["path"], w, h)
    xdatas = np.stack(xdatas, 0)
    del ham_datas
    xdatas = xdatas.astype("float32")
    xdatas /= 255
    ilen = len(ydatas)
    percentage = 0.1
    test_val_size = math.floor(percentage * ilen)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        xdatas, 
        ydatas, 
        test_size = test_val_size, 
        random_state = 42,
        shuffle = True,
        stratify = ydatas)
    del xdatas
    _ham_save_data(output_dir, "x_test", x_test)
    _ham_save_data(output_dir, "y_test", y_test)
    del x_test
    del y_test
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, 
        y_train_val, 
        test_size = test_val_size, 
        random_state = 42,
        shuffle = True,
        stratify = y_train_val)
    del x_train_val
    del y_train_val
    _ham_save_data(output_dir, "x_train", x_train)
    _ham_save_data(output_dir, "y_train", y_train)
    _ham_save_data(output_dir, "x_val", x_val)
    _ham_save_data(output_dir, "y_val", y_val)
def _ham_get_data(output_dir, fname):
    f = os.path.join(output_dir, fname + ".npy")
    return np.load(f)
def _ham_save_data(output_dir, fname, data):
    f = os.path.join(output_dir, fname + ".npy")
    ensure_dir_for_file(f)
    np.save(f, data)
def _ham_images_load(images, w, h):
    return images.map(lambda x: np.asarray(Image.open(x).resize((w, h), resample=Image.LANCZOS).convert("RGB")))

def _ham_data_convert(ham_datas, ham_images, ham_dict):
    ham_datas["path"] = ham_datas["image"].map(ham_images.get)
    ham_datas = ham_datas[(ham_datas["MEL"] == 1) | (ham_datas["NV"] == 1)]
    ham_datas = ham_datas[ham_datas["path"].notnull()]
    #ham_datas["labels"] = ham_datas["dx"].map(ham_dict.get)
    return ham_datas
    

def _ham_metadata(data_dir):
    directory = os.path.dirname(data_dir)
    return pd.read_csv(os.path.join(directory, 'ISIC_2019_Training_GroundTruth.csv'))
    #print(directory)
    #return 0
    #return pd.read_csv(os.path.join(data_dir, 'ISIC_2019_Training_GroundTruth.csv'))
def _ham_scan_images(data_dir):
    return {os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(data_dir, '*/*.jpg'))}

def _ham_malignant_dict():
    #1 for malignant
    #0 for benign
    return {
            'nv': 0,
            'bkl': 0,
            'vasc': 0,
            'df': 0,
            'mel': 1,
            'bcc': 1,
            'akiec': 1
            }
        
def ensure_dir_for_file(file_path):
    directory = os.path.dirname(file_path)
    ensure_dir(directory)
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)