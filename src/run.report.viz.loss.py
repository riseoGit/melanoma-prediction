import argparse
import os
import glob
import numpy as np
from butil.dataset import _ham_get_data, ensure_dir_for_file, _test_doctor_performance_dermoscopic, _test_doctor_performance_clinical
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import math
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,roc_auc_score
global _ty_true, _ty_preds
_ty_true = []
_ty_preds = []

def __test_doctor_performance_dermoscopic():
    global _ty_true, _ty_preds
    if len(_ty_true) == 0:
        _ty_true, _ty_preds = _test_doctor_performance_dermoscopic()
    return [_ty_true, _ty_preds]
def report_auc_sen_spe(y_true, y_proba):
    y_pred = (y_proba > 0.5).astype('int32')
    all_test = len(y_true)
    all_p = len([e for e in y_true if e == 1])
    all_n = all_test - all_p
    true_p = len([i for i, j in zip(y_pred, y_true) if (i == j) and (i == 1) ])
    true_n = len([i for i, j in zip(y_pred, y_true) if (i == j) and (i == 0) ])
    sen = 1.0
    spe = 1.0
    if all_p > 0:
        sen = float(true_p) / all_p
    if all_n > 0:
        spe = float(true_n) / all_n 
    auc = roc_auc_score(y_true, y_proba)
    return [auc,sen,spe]
def report_metric(y_test, y_pred):
    all_test = len(y_test)
    all_p = len([e for e in y_test if e == 1])
    all_n = all_test - all_p
    true_p = len([i for i, j in zip(y_pred, y_test) if (i == j) and (i == 1) ])
    true_n = len([i for i, j in zip(y_pred, y_test) if (i == j) and (i == 0) ])
    
    sen = 1.0
    spe = 1.0
    if all_p > 0:
        sen = float(true_p) / all_p
    if all_n > 0:
        spe = float(true_n) / all_n
    return [sen,spe]
def report_load(f):
    data = np.load(f,allow_pickle=True)
    return data.flat
def report_colors(type = ""):
    if type == "bar":
        return ["blue","green","purple","pink","red","orange","gray","violet","yellow","tan","gold","darkcyan","skyblue"]
    elif type == "roc":
        return ["blue","green","purple","pink","red","orange","gray","violet","yellow","tan","gold","darkcyan","skyblue"]
    return ["tan","gold","darkcyan","skyblue","yellow","green","purple","pink","red","orange","gray","blue","violet","hotpink","maroon","burlywood","kaki"]
def report_markers():
    return ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd','p','h','*','|','0','1','2','3']
def report_show_roc(y_true, y_pred, name, ind, marker = True):
    colors = report_colors("roc")
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    label = "ROC of %s, AUC = %0.1f" % (name, 100 * roc_auc)
    if marker:
        markersize = 8 - ind * 2
        plt.plot(fpr, tpr, colors[ind], marker='.', label = label, markersize = markersize )
    else:
        plt.plot(fpr, tpr, colors[ind], label = label)
def report_show_doctor(y_true, y_preds):
    label = "Dermatologists"
    c = "red"
    x = []
    y = []    
    for y_pred in y_preds:
        sen,spe = report_metric(y_true, y_pred)
        x.append(1.0 - spe)
        y.append(sen)
    print("report_show_doctor->SEN=\n", y)
    print("report_show_doctor->1 - SPE=\n", x)
    plt.scatter(x, y, marker='.', color = c, label= label)
def report_filter_spe(y_test, y_pred, sthreshold=0):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    n = len(tpr)
    ret = []
    msen = 0
    for i in range(n):
        sen = tpr[i]
        if (sen < sthreshold):
            continue
        spe = 1 - fpr[i]
        if msen == 0:
            msen = sen
            ret.append(spe)
        elif msen == sen:
            ret.append(spe)
        else:
            break
    return ret
def report_filter_sen(y_test, y_pred, sthreshold=0):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    n = len(tpr)
    ret = []
    mspe = 0
    for i in range(n-1,-1,-1):
        spe = 1 - fpr[i]
        if (spe < sthreshold):
            continue
        sen = tpr[i]
        if mspe == 0:
            mspe = spe
            ret.append(sen)
        elif mspe == spe:
            ret.append(sen)
        else:
            break
    return ret
def report_compare_doctors(y_true, y_pred):
    ty_true, ty_preds = __test_doctor_performance_dermoscopic()
    total_doctor_win = 0
    total_doctor_eq = 0
    total_doctor_loss = 0
    for ty_pred in ty_preds:
        sen,spe = report_metric(ty_true, ty_pred)
        cnnspedatas = report_filter_spe(y_true, y_pred, sen)
        cnnsped = np.amax(cnnspedatas)
        if (cnnsped > spe):
            total_doctor_win += 1
        elif (cnnsped == spe):
            total_doctor_eq += 1
        else:
            total_doctor_loss += 1
    return [total_doctor_win,total_doctor_eq,total_doctor_loss]
def report_chart_roc_info(datas, l):
    ret = ""
    mean   = np.mean(datas)
    max = np.amax(datas)
    min = np.amin(datas)
    ret += l + "=" + "%0.1f" % (100* max)
    return ret
def report_roc_compare_thresholds(y_true, y_pred, thresholds):
    ret = ""
    pre = "\n"
    for athreshold in thresholds:
        if (athreshold["type"] == "spe"):
            mdatas = report_filter_spe(y_true, y_pred, athreshold["thres"])
        else:
            mdatas = report_filter_sen(y_true, y_pred, athreshold["thres"])
  
        ret += pre + report_chart_roc_info(mdatas,athreshold["label"] + "->" + athreshold["type"])
    return ret
def report_roc_by(y_true, y_pred, f="", name = "", title="", show_d = True, show_compare = True, show_compare_thresholds = True):
    print("report_roc_by->start")
    thresholds = [ 
                    {"thres": 1.0,"type":"spe","label":"sen(100)"},
                    {"thres": 0.95,"type":"spe","label":"sen(95.0)"},
                    {"thres": 0.9,"type":"spe","label":"sen(90.0)"},
                    {"thres": 0.85,"type":"spe","label":"sen(85.0)"},
                    {"thres": 0.80,"type":"spe","label":"sen(80.0)"},
                    {"thres": 0.767,"type":"spe","label":"sen(76.7)"},
                    {"thres": 0.741,"type":"spe","label":"sen(74.1)"},
                    {"thres": 1.0,"type":"sen","label":"spe(100)"},
                    {"thres": 0.95,"type":"sen","label":"spe(95.0)"},
                    {"thres": 0.9,"type":"sen","label":"spe(90.0)"},
                    {"thres": 0.85,"type":"sen","label":"spe(85.0)"},
                    {"thres": 0.8,"type":"sen","label":"spe(80.0)"},
                    #{"thres": 0.75,"type":"sen","label":"spe(75.0)"},
                    {"thres": 0.692,"type":"sen","label":"spe(69.2)"},
                    {"thres": 0.600,"type":"sen","label":"spe(60.0)"},
                  ]
    plt.clf()
    report_show_roc(y_true, y_pred, name, 0, marker=True)
    #Dermatologists
    if show_d :
        ty_true, ty_preds = __test_doctor_performance_dermoscopic()
        report_show_doctor(ty_true, ty_preds)
    #Comparasion with Dermatologists
    gtitle = "Dermatologists"
    if show_compare_thresholds:
        gtitle += report_roc_compare_thresholds(y_true, y_pred, thresholds)
    if show_compare :
        win,eq,ls = report_compare_doctors(y_true, y_pred)
        dtitle = "WIN = " + str(win) 
        dtitle += ", EQUAL= " + str(eq)  
        dtitle += ", LOSS= " + str(ls)          
        plt.text(0.25, .02, gtitle + "\n" + dtitle)
    if title != "":
        plt.title(legend)
    
    plt.legend(loc = 'center right',scatterpoints=1)
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    ensure_dir_for_file(f)
    plt.savefig(f)
    plt.close()
    print("report_rocs_by->end")
def report_chart_autolabel(rects, ax, fontsize):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + .006,
                '%.01f' % float(height * 100),
        ha='center', va='bottom', fontsize=fontsize)
def report_chart_bars(datas, f="", by="dermoscopic"):
    print("report_chart_bars->start,by=", by)
    colors = report_colors(type="bar")
    bar_width = 0.18
    fontsize = 8
    opacity = 1
    plt.clf()
    fig, ax = plt.subplots()
    ind = 0
    dlen = len(datas)
    groups = ["AUC","SEN","SPE"]
    index = np.arange(dlen)
    for ind in range(dlen):
        data = datas[ind]
        #bdatas = np.zeros(dlen,dtype=float)
        print("report_chart_bars->type=",data["type"])
        #groups.append(data["type"])
        y_true = data[by]["y_true"]
        y_pred = data[by]["y_pred"]
        bdatas = report_auc_sen_spe(y_true,y_pred)
        #bdatas[0] = roc_auc
        rects = plt.bar(index + ind * bar_width, 
            bdatas, 
            bar_width,
            alpha=opacity,
            color=colors[ind],
            label=data["type"])
        report_chart_autolabel(rects, ax, fontsize)
    
    ax.set_ylabel('Performance')
    '''if by == "dermoscopic":
        ax.set_xlabel("Performances tested by MClass-D using a prediction threshold of 0.5")
    else:
        ax.set_xlabel("Performances tested by test-10 using a prediction threshold of 0.5")
    '''
    ax.set_xticks(index + bar_width * (dlen - 1) / 2.0)
    ax.set_xticklabels(tuple(groups))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=dlen)
    ensure_dir_for_file(f)
    plt.savefig(f)
    plt.close()
    print("report_chart_bars->end,by=", by)
def report_rocs_by(datas=[],f="", by="dermoscopic", show_d=True, title="", marker=True):
    print("report_rocs_by->start,by=", by)
    plt.clf()
    ind = 0
    #print(datas[0])
    for ind in range(len(datas)):
        data = datas[ind]
        print("report_rocs_by->type=",data["type"])
        y_true = data[by]["y_true"]
        y_pred = data[by]["y_pred"]
        report_show_roc(y_true, y_pred, data["type"], ind, marker)
        #ind += 1
    #Dermatologists
    if show_d :
        ty_true, ty_preds = __test_doctor_performance_dermoscopic()
        report_show_doctor(ty_true, ty_preds)
    if title != "":
        plt.title(legend)
    
    plt.legend(loc = 'center right',scatterpoints=1)
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    ensure_dir_for_file(f)
    plt.savefig(f)
    plt.close()
    print("report_rocs_by->end,by=", by)
def report_show_rocs_dermoscopic(datas, output_dir):
    f = os.path.join(output_dir, "roc_final_dermoscopic.jpg")
    report_rocs_by(datas,f=f, by="dermoscopic", show_d=True, title="")
def report_show_rocs_isic_test(datas, output_dir):
    f = os.path.join(output_dir, "roc_final_isic_test_doctor.jpg")
    report_rocs_by(datas,f=f, by="test", show_d=True, title="",marker=False)
    f = os.path.join(output_dir, "roc_final_isic_test.jpg")
    report_rocs_by(datas,f=f, by="test", show_d=False, title="",marker=False)
def report_show_rocs_isic_val(datas, output_dir):
    f = os.path.join(output_dir, "roc_final_isic_val_doctor.jpg")
    report_rocs_by(datas,f=f, by="val", show_d=True, title="",marker=False)
    f = os.path.join(output_dir, "roc_final_isic_val.jpg")
    report_rocs_by(datas,f=f, by="val", show_d=False, title="",marker=False)
def report_show_roc_detail(datas, output_dir):
    #by="dermoscopic"
    labels = ["val","test","dermoscopic"]
    for by in labels:
        for ind in range(len(datas)):
            data = datas[ind]
            print("report_show_roc_detail->type=",data["type"])
            y_true = data[by]["y_true"]
            y_pred = data[by]["y_pred"]
            report_print_by_threshold(y_true, y_pred, data["type"], by)
            f = os.path.join(output_dir, "detail_roc_" + by + "_" + data["type"] + ".jpg")
            report_roc_by(y_true, y_pred, f=f, name = data["type"], title="", show_d = True, show_compare = True)
def report_print_by_threshold(y_true, y_pred, type, by):
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    n = len(tpr)
    ret = ""
    for i in range(n):
        spe = 1 - fpr[i]
        sen = tpr[i]
        t = "\n%0.5f\t%0.5f\t%0.8f" % (sen, spe, threshold[i])
        ret += t
    print("=====FINAL RESULT ========", type, by)
    print(ret)
    print("=====END FINAL RESULT ========", type, by)
def report_save_bars(datas, output_dir):
    labels = ["val","test","dermoscopic"]
    for by in labels:
        f = os.path.join(output_dir, "bar_" + by + ".jpg")
        report_chart_bars(datas, f = f, by= by)
def main(FLAGS):
    data_file = FLAGS.data_file
    output_dir = FLAGS.output_dir
    rdatas = report_load(data_file)
    report_show_rocs_isic_test(rdatas,output_dir)
    report_show_rocs_isic_val(rdatas,output_dir)
    report_show_rocs_dermoscopic(rdatas,output_dir)
    report_show_roc_detail(rdatas, output_dir)
    report_save_bars(rdatas, output_dir)
if __name__ == "__main__":
    
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_file',
        type=str,
        default='result/best_loss/ResNet50/final_loss_scientific-reports.npy',
        help='input data file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='result/report-best_loss-InceptionV3',
        help='output_dir'
    )
    FLAGS = parser.parse_args()
    print ("data_file=",FLAGS.data_file)
    print ("output_dir=",FLAGS.output_dir)
    main(FLAGS)
                        