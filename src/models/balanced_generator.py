import os
import numpy as np

def _balanced_batch_apply(y = None, batch_size = 32, balanced_type = ""):
    inds_one = [i for i,x in enumerate(y) if x == 1]
    inds_zero = [i for i,x in enumerate(y) if x == 0]
    len_inds_one = len(inds_one)
    len_inds_zero = len(inds_zero)
    batch_size_one = 0
    batch_size_zero = 0
    if balanced_type == "balanced":
        batch_size_one = batch_size // 2
    else:
        batch_size_one = (batch_size * len_inds_one) // (len_inds_one + len_inds_zero)
    batch_size_zero = batch_size - batch_size_one
    step_one = len_inds_one // batch_size_one
    step_zero = len_inds_zero // batch_size_zero
    step = min([step_one, step_zero])
    return step, batch_size_one, batch_size_zero
    
class BalancedGenerator():
    def _balance_index(self, y, batch_size, balanced_type):
        inds_one = [i for i,x in enumerate(y) if x == 1]
        inds_zero = [i for i,x in enumerate(y) if x == 0]
        step, batch_size_one, batch_size_zero = _balanced_batch_apply(y, batch_size = batch_size, balanced_type = balanced_type)
        list_one = self._balance_splice(inds_one, batch_size_one, step)
        list_zero = self._balance_splice(inds_zero, batch_size_zero, step)
        ret = []
        for idx in range(step):
            data_one = list_one[batch_size_one * idx : batch_size_one * (idx + 1)]
            data_zero = list_zero[batch_size_zero * idx : batch_size_zero * (idx + 1)]
            step_data = np.append(data_one, data_zero).astype(int)
            np.random.shuffle(step_data)
            if (idx == 0):
                ret = step_data
            else:
                ret = np.append(ret, step_data)
        return ret
    def _balance_splice(self, datas, batch_size, step):
        total = step * batch_size
        ret = []
        alen = len(datas)
        np.random.shuffle(datas)
        if (total <= alen):
            return datas[0:total]
        
        if (alen <= batch_size):
            n = total // alen
            l = total - n
            if n > 0:
                for i in range(n):
                    ret = np.append(ret,datas)
                    np.random.shuffle(datas)
            if l > 0:
                np.random.shuffle(datas)
                ret = np.append(ret,datas[0:l])
            return ret
        step_amount = (alen // batch_size) * batch_size
        step_num = total // step_amount
        l = total - step_num * step_amount
        
        for i in range(step_num):
            ret = np.append(ret,datas[0:step_amount])
            np.random.shuffle(datas)
        if l > 0:
            np.random.shuffle(datas)
            ret = np.append(ret,datas[0:l])
        return ret