from sklearn.externals import joblib


import numpy as np
from scipy import signal
import scipy.stats
import cPickle
import os
from scipy import signal
import librosa 

import config1 as cfg
import sys
import prepare_data1 as pp_data

from keras.layers import InputLayer, Flatten, Dense, Dropout
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.models import model_from_json ,load_model

from keras.callbacks import ModelCheckpoint

def sparse_to_categorical(x, n_out):
    x = x.astype(int)   # force type to int
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))


def mat_2d_to_3d(x, agg_num, hop): 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
     
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def pred(rd_path,ld_md,scaler):
    # Recognize and get statistics
    n_concat =cfg.n_concat
    hop = cfg.hop            # step_len
    fold = cfg.fold
    n_labels = len(cfg.labels)
    confuse_mat = np.zeros((n_labels, n_labels))      # confusion matrix
    frame_based_accs = []


    x = cPickle.load(open(rd_path, 'rb'))
    if scaler:
        x = scaler.transform(x)

    x = mat_2d_to_3d(x, n_concat, hop)
    seq=load_model(ld_md)
    p_y_preds = seq.predict(x,batch_size=None, verbose=1, steps=None)
    
    pred_ids = np.argmax(p_y_preds, axis=-1)
    
    
    pred_id = int(get_mode_value(pred_ids))
    print "This is",cfg.id_to_lb[pred_id]
    
    
    import collections
    a=collections.Counter(pred_ids)
    print sorted(a.iterkeys())
    b=a.keys()

    
    print "u are in ",cfg.id_to_lb[pred_id]," environment \n OR\n"
    for l in range(0,len(b)):   
        per=str(a[b[l]]*100/257)
        print "This can be ",cfg.id_to_lb[b[l]],per,"%"
    return cfg.id_to_lb[pred_id]



def get_mode_value(ary):
    return scipy.stats.mode(ary)[0]

def others(rd_pred,ld_md):
    n_concat =cfg.n_concat        # concatenate frames
    hop = cfg.hop            # step_len
    fold = cfg.fold            # can be 0, 1, 2, 3
    dev_feature = cfg.dev_mel
    eva_feature = cfg.eva_mel
    #print "came till here"
    '''scaler = pp_data.get_scaler(fe_fd=dev_feature, 
                                    csv_file=cfg.dev_tr[fold], 
                                    with_mean=True, 
                                    with_std=True)'''
    #joblib.dump(scaler, cfg.dp_sc)
    scaler=joblib.load(cfg.ld_sc)
    msg = pred(rd_pred,ld_md,scaler)
    return msg
    

if __name__ == '__main__':
    #
    n_concat =cfg.n_concat        # concatenate frames
    hop = cfg.hop            # step_len
    fold = cfg.fold            # can be 0, 1, 2, 3
    
    # your workspace
    dev_feature = cfg.dev_mel
    eva_feature = cfg.eva_mel
    if sys.argv[2] == "dump":
        scaler = pp_data.get_scaler(fe_fd=dev_feature, 
                                    csv_file=cfg.dev_tr[fold], 
                                    with_mean=True, 
                                    with_std=True)
        joblib.dump(scaler, cfg.dp_sc) 

    if sys.argv[2] == "load":
        #scaler=joblib.load(cfg.ld_sc) 
        """scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
 """
    
    if sys.argv[1] == "single_pred": 
        """scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
                                    """
        pred(cfg.rd_pred,cfg.ld_md,scaler)