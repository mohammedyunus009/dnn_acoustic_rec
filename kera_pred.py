from sklearn.externals import joblib


import numpy as np
from scipy import signal
import scipy.stats
import cPickle
import os
from scipy import signal
import librosa #package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems

import config1 as cfg
import sys
import prepare_data1 as pp_data

from keras.layers import InputLayer, Flatten, Dense, Dropout
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

#sys.path.append(cfg.hat_root)
#from hat.preprocessing import mat_2d_to_3d, sparse_to_categorical
#from hat import serializations

 
# sparse label to categorical label
# x: ndarray
def sparse_to_categorical(x, n_out):
    x = x.astype(int)   # force type to int
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    
    Args:
      x: 2darray, (n_time, n_in)
      agg_num: int, number of frames to concatenate. 
      hop: int, number of hop frames. 
      
    Returns:
      3darray, (n_blocks, agg_num, n_in)
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def pred(rd_path,wr_path,ld_md,scaler):
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
 
    #cfg.n_freq = 64
    n_out = len(cfg.labels)
    seq = Sequential()
    seq.add(Dense(64 ,input_shape=(cfg.n_concat, cfg.n_freq)))

    seq.add(Flatten())

    seq.add(Dropout(0.2))
    seq.add(Dense(300, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(300, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(n_out, activation='softmax'))
    print "done"


    seq.load_weights(ld_md)

    seq.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



           # Predict

    #print x,"x"
    p_y_preds = seq.predict(x,batch_size=100, verbose=1, steps=None)
    #np.set_printoptions(threshold=np.nan)
    #print x.shape
    
    pred_ids = np.argmax(p_y_preds, axis=-1)
    #prints the id_to_lb    # (n_block,)
    
    pred_id = int(get_mode_value(pred_ids))
    print "This is",cfg.id_to_lb[pred_id]
    return cfg.id_to_lb[pred_id]
    #temp=cfg.id_to_lb[pred_id]
#print pred_ids


#def find_key(input_dict, value):
 #   return next((k for k, v in input_dict.items() if v == value), None)
    import collections
    a=collections.Counter(pred_ids)
    print sorted(a.iterkeys())
    b=a.keys()
    #print b,len(b)
    
    print "u are in ",cfg.id_to_lb[pred_id]," environment \n OR\n"
    for l in range(0,len(b)):   
        per=str(a[b[l]]*100/257)
        print "This can be ",cfg.id_to_lb[b[l]],per,"%"


    with open(wr_path,'w') as f:
        f.write(str("u are in "+cfg.id_to_lb[pred_id]+" environment \n OR\n"))
    for l in range(0,len(b)):   
        per=str(a[b[l]]*100/257)
        with open(wr_path,'a') as f:
            f.write(str("This can be "+cfg.id_to_lb[b[l]]+per+" %\n"))

    #for k in range(0,len())

""" with open(wr_path,'w') as f:
        #temp = "u are in "+cfg.id_to_lb[pred_id]+" environment \n OR\n","This can be ",cfg.id_to_lb[b[l]],a[b[l]]*100/257,"%"
        print temp
        f.write(temp)
"""
def get_mode_value(ary):
    return scipy.stats.mode(ary)[0]

def others(rd_pred,wr_pred,ld_md):
    n_concat =cfg.n_concat        # concatenate frames
    hop = cfg.hop            # step_len
    fold = cfg.fold            # can be 0, 1, 2, 3
    
    # your workspace
    dev_fe_fd = cfg.dev_fe_logmel_fd
    eva_fd_fd = cfg.eva_fe_logmel_fd
    #print "came till here"
    """scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)"""
    #joblib.dump(scaler, cfg.dp_sc)
    scaler=joblib.load(cfg.ld_sc)
    msg = pred(rd_pred,wr_pred,ld_md,scaler)
    return msg
    

if __name__ == '__main__':
    # hyper-params
    n_concat =cfg.n_concat        # concatenate frames
    hop = cfg.hop            # step_len
    fold = cfg.fold            # can be 0, 1, 2, 3
    
    # your workspace
    dev_fe_fd = cfg.dev_fe_logmel_fd
    eva_fd_fd = cfg.eva_fe_logmel_fd
    if sys.argv[2] == "dump":
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
        joblib.dump(scaler, cfg.dp_sc) 

    if sys.argv[2] == "load":
        scaler=joblib.load(cfg.ld_sc) 
        """scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
 """
    #path = sys.argv[2]
    if sys.argv[1] == "single_pred": 
        """scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
        """#print "scaler",scaler
        pred(cfg.rd_pred,cfg.wr_pred,cfg.ld_md,scaler)
        
        #pred("/home/ruksana/test/DCASE2016_Task1/src/DCASE2016_task1_scrap/features/dev/logmel/a107_210_240.f",scaler)
        """with open(csv_file, 'rb') as f:
            reader = csv.reader(f)
            lis = list(reader)
        for l in lis
        a(,scaler)
"""
