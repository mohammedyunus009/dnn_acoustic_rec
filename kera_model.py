
import cPickle
import numpy as np
np.random.seed(15)
import os
import sys
import csv

import config1 as cfg
import prepare_data1 as pp_data
from sklearn.externals import joblib



####################################
from keras.layers import InputLayer, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

a = iter(list(range(200)))    

def train(tr_feature, tr_csv, tr_feature, te_csv, 
          n_concat, hop, scaler, out_md,fold):
    # Prepare data 
    tr_x, tr_y = pp_data.formated_data(
                     fe_fd=tr_feature, 
                     csv_file=tr_csv, 
                     n_concat=n_concat, hop=hop, scaler=scaler)
    print("part-1 training completed")
    
    te_x, te_y = pp_data.formated_data(
                     fe_fd=tr_feature,
                     csv_file=te_csv, 
                     n_concat=n_concat, hop=hop, scaler=scaler)
    
    
    print("part-2 training completed")
    
    print tr_y.shape,"shapes"
    print tr_x.shape
    #sys.exit()
    n_freq = tr_x.shape[2]
    n_out = len(cfg.labels)
    seq = Sequential()
    seq.add(Dense(64 ,input_shape=(n_concat, n_freq)))

    seq.add(Flatten())

    seq.add(Dropout(0.2))
    seq.add(Dense(300, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(300, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(n_out, activation='softmax'))
    print "done"

    seq.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    seq.summary()
    #sys.exit()
    
    checkpoint = ModelCheckpoint(out_md+str(next(a)), monitor='val_acc', verbose=1,
                                   save_best_only=False, mode='max',
                                   save_weights_only=False,period=2)
    callbacks_list = [checkpoint]

    seq.fit(tr_x, tr_y, validation_data=(te_x, te_y),
                 epochs=150, batch_size=None,
                 callbacks=callbacks_list, verbose=1)
    print("saving model complete")
    
def dev_train():
        
    train(tr_feature=dev_fe_fd, 
              tr_csv=cfg.dev_tr_csv[fold], 
              tr_feature=dev_fe_fd, 
              te_csv=cfg.dev_te_csv[fold], 
              n_concat=n_concat, 
              hop=hop, 
              scaler=scaler, 
              out_md=cfg.dp_md,
              fold=0)

def dev_recognize():
    
    pp_data.recognize(cfg.ld_md, 
                  tr_feature=dev_fe_fd, 
                  te_csv=cfg.dev_te_csv[fold], 
                  n_concat=n_concat, 
                  hop=hop, 
                  scaler=scaler)
        
def eva_train():
    
    train(tr_feature=dev_fe_fd, 
              tr_csv=cfg.dev_meta_csv, 
              tr_feature=eva_fd_fd, 
              te_csv=cfg.eva_meta_csv, 
              n_concat=n_concat, 
              hop=hop, 
              scaler=scaler, 
              out_md=cfg.dp_md,
              fold=0)

def eva_recognize():
    pp_data.recognize(cfg.ld_md, 
                  tr_feature=eva_fd_fd, 
                  te_csv=cfg.eva_meta_csv, 
                  n_concat=n_concat, 
                  scaler=scaler, 
                  hop=hop)


if __name__ == '__main__':
    # hyper-params
    n_concat=cfg.n_concat        # concatenate frames
    hop=cfg.hop      # step_len
    fold=0            # can be 0, 1, 2, 3

    # your workspace
    dev_fe_fd = cfg.dev_fe_logmel_fd
    eva_fd_fd = cfg.eva_fe_logmel_fd

    if sys.argv[1] == "--all":
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
    
        dev_train()
        #dev_recognize()
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_meta_csv, 
                                    with_mean=True, 
                                    with_std=True)
    
        eva_train()
        #eva_recognize()

    elif sys.argv[1] == "--dev_train":
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
    
        dev_train() 
    
    elif sys.argv[1] == "--dev_recognize":
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
        scaler = joblib.load(cfg.ld_sc)
        dev_recognize()
        
    elif sys.argv[1] == "--eva_train": 
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_meta_csv, 
                                    with_mean=True, 
                                    with_std=True)
        #joblib.dump(scaler, cfg.dp_sc)
        scaler = joblib.load(cfg.ld_sc)
        eva_train()                                             
              
    elif sys.argv[1] == "--eva_recognize":
        scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_meta_csv, 
                                    with_mean=True, 
                                    with_std=True)
        #scaler = joblib.load(cfg.ld_sc)
        joblib.dump(scaler, cfg.dp_sc)
        eva_recognize()
        
    else: 
        raise Exception("Incorrect argv!")

         
    
    
