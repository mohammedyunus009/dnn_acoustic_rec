

#import cPickle
import numpy as np
np.random.seed(15)
import os
import sys
import csv

import config as cfg
import prepare_data as pp_data



####################################
from keras.layers import InputLayer, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint


def train(tr_fe_fd, tr_csv_file, te_fe_fd, te_csv_file, 
          n_concat, hop, scaler, out_md_fd):
    # Prepare data 
    tr_x, tr_y = pp_data.get_matrix_format_data(
                     fe_fd=tr_fe_fd, 
                     csv_file=tr_csv_file, 
                     n_concat=n_concat, hop=hop, scaler=scaler)
    print("part-1 training completed")
    
    te_x, te_y = pp_data.get_matrix_format_data(
                     fe_fd=te_fe_fd, 
                     csv_file=te_csv_file, 
                     n_concat=n_concat, hop=hop, scaler=scaler)
    
    n_freq = tr_x.shape[2]
    
    print("part-2 training completed")
    
    print tr_x.shape, tr_y.shape,"shapes"


    n_out = len(cfg.labels)
    seq = Sequential()
    seq.add(Dense(100 ,input_shape=(n_concat, n_freq)))

    seq.add(Flatten())

    seq.add(Dropout(0.2))
    seq.add(Dense(200, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(200, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(n_out, activation='softmax'))
    print "done"

    md = seq.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    seq.summary()
    
    filepath = "saved/weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                   save_best_only=False, mode='max',
                                   save_weights_only=False,period=2)
    callbacks_list = [checkpoint]

    seq.fit(tr_x, tr_y, validation_data=(te_x, te_y),
                 epochs=50, batch_size=500,
                 callbacks=callbacks_list, verbose=1)
    print("building model complete")
    

    #score = seq.evaluate(tr_x, tr_y, verbose=0)
    #print("%s: %.2f%%" % (seq.metrics_names[1], score[1]*100))
    ####
    #model_json = seq.to_json()
    #with open("model2.json", "w") as json_file:
     #   json_file.write(model_json)
  # serialize weights to HDF5
    #seq.save_weights("model2.h5")
    #print("Saved model to disk")

def dev_train():
    scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)
        #print "scaler",scaler
        
    train(tr_fe_fd=dev_fe_fd, 
              tr_csv_file=cfg.dev_tr_csv[fold], 
              te_fe_fd=dev_fe_fd, 
              te_csv_file=cfg.dev_te_csv[fold], 
              n_concat=n_concat, 
              hop=hop, 
              scaler=scaler, 
              out_md_fd=cfg.dev_md_fd)

def eva_recognize():
    scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_tr_csv[fold], 
                                    with_mean=True, 
                                    with_std=True)

    pp_data.recognize(md_path=cfg.dev_md_fd+'/md10_epochs.p', 
                  te_fe_fd=dev_fe_fd, 
                  te_csv_file=cfg.dev_te_csv[fold], 
                  n_concat=n_concat, 
                  hop=hop, 
                  scaler=scaler)
        
def eva_train():
    scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_meta_csv, 
                                    with_mean=True, 
                                    with_std=True)
    train(tr_fe_fd=dev_fe_fd, 
              tr_csv_file=cfg.dev_meta_csv, 
              te_fe_fd=eva_fd_fd, 
              te_csv_file=cfg.eva_meta_csv, 
              n_concat=n_concat, 
              hop=hop, 
              scaler=scaler, 
              out_md_fd=cfg.eva_md_fd)

def eva_recognize():
    scaler = pp_data.get_scaler(fe_fd=dev_fe_fd, 
                                    csv_file=cfg.dev_meta_csv, 
                                    with_mean=True, 
                                    with_std=True)
    pp_data.recognize(md_path=cfg.eva_md_fd+'/md10_epochs.p', 
                  te_fe_fd=eva_fd_fd, 
                  te_csv_file=cfg.eva_meta_csv, 
                  n_concat=n_concat, 
                  scaler=scaler, 
                  hop=hop)


if __name__ == '__main__':
    # hyper-params
    n_concat = 11        # concatenate frames
    hop = 5            # step_len
    fold = 0            # can be 0, 1, 2, 3
    
    
    # your workspace
    dev_fe_fd = cfg.dev_fe_logmel_fd
    eva_fd_fd = cfg.eva_fe_logmel_fd

    if sys.argv[1] == "--all":
        dev_train()
        dev_recognize()
        eva_train()
        eva_recognize()

    elif sys.argv[1] == "--dev_train":
        dev_train() 
    
    elif sys.argv[1] == "--dev_recognize":
        dev_recognize()
        
    elif sys.argv[1] == "--eva_train": 
        eva_train() 
              
    elif sys.argv[1] == "--eva_recognize":
        eva_recognize()
        
    else: 
        raise Exception("Incorrect argv!")

         
