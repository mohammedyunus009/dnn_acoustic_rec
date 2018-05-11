import csv
import sys
import numpy as np
import scipy.stats
import cPickle
import os
import config as cfg



def one_hot_encoding(x, n_out):
    x = x.astype(int)   # force type to int
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))



def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def get_mode_value(ary):
    return scipy.stats.mode(ary)[0]




def get_scaler(fe_fd, csv_file, with_mean, with_std):
    
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)


    x_all = []
    for li in lis:
        try:
            [na, lb] = li[0].split('\t')
        except:
            na = li[0]

        na = na.split('/')[1][0:-4]
       
        path = fe_fd + '/' + na + '.f'
        x = cPickle.load(open(path, 'rb'))
        x_all.append(x)
     


    x_all = np.concatenate(x_all, axis=0)
    
    from sklearn import preprocessing#Simple and efficient tools for data mining and data analysis			
    scaler = preprocessing.StandardScaler(with_mean, with_std).fit(x_all)
    
    print("scalar complete")
    return scaler 

def mat_2d_to_3d(x, agg_num, hop):
    
    len_x, n_in = x.shape
    if (len_x < agg_num): #not in get_matrix_data
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
    
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)





def formated_data(fe_fd, csv_file, n_concat, hop, scaler):
    
    """Get training data and ground truth in matrix format. 
    
     
    """
        
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    print "gettting_matrix_format_data"      
    x3d_all = []
    y_all = []
    
    for li in lis:
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = fe_fd + '/' + na + '.f'
        x = cPickle.load(open(path, 'rb'))#(1291, 64)
        
        if scaler:
            x = scaler.transform(x) 
        
        x3d = mat_2d_to_3d(x, cfg.n_concat, cfg.hop)     # (n_blocks, n_concat, n_freq)
        
        
        x3d_all.append(x3d)
        y_all += [cfg.lb_to_id[lb]] * len(x3d) 
    
    y_all = np.array(y_all) 
    
    y_all = one_hot_encoding(y_all, len(cfg.labels)) # (n_samples, n_labels)
    
    x3d_all = np.concatenate(x3d_all)   # (n_samples, n_concat, n_freq)
    return x3d_all, y_all

###Recognize

def recognize(ld_md, te_feature, te_csv, n_concat, hop, scaler):
    
    from keras.layers import InputLayer, Flatten, Dense, Dropout
    from keras.models import Sequential
    from keras.models import model_from_json, load_model

    from keras.callbacks import ModelCheckpoint
    
    seq = load_model(ld_md)
    # Get test file names
    with open(te_csv, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)

    
    # Predict for each scene
    for li in lis:
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = te_feature + '/' + na + '.f'
        x = cPickle.load(open(path, 'rb'))
        print path
        if scaler:
            x = scaler.transform(x)

        x = mat_2d_to_3d(x, cfg.n_concat, cfg.hop)        # Predict

        p_y_preds = seq.predict(x,batch_size=None, verbose=1, steps=None)
        pred_ids = np.argmax(p_y_preds, axis=-1) #return the indices of the array in whicht the values is maximum
        pred_id = int(get_mode_value(pred_ids))
        gt_id = cfg.lb_to_id[lb]
        confuse_mat[gt_id, pred_id] += 1

    print confuse_mat