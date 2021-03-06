import csv
import sys
import numpy as np
import scipy.stats
import cPickle
import os
import config as cfg
from keras.models import load_model
def get_scaler(fe_fd, csv_file, with_mean, with_std):
    """Calculate scaler from data in csv_file. 
    
    Args:
      fe_fd: string. Feature folder. 
      csv_file: string. Path of csv file. 
      with_mean: bool. 
      with_std: bool. 
      
    Returns:
      scaler object. 
    """
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
    
    from sklearn import preprocessing #Simple and efficient tools for data mining and data analysis			
    scaler = preprocessing.StandardScaler(with_mean, with_std).fit(x_all)
    
    print "scalar complete"
    return scaler 





def formated_data(fe_fd, csv_file, n_concat, hop, scaler):
    """Get training data and ground truth in matrix format. 
    
    Args:
      fe_fd: string. Feature folder. 
      csv_file: string. Path of csv file. 
      n_concat: integar. Number of frames to concatenate. 
      hop: integar. Number of hop frames. 
      scaler: None | object. 
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
    """Recognize and get statistics. 
    
    Args:
      ld_md: string. Path of model. 
      te_feature: string. Folder path containing testing features. 
      te_csv: string. Path of test csv file. 
      n_concat: integar. Number of frames to concatenate. 
      hop: integar. Number of frames to hop. 
      scaler: None | scaler object. 
    """

    # Load model
    seq=load_model(ld_md)

    # Recognize and get statistics
    n_labels = len(cfg.labels)
    confuse_mat = np.zeros((n_labels, n_labels))      # confusion matrix
    frame_based_accs = []
    
    # Get test file names
    with open(te_csv, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    # Predict for each scene
    for li in lis:
        # Load data
        [na, lb] = li[0].split('\t')
        na = na.split('/')[1][0:-4]
        path = te_feature + '/' + na + '.f'
        x = cPickle.load(open(path, 'rb'))
        if scaler:
            x = scaler.transform(x)
        x = mat_2d_to_3d(x, n_concat, hop)
    
        # Predict
        p_y_preds = seq.predict(x,batch_size=None, verbose=1, steps=None)    # (n_block,label)
        pred_ids = np.argmax(p_y_preds, axis=-1)     # (n_block,)
        pred_id = int(get_mode_value(pred_ids))
        gt_id = cfg.lb_to_id[lb]
        
        # Statistics
        confuse_mat[gt_id, pred_id] += 1            
        n_correct_frames = list(pred_ids).count(gt_id)
        frame_based_accs += [float(n_correct_frames) / len(pred_ids)]
            
    clip_based_acc = np.sum(np.diag(np.diag(confuse_mat))) / np.sum(confuse_mat)
    frame_based_acc = np.mean(frame_based_accs)
    
    print 'event_acc:', clip_based_acc
    print 'frame_acc:', frame_based_acc
    print confuse_mat


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

