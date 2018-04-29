

 
 

# development config
dev_wav_fd = "/TUT-acoustic-scenes-2016-development/audio"

fd1="/home/ubuntu/test/data/" #FOR DEV
fd2="/home/ubuntu/test/data/"	#for eva
#mkdir test
dev_csv_fd = fd1 + "TUT-acoustic-scenes-2016-development/evaluation_setup"

dev_tr_csv = [dev_csv_fd+"/fold1_train.txt", dev_csv_fd+"/fold2_train.txt", 
               dev_csv_fd+"/fold3_train.txt", dev_csv_fd+"/fold4_train.txt"]

dev_te_csv = [dev_csv_fd+"/fold1_evaluate.txt", dev_csv_fd+"/fold2_evaluate.txt", 
               dev_csv_fd+"/fold3_evaluate.txt", dev_csv_fd+"/fold4_evaluate.txt"]
dev_meta_csv = fd1 + "TUT-acoustic-scenes-2016-development/meta.txt"

# evaluation config
eva_wav_fd = "/test/DCASE2016_Task1/data/TUT-acoustic-scenes-2016-evaluation/audio"
eva_meta_csv = fd2 + "TUT-acoustic-scenes-2016-evaluation/meta.txt"

# your workspace
scrap_fd = "/home/ubuntu/"	#"/home/ruksana/test/DCASE2016_Task1/src/DCASE2016_task1_scrap"
fe1 = scrap_fd + "dev_datasets/features"
fe2 = scrap_fd + "datasets/features"
dev_fe_logmel_fd = fe1 + "/dev/logmel"
eva_fe_logmel_fd = fe2 + "/eva/logmel"
md_fd = scrap_fd + "/models"
dev_md_fd = scrap_fd + "/dev"
eva_md_fd = scrap_fd + "/eva"

# 1 of 15 acoustic label
labels = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 
           'grocery_store', 'home', 'beach', 'library', 'metro_station', 
           'office', 'residential_area', 'train', 'tram', 'park']
            
lb_to_id = {lb:id for id, lb in enumerate(labels)}
id_to_lb = {id:lb for id, lb in enumerate(labels)}


fs = 44100.
n_fft = 1024.


n_concat=11
n_freq=64

fold = 0
hop = 5

ld_md='final_weights-improvement-148-0.998.hdf5'
dp_md='/home/ubuntu/trained_data1/output_eva/final_weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5'
#mkdir outputs

rd_pred = ''
wr_pred = ''

dp_sc = 'filename.plk'
ld_sc = 'filename.plk'