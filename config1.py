#development config
dev_wav = "/home/ubuntu/TUT-acoustic-scenes-2016-development/audio"

fd1="/home/ubuntu/test/data/" #FOR DEV
fd2="/home/ubuntu/test/data/"	#for eva
#mkdir test
dev_add = fd1 + "TUT-acoustic-scenes-2016-development/evaluation_setup"

dev_tr = [dev_add+"/fold1_train.txt", dev_add+"/fold2_train.txt", 
               dev_add+"/fold3_train.txt", dev_add+"/fold4_train.txt"]

dev_te = [dev_add+"/fold1_evaluate.txt", dev_add+"/fold2_evaluate.txt", 
               dev_add+"/fold3_evaluate.txt", dev_add+"/fold4_evaluate.txt"]
dev_meta = fd1 + "TUT-acoustic-scenes-2016-development/meta.txt"

# evaluation config
eva_wav = "/home/ubuntu/TUT-acoustic-scenes-2016-evaluation/audio"
eva_meta = fd2 + "TUT-acoustic-scenes-2016-evaluation/meta.txt"

# your workspace
workspace = "/home/ubuntu/"	
fe1 = workspace + "dev_datasets/features"
fe2 = workspace + "datasets/features"
dev_mel = fe1 + "/dev/logmel"
eva_mel = fe2 + "/eva/logmel"
md_fd = workspace + "/models"
#dev_md = workspace + "/dev"
#eva_md = workspace + "/eva"

# 1 of 15 acoustic label
labels = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 
           'grocery_store', 'home', 'beach', 'library', 'metro_station', 
           'office', 'residential_area', 'train', 'tram', 'park']

j=0
id_to_lb={}
for i in labels:
	id_to_lb.update({j:i})
	j=j+1

j=0
lb_to_id={}
for i in labels:
	lb_to_id.update({i:j})
	j=j+1



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