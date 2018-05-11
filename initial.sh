cd /home/ruksana/test/DCASE2016_Task1/dnn_acoustic_rec
source ~/test/DCASE2016_Task1/venv/bin/activate

sudo apt update
sudo apt upgrade
sudo apt dist-upgrade

sudo apt install htop
sudo apt install tree
sudo apt install python
sudo apt install python-dev
sudo apt install python-virtualenv
sudo apt install python-pip

sudo apt install tree
sudo apt install htop

virtualenv kera
source kera/bin/activate
pip install tensorflow
pip install keras
pip install sklearn

pip install numpy
pip install librosa
pip install scipy


pip install matplotlib
pip install h5py
pip install csv

mv output trained_data1
mv output1 trained_data1
cp ~/test/1.txt ~/trained_data1

git add -A
git commit -m "initial"
git push

##################

pip install --upgrade pip

source kera1/bin/activate

mkdir /home/ubuntu/trained_data1/output_eva # in trained_data1



mv data data1 #mv data data2
mkdir data #chech posibilities
cd /home/ubuntu/data

python kera_model.py --eva_train | tee -a eva.txt


git add -A
git commit -m "eva_commit"
git push
mohammedyunus009
# get back the stored scaler_file(evaluation)
#get back the stored scaler_file(development)




#pip upgrde

mkdir flask_test
cd flask_test

virtualenv flask
source flask/bin/activate

pip install numpy
pip install tensorflow
pip install keras
pip install sklearn

pip install librosa

pip install scipy


pip install matplotlib
pip install h5py
pip install csv




ssh -i "yunus1.pem" ubuntu@13.232.64.181

mkdir /home/ubuntu/results
cd /home/ubuntu/results/2
source ~/kera1/bin/activate

final_weights-improvement-40-0.692.hdf50

/home/ubuntu/trained_data1/output_dev_LSTM

mkdir /home/ubuntu/trained_data1/output_dev_LSTM
mkdir /home/ubuntu/trained_data1/output_eva_250
mkdir /home/ubuntu/trained_data1/output_eva_215

python kera_model.py --dev_train |tee -a dev_LSTM.txt
python kera_model.py --eva_train |tee -a eva_250.txt
python kera_model.py --eva_train |tee -a eva_215_temp.txt

cp /home/ubuntu/trained_data1/output_dev_LSTM /home/ubuntu/1
cp /home/ubuntu/trained_data1/output_eva_250 /home/ubuntu/2
cp /home/ubuntu/trained_data1/output_eva_215 /home/ubuntu/3
 
python kera_model.py --dev_recognize |tee -a dev_LSTM_rec.txt
python kera_model.py --eva_recognize |tee -a eva_250_rec.txt
python kera_model.py --eva_recognize |tee -a eva_215_rec.txt


/home/ubuntu/trained_data1/output_eva/final_weights-improvement-98-0.702.hdf50

