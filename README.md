# Introduction (machine learning ,sound recognition)

This is a keras implementation project
This project is used to recognize environmental sounds ,with a high accuracy of 80% percent ,it has a deep neural network (LSTM can be implemented),can be used to train on the extracted features of audio files in 
[https://github.com/mohammedyunus009/dev_datasets](https://github.com/mohammedyunus009/dev_datasets) and [https://github.com/mohammedyunus009/datasets](https://github.com/mohammedyunus009/datasets).


It has the capability to recognize on 15 diffrent classes of sounds such as :

'bus'  'cafe/restaurant'  'car'  'city_center'  'forest_path'  
'grocery_store'  'home'  'beach'  'library'  'metro_station'  
'office'  'residential_area'  'train'  'tram'  'park'

vtu under graduate project (visvervaraya university of technology)


# Requirements

This library runs with keras. 

`pip install -r requirements.txt`

OR for conda and linux users run 
`sh setup.sh`

# Quickstart

STEP 1
configure the config file in `src` folder

STEP 2 

*`python calculate_logmel.py` to extract features and pickle in memory


STEP 3
* `python kera_model.py --dev_train`  to run in development mode (train on development dataset)
* `python kera_model.py --eva_train`  to run in evaluation mode (train on evaluation dataset)
Reconfigure the configuration file in src

* `python kera_model.py --dev_recognize`  used to calculate the accuracy of the model in development mode

* `python kera_model.py --eva_recognize`  used to calculate the accuracy of the model in evaluation mode

STEP 4
* `python session.py` used to put in production and test new files

Contact me for more information