# Hate Crime Prediction based on Hate Speech

## Group member
* Shunwei Yuan (sy3001)
* Haohui Shen (hs4239)
* Keyu Wu (kw3103)

## Project Structure
```
Project
    datasets/: where the datasets are stored
    
    models/: where the models for hate speech detecting are stored
    
    preprocessed_data/: stores the preprocessed data in json and csv format
    
    predict.py: implementation for predicting the hate/non-hate label for a dataset
    
    preprocessing.py: implementation for preprocessing the hate speech dataset and training and testing the hate speech detector
    
    stopwords.py: stores the stop words as a Python list
    
    train.py: training code for hate speech detector
    
    hate_speech_plotting.py: plotting the proportion of hate tweets of retrieved tweets through twitter API
    
    newdata.py: implementation of retrieving tweets and then redict the hate/non-hate label for each tweet

    requirements.txt: dependent libraries
```


## How to run
Environment requirement: ```Python 3.7```

All dependent libraries are listed in ```requirements.txt```, please execute ```pip install -r requirements.txt``` to install them

To train the detector, simply run the ```train.py```.

To retrieve the data from twitter API, simply run the ```newdata.py```.

To get the final prediction, simply run the ```crime_data_analyze.ipynb```.

Please note that all the code takes a lot of time.