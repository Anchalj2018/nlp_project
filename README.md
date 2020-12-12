# NLP Project: Sentiment Classification 

## Summary

The goal of the project is to be able to infer the sentiment ratings for hotel reviews.


### The Data
The input train and dev dataset consists of reviews, associated ids and sentimet ratings. The reviews are labeled with 5 sentiment ratings. test dataset has only reviews and ids. Following csv files are stored in `nlp_project/data` folder.

1. sentiment_dataset_train.csv
2. sentiment_dataset_dev.csv
3. sentiment_dataset_test.csv


Project Organization
------------

    ├── NLP_project             <- Main folder containing data preprocessing and modeling work
    │   ├── data                <- Original .csv files
    │   ├── code                <- jupyter notebooks and .py scripts for modeling and preprocessing tasks
    │   ├── Results             <- Results related to modeling and preprocessing
    │   │   ├── eval_report     <- Classification report and figures
    │   │   └── model_param     <- Model files and other intermediate results
    │   │   └── prediction      <- csv iles of predictions for dev and test datasets  
    │   ├── readme_images       <- Images for readme file
    │   ├── README.md           <- README for this project.

--------

## Prerequisites
The following packages need to be installed and must be running.
```
pandas
numpy
scikit-learn
nltk
nltk.download(stopwords)
tensorflow
keras
bs4
h5py
matplotlib
seaborn
```

## Execution Instructions

### 1. Clone this repository 
```
git clone https://github.com/Anchalj2018/nlp_project.git
```

### 2. To generate predictions using already trained  Lositic Regression-Multinomial and LSTM models

Run following command from root.

```
python nlp_project/code/inference.py <predicition_file.csv>  
```
- `<predicition_file.csv> ` is the name of file on which predictions are to be made. The file is stored in `nlp_projecr/data/` folder.

For example:
```
python nlp_project/code/inference.py sentiment_dataset_test.csv
```
It creates the following csv files:
- `nlp_project/results/prediction/logistic_test_prediction.csv`
- `nlp_project/results/prediction/lstm_test_prediction.csv`

Note: the script also generates following classification reports  for the models if the input prediction file has sentiment rating
- `nlp_project/results/eval_report/logistic_test_classific_report.csv`
- `nlp_project/results/eval_report/lstm_test_classific_report.csv`


### 3.  To train the Logistic Regression model and understand the results 

1. Navigate to jupyter notebook in `nlp_project/code/`  folder.
2. Execute `Baseline_model.ipynb` notebook to understand the results.

It creates the following files:
- `nlp_project/results/prediction/logistic_dev_prediction_final.csv`
- `nlp_project/results/eval_report/lg_train_classific_report_final.csv`
- `nlp_project/results/eval_report/lg_dev_classific_report_final.csv`
- `nlp_project/results/model_param/logistic_model_final.sav`


### 4.  To train the State of Art LSTM model and understand the results 

1. Navigate to jupyter notebook in `nlp_project/code/`  folder.
2. Execute `LSTM_selected_final.ipynb` notebook to understand the results.

It creates the following files:
- `nlp_project/results/results/model_param/tokenizer_data_final.pkl`
- `nlp_project/results/results/model_param/lstm_weights_best_final.h5`
- `nlp_project/results/prediction/lstm_dev_prediction_final.csv`
- `nlp_project/results/eval_report/lstm_train_classific_report_final.csv`
- `nlp_project/results/eval_report/lstm_dev_classific_report_final.csv`
- `nlp_project/results/model_param/lstm_model_final.h5`
- `nlp_project/results/results/eval_report/lstm_accuracy_final.png`

Note: you can also see `LSTM_initial_trial.ipynb` in same directory. it is the same model with different parameters.


## Results

### Baseline Model-Logistic Regression

Logistic Regression model with different parameters was investigated for accuracy on classification. Results are:-

![Alt text](readme_images/logistic_results.png?raw=true )

(results are compiled from `Baseline_model.ipynb` jupyter notebook)

* In above table, C is knowns as inverse of regularization strength.
* From above 3 results, logistic regression model with n gram range  (1,1),C=1  give more generalized results.  
* Model witn ngram range(1,1), C=5 captures more variance from the  train data  but it seems to bit overfitting as there is no improvement in the Dev accuracy result.



### State of Art  Model- LSTM model

LSTM model was trained for 5 epochs for two different variation of network parameters. 
In each case, during training of the model, weights for the best performing models were only saved. Results are:-


![Alt text](readme_images/lstm_results.png?raw=true )

(Result 1 is from `LSTM_initial_trial.ipynb` and result 2 is from `LSTM_selected_final.ipynb` jupyter notebook respectively.)

* We obtained similar results for both variations in LSTM model. Result 2 is slightly less overfitting.

LSTM selected model accuracy

![Alt text](results/eval_report/lstm_accuracy_final.png?raw=true )

LSTM  initial model accuracy

![Alt text](results/eval_report/lstm_accuracy_initial.png?raw=true )



The model can be further tuned with 
1. Training with more data.
2. Increasing number of epochs for traiing.
3. Variation in architecture.


### Comparison of LSTM and Logistic Regression models

We observe that both models performed approximately in similar fashion. Logistic Regression model scored slightly higher for Dev  dataset accuracy results. However, there is scope for improvement in LSTM model by training with more data and iterations.










