# NLP Project: Sentiment Classification 

## Summary

The goal of the project is to be able to infer the sentiment ratings for hotel reviews.


### The Data
The input dataset consists of reviews, associated id's and sentimet ratings. The reviews are labeled with 5 sentiment ratings.


Project Organization
------------

    ├── NLP_project             <- Main folder containing data preprocessing and modeling work
    │   ├── data                <- Original .csv files
    │   ├── code                <- jupyter notebooks and .py scripts for modeling and preprocessing tasks
    │   ├── Results             <- Results related to modeling and preprocessing
    │   │   ├── eval_report     <- classification report and figures
    │   │   └── model_param     <- model files and other intermediate results
    │   │   └── prediction      <- csv iles of predictions for dev and test datasets  
    │   ├── readme_images       <- images for readme file
    │   ├── README.md           <-  README for this project.

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

## Execution instructions

### 1. Clone this repository 
```
git clone https://github.com/Anchalj2018/nlp_project.git
```

### 2. To generate predictions using pre-trained  lositic regressoion-multinomial and LSTM models

Run following command from root.

```
python nlp_project/code/inference.py <predicition_file.csv>  
```
- `<predicition_file.csv> ` is the name of file on which predictions are to be made. The file is stored in `nlp_projecr/data/ folder`.

For example:
```
python nlp_project/code/inference.py sentiment_dataset_test.csv
```
It creates the following csv files:
- `nlp_project/results/prediction/logistic_test_prediction.csv`
- `nlp_project/results/prediction/lstm_test_prediction.csv`


### 3.  To train the logistic regression model and understand the results 

1. Navigate to jupter notebook in `nlp_project/code/`  folder
2. execute `Baseline_model.ipynb` notebook to understand the results

It ouputs the following files:
- `nlp_project/results/prediction/logistic_dev_prediction_final.csv`
- `nlp_project/results/eval_report/lg_train_classific_report_final.csv`
- `nlp_project/results/eval_report/lg_dev_classific_report_final.csv`
- `nlp_project/results/model_param/logistic_model_final.sav`


### 3.  To train the LSTM model and understand the results 

1. Navigate to jupyter notebook in `nlp_project/code/`  folder
2. Execute `LSTM_selected_final.ipynb` notebook to understand the results

It outputs the following files:
- `nlp_project/results/results/model_param/tokenizer_data_final.pkl`
- `nlp_project/results/results/model_param/lstm_weights_best_final.h5`
- `nlp_project/results/prediction/lstm_dev_prediction_final.csv`
- `nlp_project/results/eval_report/lstm_train_classific_report_final.csv`
- `nlp_project/results/eval_report/lstm_dev_classific_report_final.csv`
- `nlp_project/results/model_param/lstm_model_final.h5`
- `nlp_project/results/results/eval_report/lstm_accuracy_final.png`

Note: you can also check `LSTM_initial_trial.ipynb`  which is same model with different parameters to understand the results


## Results

### Logistic Regression- multinomial

Logitic regression model with different parameters was investigated for accuracy on classification.Results are

![Alt text](readme_images/logistic_results.png?raw=true )

* From above 3 results, logistic regression model with n gram range  (1,1),C=1  give more generalized results.  
* Model witn ngram range(1,1), C=5 captures more variance from the  train data  but it seems to be bit overfitting as there is no improvement in the Dev accuracy result.



### LSTM model

LSTM model was trained for 5 epochs for two different variation of network parameters. 
In each case, during training of the model, weights for the best performing models were only saved.

Result

![Alt text](readme_images/lstm_results.png?raw=true )

Result 1 is from `LSTM_initial_trial.ipynb` and result 2 is from `LSTM_selected_final.ipynb` jupyter notebook respectively.
* WE obtained similar results for both variations in LSTM model. Result 2 is slightly less overfiiting.

LSTM selected_model accuarcy

![Alt text](results/eval_report/lstm_accuracy_final.png?raw=true )

LSTM  initial model accuarcy

![Alt text](results/eval_report/lstm_accuracy_initial.png?raw=true )



The model can be further tuned with 
1. more data
2. increasing number of epoch for traiing
3. variation in architectute


### Comparison of two models

We observe that both model performed approximately in similar fashion. Logistic regression model scored slighter higher for Dev  dataset accuracy results










