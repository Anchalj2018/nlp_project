#!/usr/bin/env python
# coding: utf-8


import sys
import pre_process_text as pt
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import numpy as np
import pickle
import h5py
from keras import models
from keras.preprocessing.sequence import pad_sequences



"""
Purpose: This script reads in data as csv file from command line to make predictions and saves prediction results as csv files
         It generates 2 sets of predictions using 2 pre-fitted models, logistic Regression and LSTM model.
         
Outputs: script outputs 2 csv files , 1 for each model predictions. 
         csv file consists of 2 columns,one for id and  second for predicted rating.
         files are stored in  nlp_project/results/prediction folder.
         
Usage:   python nlp_project/code/predict_data.py <predicition_file.csv>   ( relative to root  folder)

<predicition.csv>        The input file for which prediction to be made e.g. sentiment_dataset_test.csv

"""



#########################################    EVALUATION of MODEL     ##############################################

# eval_model function is used  to generate classification report for model(if we have the label of rating in input data )
def eval_model(df, model_name):
    
       
    """
        Purpose: this function generates classification report on the original and predicted results     

        Parmaters: df: input dataframe
                  model_name : text string for the model name

        output: csv file for test classification is saved tonlp_project/results/eval_report  

    """
        
    labels=['1','2','3','4','5']
    #filepath to save report
    filepath="nlp_project/results/eval_report/"+model_name+"_test_classific_report.csv"
    
    #print classification report
    print( "\n\nTest classification Report for "+model_name,"\n\n",classification_report(df['rating'], df['pred_rating'],target_names=labels))

    #save the report
    eval_report=classification_report(df['rating'], df['pred_rating'],target_names=labels,output_dict=True)
    eval_t=pd.DataFrame(eval_report).transpose().reset_index()
    eval_t.rename(columns={'index':'label'},inplace=True)
    
    #save file
    eval_t.to_csv(filepath,index=False)
    

################################################  MAIN FUNCTION  #############################################


def main(prediction_file):
    

    # path for the input file relative to  the root
    data_path="nlp_project/data/"
    
    #read the file
    test=pd.read_csv(data_path+prediction_file)   
    
        
    # clean the text
    test['clean_review']=test['review'].apply(lambda x: pt.clean_text(x))

    #make copy of datasets since processing is different for two models
    test_lg=test.copy()
    test_lstm=test.copy()


    
#*****************************************  Logistic Regression MODEL ********************************************************

    #preprocess the review text to TF-IDF vectors
    test_lg['clean_review']=pt.pre_process(test_lg['clean_review'])

    #load the model
    lg_model= pickle.load(open('nlp_project/results/model_param/logistic_model_final.sav', 'rb'))

    #predict the sentiment rating
    test_lg['pred_rating']=lg_model.predict(test_lg['clean_review'])

    #save the predictions
    print("\nPredictions for logistic regression model are saved to nlp_project/results/prediction/logistic_test_prediction.csv\n")
    test_lg[['id','pred_rating']].to_csv("nlp_project/results/prediction/logistic_test_prediction.csv",index=False)


    

#******************************************** LSTM MODEL *******************************************************************

    #load the parameters for text pre-processing
    with open("nlp_project/results/model_param/tokenizer_data_final.pkl", 'rb') as f:
        data = pickle.load(f)
        tokenizer = data['tokenizer']
        max_words = data['num_words']
        max_seq_len = data['maxlen']

    # tokenize the reviews and create fixed length sequences
    lstm_X=tokenizer.texts_to_sequences(test_lstm['clean_review'].values)
    lstm_X=pad_sequences(lstm_X, maxlen=max_seq_len)

    #load the model
    lstm_model = models.load_model("nlp_project/results/model_param/lstm_model_final.h5")
    
    #Make predictions
    lstm_pred=lstm_model.predict(lstm_X)

    # get the sentiment rating label from the predictions
    labels=['1','2','3','4','5']
    test_lstm['pred_rating']=[labels[l] for l in  np.argmax(lstm_pred,axis=1)]

    #save the predictions
    print("\nPredictions for LSTM model are saved to nlp_project/results/prediction/lstm_test_prediction.csv\n")
    test_lstm[['id','pred_rating']].to_csv("nlp_project/results/prediction/lstm_test_prediction.csv",index=False)
    


#******************************************EVALUATE THE MODEL IF RATING IS AVAILABLE **************************************
    
    ## if the input prediction file has sentiment rating available, then we can  generate the classification report
    if 'rating' in test:
        
        if(test['rating'].dtypes==int):
            test_lg['rating']=test_lg['rating'].astype(str)
            test_lstm['rating']=test_lstm['rating'].astype(str)
        
        #classification report for logistic regression model
        model_name="logistic"
        eval_model(test_lg,model_name)
        
        #classification report for LSTM model
        model_name="lstm"
        eval_model(test_lstm,model_name)
        
    
    
if __name__ == "__main__":
        
    predictions_file=sys.argv[1]     
    main(predictions_file)    

