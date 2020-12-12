#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import re
import string
import nltk
from bs4 import BeautifulSoup



replace_symbols = re.compile('[/(){}\[\]\|@,;]')    
punc_re = re.compile('[%s]' % re.escape(string.punctuation))

def clean_text(text):
    
    """
    Purpose: This function performs cleaning of text from undesired symbols
    
    parameters: text string
    Returns:  text string
    """
        
        
    text=BeautifulSoup(text,"lxml").text
    text = text.lower()
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    text = replace_symbols.sub(' ', text)                   # replace replace_symbols by space in text
    text = re.sub(r"[-()\"#/@;:{}`+=~|.!?,']", "", text)      #Replacing special character with none
    text = re.sub(r'[0-9]+', '', text)                        #Replacing numbers with none
    text=" ".join(text.translate(str.maketrans('', '', string.punctuation)) for text in text.split() if text.isalpha())
    text=punc_re.sub(' ', text)  
    
    return(text)



stop = set(nltk.corpus.stopwords.words('english'))      # stopwords
stemmer_func = nltk.stem.snowball.SnowballStemmer("english").stem    #stemmer function

def pre_process(text): 
    
    """
        Purpose: This function removes stopwords and performs stemming on text data

        parameters: input dataframe
        Returns: processed dataframe
    """
    
    
    #Removing stop words  
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop)) 
    
    # stemming       
    text = text.apply(lambda x: " ".join(stemmer_func(word) for word in x.split())) 
    return(text)

