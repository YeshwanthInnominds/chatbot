# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:53:05 2018

@author: Yeshwanth

Chat bot

"""

import json

import nltk

import numpy as np

from scipy.spatial.distance import euclidean

import pandas as pd

import re

from keras.models import Sequential

from keras.layers import LSTM



"""
Processing the data:
    
"""

conversations = pd.read_csv("movie_lines.csv")


cor = conversations['conversation']

x = []

y = []

for i in range(1000):
    x.append(str( cor[i]))
    y.append(str(cor[i+1]))
            
tok_x = []
tok_y = []

for i in range(len(x)):
    tok_x.append( nltk.word_tokenize(x[i].lower()))
    tok_y.append( nltk.word_tokenize(y[i].lower()))
    
    
sentend = np.ones((1,) , dtype = np.float32)

## building vocab
vocab = set()

for sent in tok_x:
    for word in sent:
        vocab.add(word)

for sent in tok_y:
    for word in sent:
        vocab.add(word)

list_vocab = list(vocab)

voc = {}
for word in vocab:
    voc[word] = round( list_vocab.index(word) /236,5)


vec_x = []
for sent in tok_x:
    sent_vec = [voc[w] for w in sent if w in list_vocab]
    vec_x.append(sent_vec)

vec_y = []
for sent in tok_y:
    sent_vec = [voc[w] for w in sent if w in list_vocab]
    vec_y.append(sent_vec)


for tok_sent in vec_x:
    tok_sent[14:] = []
    tok_sent.append(sentend[0])
    

for tok_sent in vec_x:
    if len(tok_sent) < 15:
        for i in range( 15 - len(tok_sent)):
            tok_sent.append(sentend[0])

for tok_sent in vec_y:
    tok_sent[14:] = []
    tok_sent.append(sentend[0])
    

for tok_sent in vec_y:
    if len(tok_sent) < 15:
        for i in range( 15 - len(tok_sent)):
            tok_sent.append(sentend[0])

###############################################################################


vec_x = np.expand_dims(vec_x, axis = 0)
vec_y = np.expand_dims(vec_y,axis = 0)

model = Sequential()


model.add( LSTM(15,input_shape=(None,15),
                return_sequences = True,
                init = 'glorot_normal',
                inner_init= 'glorot_normal',
                activation = 'sigmoid'
                ))
model.add( LSTM(15,input_shape=(None,15),
                return_sequences = True,
                init = 'glorot_normal',
                inner_init= 'glorot_normal',
                activation = 'sigmoid'
                ))


model.compile( loss= 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(vec_x, vec_y,epochs = 100)





while(True):
    x = input("Enter you message ")
    sent = nltk.word_tokenize(x.lower())
    sentvec = [voc[w] for w in sent if w in list_vocab]
    sentvec[14:]  = []
    sentvec.append(sentend)
    if len(sentvec) < 15:
        for i in range( 15 - len(sentvec)):
            sentvec.append(sentend)
    sentvec = np.array(sentvec)
    sentvec = np.expand_dims(sentvec, axis = 0)
    sentvec = np.expand_dims(sentvec, axis = 0)
    pred = model.predict(sentvec)
    
    out  = []
    for word in pred[0][0]: 
        my = []
        for val in voc.values():
            temp = euclidean( val , word)
            my.append(temp)
    
        out.append(list(voc.keys())[list(voc.values()).index(list(voc.values())[my.index(min(my))])])
    for word in out:
        print(word, end=" ")
    





















