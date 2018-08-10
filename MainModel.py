#!/usr/bin/python
import os
import sys
import random
from keras.models import Sequential, Model
from keras.layers import Dense, AveragePooling1D, Embedding, Input, LSTM, TimeDistributed, Bidirectional, concatenate, Conv1D, Flatten, Dropout, MaxPooling1D
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Progbar
from keras.initializers import RandomUniform
from AddedFunctions import ExtractFeatures, text2VecConversion, mappingTheCharacters

##read in the dataset

TrD_sentences, TrD_tags, TrD_Characs, TrD_Ylabel = ExtractFeatures("/home/deepthi/MainData/Huddl/conll2003/train.txt")
TstD_sentences, TstD_tags, TstD_Characs, TstD_Ylabel = ExtractFeatures("/home/deepthi/MainData/Huddl/conll2003/train.txt")
ValD_sentences, ValD_tags, ValD_Characs, ValD_Ylabel = ExtractFeatures("/home/deepthi/MainData/Huddl/conll2003/train.txt")


label2Idx = {}
for label in TrD_tags:
    label2Idx[label] = len(label2Idx)

TrD_charIndices, char2Idx = mappingTheCharacters(TrD_Characs)
TstD_charIndices, char2Idx = mappingTheCharacters(TstD_Characs)
ValD_charIndices, char2Idx = mappingTheCharacters(ValD_Characs)


### :: Read in glove dictionary and associate our words to the words from the dictionary::
TrD_wordIdx, Glov_wordEmbd = text2VecConversion(TrD_sentences)
TstD_wordIdx, Glov_wordEmbd = text2VecConversion(TstD_sentences)
ValD_wordIdx, Glov_wordEmbd = text2VecConversion(ValD_sentences)

print('step3')


## Build the model
##sequence_input = Input(shape=(None,),dtype='int32',name='sequence_input')
##embedded_sequences = Embedding(input_dim=Glov_wordEmbd.shape[0], output_dim=Glov_wordEmbd.shape[1],  weights=[Glov_wordEmbd], trainable=False)(sequence_input)
##
##x = Conv1D(128, 5, activation='relu')(embedded_sequences)
##x = MaxPooling1D(5)(x)
##x = Conv1D(128, 5, activation='relu')(x)
##x = MaxPooling1D(5)(x)
##x = Conv1D(128, 5, activation='relu')(x)
##x = MaxPooling1D(35)(x)
##x = Flatten()(x)
##x = Dense(128, activation='relu')(x)
##preds = Dense(len(labels_index), activation='softmax')(x)
##
##
### Fit the model
##model = Model(sequence_input, preds)
##model.compile(loss='categorical_crossentropy',
##              optimizer='rmsprop',
##              metrics=['acc'])



words_input = Input(shape=(None,),dtype='int32',name='words_input')
words = Embedding(input_dim=Glov_wordEmbd.shape[0], output_dim=Glov_wordEmbd.shape[1],  weights=[Glov_wordEmbd], trainable=False)(words_input)
character_input=Input(shape=(None,52,),name='char_input')
embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
dropout= Dropout(0.5)(embed_char_out)

conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words, char])
output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)

print(output.shape)
model = Model(inputs=[words_input, character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()
print(model.summary())


##Xtrain = np.asarray(np.expand_dims(TrD_charIndices,-1))
##charFinalIp = np.asarray(np.expand_dims(TrD_charIndices,1))
##Ytrain = np.asarray(np.expand_dims(TrD_Ylabel,-1))



model.fit([TrD_wordIdx,TrD_charIndices], TrD_Ylabel, validation_data=(ValD_wordIdx, ValD_Ylabel),
          epochs=2, batch_size=128)

##model.train_on_batch([Xtrain, charFinalIp], Ytrain)

######
####
##start = 0
##last = 18511
##a = Progbar(18511)
####epochs = 50
##for itr in range(0, 10):
####    batch = range(start,last)
##    Xtrain = np.asarray(np.expand_dims(wordIndices[start:last],-1))
##    charFinalIp = np.asarray(np.expand_dims(charIndices[start:last,:],1))
##    Ytrain = np.asarray(np.expand_dims(Ylabel[start:last,:],-1))
##    model.train_on_batch([Xtrain, charFinalIp], Ytrain)
##    a.update(itr)
##    start = last+1
##    last = start + 18510
##    print(start, last)
##
##
##print("step4")


##model.train_on_batch([Xtrain, charFinalIp], Ytrain)
##model.fit([Xtrain, charFinalIp], Ytrain,epochs=100, batch_size=6000, verbose=1)
