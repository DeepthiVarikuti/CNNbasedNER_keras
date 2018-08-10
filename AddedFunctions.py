import os
import sys
import random
import numpy as np
from keras.utils import to_categorical, Progbar
from keras.preprocessing.sequence import pad_sequences

def ExtractFeatures(filename):
    # readin the textfile and break down the sentences into words and tags,
    #finally categorize the tags into N classes
    
    with open(filename) as f:
        sentences, tags, Characs = [], [], []
        l = 0
        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(sentences) != 0:
                    sentences, tags = sentences, tags
            else:
                ls = line.split(' ')
                word, tag = ls[0],ls[-1]
                chars = [c for c in ls[0]]
                Characs.append(chars)
                sentences.append(word)
                tags.append(tag)
                
    vals, counts = np.unique(tags, return_counts=True)
    label_id = [x for x in range(len(vals))]
    labelIndices =[None] * len(tags)

    for j, fixID in enumerate(vals):
        for i, IndID in enumerate(tags):
            if IndID == fixID:
               labelIndices[i] = label_id[j]

    Ylabel = to_categorical(np.asarray(labelIndices))
    
    return sentences, tags, Characs, Ylabel

def text2VecConversion(sentences):
    ## :: Read in glove dictionary and associate our words to the words from the dictionary::
    word2Idx = {}
    wordEmbeddings = []

    fEmbeddings = open("/home/deepthi/MainData/Huddl/Glove_6B/glove.6B.100d.txt", encoding="utf-8")

    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]  
        if len(word2Idx) == 0: 
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split)-1) 
            wordEmbeddings.append(vector)
            
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings.append(vector)
            
        if split[0].lower() in word:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)
            
    wordEmbeddings = np.array(wordEmbeddings)
   
    # associating our vocabulary to the golve vocabulary
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']
    wordCount = 0
    unknownWordCount = 0
    wordIndices = [] 
    for word in sentences:   
        wordCount += 1
        if word in word2Idx:
           wordIdx = word2Idx[word]
        elif word.lower() in word2Idx:
           wordIdx = word2Idx[word.lower()]
        else:
           wordIdx = unknownIdx
           unknownWordCount += 1
        wordIndices.append(wordIdx)

    return wordIndices, wordEmbeddings

def mappingTheCharacters(Characs):
    # :: mapping for characters ::
    char2Idx = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx[c] = len(char2Idx)

    charIndices = []
    for char in Characs:
        charIdx = []
        for x in char:
            charIdx.append(char2Idx[x])
        charIndices.append(charIdx)

    charIndices = pad_sequences(charIndices,52,padding='post')

    return charIndices, char2Idx

