Implementation of a keras based neural networks for Named-Entity-Recoganition

Pre-requisite:

Make golve by downloading the GloVe dictionary ("Glove_6B/glove.6B.100d.txt") and replace the respective path. 

Inorder to train the model, please run "MainModel.py"

Explanation:

1) A sentence within the CoNLL's dataset seems to give a tag to each word

EU    rejects   German   call   to   boycott   British   lamb
B-ORG   O  	B-MISC 	   O  	 O 	 O 	B-MISC    O


Therefore, the first step is to seperate the Words and their respective tags for all the given datasets.

2) secondly, Identification of the 'N' categories of tags given in the dataset has been implemented. After identifying the 'N' number of categories, each tag is associated with a number and the indices of respective tag category is assigned to that specific number

from AddedFunctions.py, ExtractFeatures implemented the step 1 and step 2, to provide an output of Words, Tags, Characters(individual characters of the 'Words') and finally Ylabel (i.e., Tags converted into categorical numbers)

3) Now the individual characters that are generally existing is associated with a numerical representation and futher associate these numbers with the individual characters of the words, in order to provide a character-based representation of each word

from AddedFunctions.py, mappingTheCharacters function performs what is explained in the step 3

4) Now, the words and also the characters from each words are linked / verified with an existing dictionary (i.e., Glove dictionary). In this procedure, a pre-existing word representation is used to correlate the words from our datasets and further assign a numerical value (i.e., converting the text format to vector format ina meaningful way)

from AddedFunctions.py, text2VecConversion function performs what is explained in the step 3

5) The model is build using the features such as, 1) words based vector representation and 2) character based vector representation


6) Convoluted Neural networks from keras is used to build the model on the above mentioned feature space. In which the following steps are integrated. 

In which, first as embedding layer is applied on the feature space (words based vectors & Character based vector), inorder to convert them into an output space to obtain the weights of the features. 

A CNN consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of convolutional layers, pooling layers, fully connected layers and normalization layers

Further the hidden layers are applied, i.e., a time distributed convolution layer & pooling layer. Therefore, a lower-dimensional cross-correlated information is extracted from the feature space. 

There on, a flattening step is applied to bring the dimensions of the array into a single long continuous vector, which further helps us to make use of fully connected layer. 

Then, addressed the overfitting issue by applying dropout to the output nodes based on the suggestion from "Chiu & Nihols 2016 (Named Entity Recognition with Bidirectional LSTM-CNNs). 

After concatenating the features space (i.e., word based representation and character based representation), a recurrent layer is implements using bidirectional-LSTM, which process the timesteps in two recursive layers (input sequence, and a copy of the input sequence). 

Finally, a fully connected layer is applied using 'Dense' function (A linear operation in which every input is connected to every output by a weight). 

7) Thus, the model is build using CNN, finally the model is compiled and fitted on the text sequences and the tags. 


Note: This is my understanding of the concepts and how to develop the code in python. However, i couldn't resolve the issues to fit the model between X and Y. I would need more time to deeply understand the concepts, which is necessary to resolve the issues. Therefore, i decided to stop at this point and submit my work. Thank you for the oppurtunity. 

