# CNNbasedNER_keras

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
