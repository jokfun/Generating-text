"""
TODO :
- Make it easier to use the project
- Improving hyperparameters
- Change the network
"""

import numpy as np
import os
import pickle
from keras.callbacks import ModelCheckpoint
from string import punctuation
import pickle

path = "wells/"
files = ["invisible_copy.txt","time_copy.txt"]#,"war_copy.txt"]

#Create the dataset with the different files
text = ""
for k in files:
    #Read the file
    content = open(path+k,encoding="utf-8").read()

    # remove caps and replace two new lines with one new line
    content = content.lower().replace("\n\n", "\n")

    # remove all punctuations
    # content = content.translate(str.maketrans("", "", punctuation))

    text+=content

#Determine all characters existing in the dataset
n_chars = len(text)
unique_chars = ''.join(sorted(set(text)))
print("unique_chars:", unique_chars)
n_unique_chars = len(unique_chars)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

#dictionary that converts characters to integers
#We save them in a pickle file (will be used for generating words)
char2int = {c: i for i, c in enumerate(unique_chars)}
fileObject = open("char2int.pickle",'wb')
pickle.dump(char2int,fileObject)
fileObject.close()

# dictionary that converts integers to characters
#We save them in a pickle file (will be used for generating words)
int2char = {i: c for i, c in enumerate(unique_chars)}
fileObject = open("int2char.pickle",'wb')
pickle.dump(int2char,fileObject)
fileObject.close()


#hyper parameters
"""
WARNING :
you need a lot of ram for the model to work, so pay attention to the following parameters:
- size of the dataset (==len(text))
- unique_chars : number of possible char, you can remove punctuation, capital letters or numbers...
"""
batch_size = 128
#size of the input
sequence_length = 100
epochs = 20
sentences = []

y_train = []
for i in range(0, len(text) - sequence_length):
    sentences.append(text[i: i + sequence_length])
    y_train.append(text[i+sequence_length])
print("Number of sentences:", len(sentences))

# vectorization of the training values
X = np.zeros((len(sentences), sequence_length, n_unique_chars))
y = np.zeros((len(sentences), n_unique_chars))

#Create features and labels var
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char2int[char]] = 1
        y[i, char2int[y_train[i]]] = 1
print("X.shape:", X.shape)
print("y.shape:", y.shape)

#create the model
model = model(n_unique_chars,sequence_length)

#Free some space
sentences = None
text = None

#Print the configuration of the network
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# make results folder if does not exist yet
if not os.path.isdir("results"):
    os.mkdir("results")
# save the model in each epoch

#Create a checkpoint
checkpoint = ModelCheckpoint("results/netCheckpoint-v1-{loss:.2f}.h5", verbose=1)

#Learning phase
#Can be super long
model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])



