import numpy as np
import os
import pickle
from keras.callbacks import ModelCheckpoint
from string import punctuation
import tqdm
from network import model

#Load the network that has been driven
#the model properties must be the same of the learning one
model.load_weights("results/netCheckpoint-v1-1.27.h5")

#The sentence you want to start with
seed = "i have no idea how to"

#Load the properties of your dataset
char2int = pickle.load(open("char2int.pickle", "rb"))
int2char = pickle.load(open("int2char.pickle", "rb"))

# building the model
sequence_length = 100
n_unique_chars = len(char2int)
model = model(n_unique_chars,sequence_length)

# generate 1000 characters
generated = ""
for i in tqdm.trange(1000):
    # make the input sequence
    X = np.zeros((1, sequence_length, n_unique_chars))
    for t, char in enumerate(seed):
    	#Convert the input into numeric values
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    #prediction
    predicted = model.predict(X, verbose=0)[0]
    #take the index of the 'most important' value
    next_index = np.argmax(predicted)
    #match of the selected predicted value
    next_char = int2char[next_index]
    #add the character to results
    generated += next_char
    #update the seed
    seed = seed[1:] + next_char

print("Generated text:")
print(generated)