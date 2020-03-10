from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

"""
	Custom the network here
	Be careful of your machine's capacity
"""
def model(n_unique_chars,sequence_length):
	"""
		Create the model of the network

		Required parameters :
		n_unique_chars : number of unique chars in the dataset
		sequence_length:  size of the input sentence
	"""
	#LASTM ar eused there because they're powerful
	model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars)),
    Dropout(0.2),
    Dense(n_unique_chars, activation="softmax"),
	])
	return model