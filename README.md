# Generate text with machine learning

Generating text from a set of books

## Basic use

1. Learning phase, beware it requires a lot of memory space (ram) so do not hesitate to customize the hyperparameters 
(beware you need a large dataset and a sufficient number of epochs to have a powerful model) : 
```
	python runme.py
```

2. Depending on the name of the checkpoint you must surely edit its path in the file generate.py, 
after that run the following command to generate a new text : 
```
	python generate.py
```

## Author
Raphael Teitgen raphael.teitgen@gmail
The original idea came from [here](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)