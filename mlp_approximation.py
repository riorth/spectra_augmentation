import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#Read data from file into list and parse it
with open('data.txt') as file:
    lines = [line.rstrip().split() for line in file]
    #lines=[[int(i) for single_line in lines] for i in single_line] 
    lines=[[int(i) for i in single_line] for single_line in lines]   

train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size = 0.1, random_state=42)

num_neurons = 300

classifier = MLPClassifier(hidden_layer_sizes=num_neurons, max_iter=35, activation='relu', solver='sgd', verbose=10, random_state= 42, learning_rate='invscaling')
classifier.fit(train_X, train_Y)

prediction = classifier.predict(test_X)
score = np.round(metrics.accuracy_score(test_X, test_Y), 2)