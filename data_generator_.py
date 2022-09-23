# train a generative adversarial network on a one-dimensional function
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
import tensorflow as tf
from matplotlib import pyplot
import os 
import numpy as np
from keras import backend
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import TensorBoard
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# define the standalone discriminator model
def define_discriminator(n_inputs=88):
	model = Sequential()
	model.add(tf.keras.Input(shape=(n_inputs,)))
	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))	
	#model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(128))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	#model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(128))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	#model.add(tf.keras.layers.BatchNormalization())
	#model.add(tf.keras.layers.Dense(1, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
	#model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
	#model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
	
	return model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=88):
	model = Sequential()
	##model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
	# best config
	# model.add(tf.keras.Input(shape=(latent_dim,)))
	# model.add(tf.keras.layers.Dense(64))
	# model.add(tf.keras.layers.BatchNormalization())
	# #model.add(tf.keras.layers.ReLU())	
	# model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	# model.add(tf.keras.layers.Dense(16))
	# model.add(tf.keras.layers.BatchNormalization())
	# model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	# model.add(tf.keras.layers.Dense(4))
	# model.add(tf.keras.layers.BatchNormalization())
	# model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	# #model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
	# model.add(tf.keras.layers.Dense(n_outputs))
	# model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	# #model.add(Dense(n_outputs, activation='linear'))

	#test configuration
	model.add(tf.keras.Input(shape=(latent_dim,)))
	model.add(tf.keras.layers.Dense(256))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	model.add(tf.keras.layers.BatchNormalization())	
	model.add(tf.keras.layers.Dense(16))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	model.add(tf.keras.layers.BatchNormalization())	
	model.add(tf.keras.layers.Dense(n_outputs))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	#model.add(Dense(n_outputs, activation='linear'))	
	return model

def get_real_samples():
    #############################################################################################################
        #msu data analysis
    #############################################################################################################     
	#path_to_input_data = os.path.join('data/in/')
	path_to_input_data = os.path.join('data/in_all/')
	path_to_output_data = os.path.join('data/out/')

#############################################################################################################    
	# Чтение данных из файлов
#############################################################################################################    
	file_list = os.listdir(path_to_input_data)
	X=[]
	split_point = 196
	num_entries = 400
	for fn in file_list:
		with open(os.path.join(path_to_input_data, fn)) as ref:
			result = np.genfromtxt(ref, delimiter=" ", skip_header=14)
			#result = np.loadtxt(ref,delimiter="\t")
			#X_1 = np.copy(result[115:split_point,1].T)
			#X_2 = np.copy(result[split_point:num_entries,2].T)
			X_1 = np.copy(result[:,1].T)
			X_2 = np.copy(result[:,2].T)
			#X.append(np.array(result[115:num_entries,1:].T))
			X.append(np.asarray(np.concatenate((X_1, X_2))/np.max(np.concatenate((X_1, X_2)))))
			#c=np.linalg.norm(X,axis=1,ord=1)
			#d=np.reshape(c,(26,1))
			#tmp=np.divide(X,d)		
			#X=np.copy(np.multiply(tmp,30))
	y = ones((np.shape(X)[0], 1)) - 0.05 * np.random.uniform(size=(np.shape(X)[0], 1))
	#y = smooth_positive_labels(y)
	# random_indices = np.random.choice(26, size=n, replace=False)
	# X_sub = X[random_indices, :]
	# y_sub = y[random_indices, :]
	#return X, y

	# file_list = os.listdir(path_to_output_data)
	# y=[]
	# for fn in file_list:
	# 	with open(os.path.join(path_to_output_data, fn)) as ref:
	# 		result = np.genfromtxt(ref, delimiter=",")
	# 		#y_1 = np.copy(result[115:split_point,2].T)
	# 		#y_2 = np.copy(result[split_point:num_entries,1].T)			
	# 		y_1 = np.copy(result[:,1].T)
	# 		y_2 = np.copy(result[:,2].T)			

	# 		#result = np.loadtxt(ref,delimiter="\t")		
	# 		y.append(np.concatenate((y_1, y_2))/np.max(np.concatenate((y_1, y_2))))
	# 		#y.append(np.array(result[115:500,1:3].T))
	
	return np.asarray(X), np.asarray(y)



X, y = get_real_samples()
#print(np.any(np.isnan(X)))
print(np.any(np.isnan(y)))
num_inputs = len(X[0])
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.2, random_state=42)


# design the neural network model
model = Sequential()
model.add(Dense(num_inputs * 2, input_dim=num_inputs, activation='softmax', kernel_initializer='he_uniform'))
#model.add(Dense(num_inputs * 2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_inputs))
# define the loss function and optimization algorithm
opt = opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=opt)
# ft the model on the training dataset
model.fit(train_X, train_Y, epochs=50, batch_size=2, verbose=1)
# make predictions for the input data
#yhat = model.predict(X)
prediction = model.predict(test_X)
score = np.round(metrics.accuracy_score(test_X, test_Y), 2)






print('Done')