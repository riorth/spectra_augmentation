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
import matplotlib.pyplot as plt
from keras.utils import plot_model

# define the standalone discriminator model
def define_discriminator(n_inputs=600):
	model = Sequential()
	model.add(tf.keras.Input(shape=(n_inputs,)))
	model.add(tf.keras.layers.Dense(1024))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))	
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(256))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	#model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	#model.add(tf.keras.layers.BatchNormalization())
	#model.add(tf.keras.layers.Dense(1, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
	#model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
	#model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
	
	return model


# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

# define the standalone generator model
def define_generator(latent_dim, n_outputs=600):
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
	# model.add(tf.keras.layers.Dense(256))
	# model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	# model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	model.add(tf.keras.layers.BatchNormalization())	
	model.add(tf.keras.layers.Dense(16))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	model.add(tf.keras.layers.BatchNormalization())	
	model.add(tf.keras.layers.Dense(n_outputs))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.3))	
	#model.add(Dense(n_outputs, activation='tanh'))	
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
	#model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)) #was best
	return model

# generate n real samples with class labels
# def generate_real_samples(n):
# 	# generate inputs in [-0.5, 0.5]
# 	X1 = rand(n) - 0.5
# 	# generate outputs X^2
# 	X2 = X1 * X1
# 	# stack arrays
# 	X1 = X1.reshape(n, 1)
# 	X2 = X2.reshape(n, 1)
# 	X = hstack((X1, X2))
# 	# generate class labels
# 	y = ones((n, 1))
# 	return X, y

# def get_real_samples(n):
#     #############################################################################################################
#         #msu data analysis
#     #############################################################################################################     
# 	path_to_data = os.path.join('data/')

# #############################################################################################################    
# 	# Чтение данных из файлов
# #############################################################################################################    
# 	file_list = os.listdir(path_to_data)
# 	X=[]
# 	for fn in file_list:
# 		with open(os.path.join(path_to_data, fn)) as ref:
# 			result = np.genfromtxt(ref, delimiter=",")
# 			#result = np.loadtxt(ref,delimiter="\t")
# 			X = np.copy(result[:,1:].T)
# 			c=np.linalg.norm(X,axis=1,ord=1)
# 			d=np.reshape(c,(26,1))
# 			tmp=np.divide(X,d)		
# 			X=np.copy(np.multiply(tmp,30))
# 	y = ones((np.shape(X)[0], 1)) - 0.05 * np.random.uniform(size=(np.shape(X)[0], 1))
# 	#y = smooth_positive_labels(y)
# 	random_indices = np.random.choice(26, size=n, replace=False)
# 	X_sub = X[random_indices, :]
# 	y_sub = y[random_indices, :]
# 	#return X, y
# 	return X_sub, y_sub

def get_real_samples(n):
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
	# split_point = 196
	# num_entries = 400
	for fn in file_list:
		with open(os.path.join(path_to_input_data, fn)) as ref:
			result = np.genfromtxt(ref, delimiter=" ", skip_header=14)
			#result = np.loadtxt(ref,delimiter="\t")
			#X_1 = np.copy(result[115:split_point,1].T)
			#X_2 = np.copy(result[split_point:num_entries,2].T)
			X_1 = np.copy(result[100:400,1].T)
			X_2 = np.copy(result[100:400,2].T)
			#X.append(np.array(result[115:num_entries,1:].T))
			X.append(np.asarray(np.concatenate((X_1, X_2))/np.max(np.concatenate((X_1, X_2)))))
			#X.append(np.asarray(np.concatenate((X_1, X_2))))
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
	random_indices = np.random.choice(np.shape(X)[0], size=n, replace=False)
	X_sub = np.asarray(X)[random_indices]
	y_sub = y[random_indices]
	return X_sub, y_sub

#	return np.asarray(X), np.asarray(y)


# example of smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
	return y - 0.3 + (randn(y.shape[0]) * 0.5)

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	#y = zeros((n, 1)) + 0.05 * np.random.uniform() #Found info, that smoothing generator labels is not good
	y = zeros((n, 1))
	return X, y

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=10):
	# prepare real samples
	#x_real, y_real = generate_real_samples(n)
	x_real, y_real = get_real_samples(n)
	# evaluate discriminator on real examples
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=1)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	# evaluate discriminator on fake examples
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=1)
	# summarize discriminator performance
	print(epoch, acc_real, acc_fake)
	# scatter plot real and fake data points
	pyplot.plot(x_real[0, :], color='red')
	pyplot.plot(x_fake[0, :], color='blue')
	#pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
	#pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
	pyplot.savefig('res_epoch_'+str(epoch)+'.png')
	#pyplot.show()
	pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=20, n_eval=1000):
	# determine half the size of one batch, for updating the discriminator
	half_batch = int(n_batch / 2)
	#train_batch = 26
	log_dir = './logs'
	use_tensorboard = False
	if use_tensorboard:
		summary_writer = tf.summary.create_file_writer(log_dir)
	# manually enumerate epochs
	for j in range(100):
		x_real, y_real = get_real_samples(half_batch)
		d_loss_real = d_model.train_on_batch(x_real, y_real)
		print("%d [D loss: %f, acc.: %.2f%%]" %(j + 1, d_loss_real[0], 100 * d_loss_real[1]))
	for i in tqdm(range(n_epochs)):
		# prepare real samples
		x_real, y_real = get_real_samples(half_batch)
		#x_real, y_real = get_real_samples(n_batch)
		# prepare fake examples
		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator
		d_loss_real = d_model.train_on_batch(x_real, y_real)
		d_loss_fake = d_model.train_on_batch(x_fake, y_fake)
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		g_loss = gan_model.train_on_batch(x_gan, y_gan)
		if use_tensorboard:
			with summary_writer.as_default():
				tf.summary.scalar('d_loss', d_loss[0], step=1)
				tf.summary.scalar('g_loss', g_loss[0], step=1)
		# evaluate the model every n_eval epochs
		if (i+1) % n_eval == 0:
			print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(i + 1, d_loss[0], 100 * d_loss[1], g_loss))
			summarize_performance(i, g_model, d_model, latent_dim)

# size of the latent space
#latent_dim = 5
latent_dim = 300

load_existing_models = True
if load_existing_models == True:
	generator = tf.keras.models.load_model('generator')
	discriminator = tf.keras.models.load_model('discriminator')
	gan_model = tf.keras.models.load_model('gan')
else:
	# create the discriminator
	discriminator = define_discriminator()
	# create the generator
	generator = define_generator(latent_dim)
	# create the gan
	gan_model = define_gan(generator, discriminator)
	# train model

	train(generator, discriminator, gan_model, latent_dim)
	generator.save('generator')
	discriminator.save('discriminator')
	gan_model.save('gan')

# import visualkeras
# visualkeras.layered_view(gan_model) 
plot_model(gan_model, to_file='gan_model.png', show_shapes=True, show_layer_names=True)
augmented_data, y_fake = generate_fake_samples(generator, latent_dim, 100)
np.savetxt('augmented_data.csv', augmented_data, delimiter=',')

print('Done')