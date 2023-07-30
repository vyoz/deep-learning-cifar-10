# test harness for evaluating models on the cifar10 dataset
import sys
import os
import numpy 
from datetime import datetime
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# load data in percentage
def load_data_partial(percent, dataX, dataY):
	len_orig = len(dataX)
	idx = numpy.arange(len_orig)
	numpy.random.shuffle(idx)
	subset_dataX = dataX[:int(percent*len(idx))]
	subset_dataY = dataY[:int(percent*len(idx))]
	len_new = len(subset_dataX)
	#print("origial data len:%d, new data let:%d" % (len_orig, len_new))
	return (subset_dataX, subset_dataY)
	
# load train and test dataset
def load_dataset(percent):
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()

	(trainX, trainY) = load_data_partial(percent, trainX, trainY)
	(testX, testY) = load_data_partial(percent, testX, testY)

	# one hot encode target values
	print("train data size:%d" % len(trainY))
	trainY = to_categorical(trainY)
	print("test data size:%d" % len(testY))
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model(train_method):
	if( train_method == "model1" ):
		return define_model1()
	elif ( train_method == "model2" ):
		return define_model2()
	elif ( train_method == "model3" ):
		return define_model3()
	else:
		print("invalid train method:%s" % train_method)
		sys.exit(-1)

# define cnn model - 1
def define_model1():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# define cnn model - 2
def define_model2():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
 model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
 model.add(MaxPooling2D((2, 2)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
 model.add(Dense(10, activation='softmax'))
 # compile model
 opt = SGD(lr=0.001, momentum=0.9)
 model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
 return model


# define cnn model - 3
def define_model3():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness(percent, epochs_num, train_method, file_model):
	# load dataset
	trainX, trainY, testX, testY = load_dataset(percent)
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model(train_method)
	# fit model
	history = model.fit(trainX, trainY, epochs=epochs_num, batch_size=64, validation_data=(testX, testY), verbose=0)
	model.save(file_model)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
	return acc * 100.0

def curr_time():
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

# entry point, run the test harness
# usage
#	python test-harness.py <data-percent> <epochs-num> <train-method>
if len(sys.argv) != 4 :
	print("Usage:\n%s <data-percent> <epochs-num> <train-method>" % sys.argv[0])
	print("where:\n")
	print("  <data-percent>: a float number from 0 to 1\n")
	print("  <epochs-num>: epochs num, range from 10 to 200\n")
	print("  <train-method>: available methods: [model1, model2, model3]\n")
	sys.exit(-1)

percent = float(sys.argv[1])
epochs_num = int(sys.argv[2])
train_method = sys.argv[3]

file_model = os.getcwd() + "/" + train_method + ".percent-" + str(percent) + ".epochs-" + str(epochs_num) + ".keras"
file_log = file_model + ".log"
f=open(file_log, "at")
msg = "{}: training with data set percentage:{} epochs-num:{} train-method:{}\n".format(curr_time(), percent, epochs_num, train_method)
print(msg)
f.write(msg)
acc = run_test_harness(percent, epochs_num, train_method, file_model)
msg =  "{}: result:{} model file:{}\n".format(curr_time(), acc, file_model)
print(msg)
f.write(msg);
f.close()
