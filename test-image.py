import os
import sys
# evaluate the deep model on the test dataset
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from datetime import datetime

def curr_time():
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def test_model(file_model, file_image):
	# load the image
	print("loading image:%s ..." % file_image)
	img = load_image(file_image)
	# load model
	print("loading model:%s..." % file_model)
	model = load_model(file_model)
	# predict the class
	# Convert the model's output indices to labels
	predictions = model.predict(img)
	print(predictions)
	label = convert_indices_to_labels(predictions.argmax(axis=-1))[0]
	print(label)
	return(label)

def convert_indices_to_labels(predictions):
	LABEL_LIST = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	index_to_label = dict(zip(list(range(len(LABEL_LIST))), LABEL_LIST))
	return [index_to_label[index] for index in predictions]

# entry point, run the test model
# usage
#	python test-model.py <file-model> <file-image>
if len(sys.argv) != 3 :
	file_model = "./final_model.h5"
	file_image = "./sample_image.png"
	print("Usage:\n%s <file-model> <file-image>" % sys.argv[0])
	print("where:\n")
	print("  <file-model>: default:%s, the model to be test\n" % file_model)
	print("  <file-image>: default:%s, a test image file\n" % file_image)
else:
	file_model = sys.argv[1]
	file_image = sys.argv[2]
	
file_log = file_image + ".log"
f=open(file_log, "at")
msg = "{}: running image file:{} on the model:{}\n".format(curr_time(), file_image, file_model)
print(msg)
f.write(msg)
result = test_model(file_model, file_image)
msg =  "{}: file:{} result:{} \n".format(curr_time(), file_image, result)
print(msg)
f.write(msg);
f.close()

