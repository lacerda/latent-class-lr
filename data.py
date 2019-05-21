from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import numpy as np
import os
import gzip
import struct
import array

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve

# (Helper): 
# given a URL and a filename, download the file (if necessary) into /data (and create the /data folder if necessary)
def download(url, filename):
	if not os.path.exists('data'):
		#print('Download: creating data folder')
		os.makedirs('data')
	out_file = os.path.join('data', filename)
	if not os.path.isfile(out_file):
		#print('Download: retrieving file ' + filename)
		urlretrieve(url, out_file)
	else:
		#print('Download: file ' + filename + ' already exists')
		pass

# (main base function: download MNIST data if necessary and unpack it)
# get numpy arrays of the MNIST images and labels, split into training and test sets (60K/10K)
# each image is a 2D 28x28 numpy array with each pixel being a uint8 from 0 to 255 (monochrome)
# each label is a uint8
def mnist():
	base_url = 'http://yann.lecun.com/exdb/mnist/'

	def parse_labels(filename):
		with gzip.open(filename, 'rb') as fh:
			magic, num_data = struct.unpack(">II", fh.read(8))
			return np.array(array.array("B", fh.read()), dtype=np.uint8)

	def parse_images(filename):
		with gzip.open(filename, 'rb') as fh:
			magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
			return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

	for filename in ['train-images-idx3-ubyte.gz',
					 'train-labels-idx1-ubyte.gz',
					 't10k-images-idx3-ubyte.gz',
					 't10k-labels-idx1-ubyte.gz']:
		download(base_url + filename, filename)

	train_images = parse_images('data/train-images-idx3-ubyte.gz')
	train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
	test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')
	test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')

	return train_images, train_labels, test_images, test_labels

# get numpy arrays of the MNIST images and labels, split into training and test sets of specified size (up to 60K / 10K)
# if binarize_img is True, each image is binarized -- each pixel is a float64 that is either 0.0 or 1.0. If binarize_img is False, each pixel is a uint8 from 0 to 255
# if flatten_img is True, each image has been flattened into a 1D 784 -long numpy array. If flatten_img is False, each image is a 2D 28x28 numpy array
# if one_hot_label is True, each label is each label is a 1D 10 -long int32 numpy array from 0 to 1 (one-hot). If one_hot_label is False, each label is a uint8 from 0 to 9 representing the real digit.
def mnist_processed(num_train=10000, num_test=10000, binarize_img=True, flatten_img=True, one_hot_label=True, corrupt=False, p_c_one=0.0, p_c_two=0.0):
	# build the filename that we expect our pickle file to have
	filename = "mnist_processed_" + str(num_train) + "_" + str(num_test) + "_" + str(int(binarize_img)) + "_" + str(int(flatten_img)) + "_" + str(int(one_hot_label)) + "_" + str(int(corrupt)) + "_" + str(p_c_one) + "_" + str(p_c_two) + ".npz"
	picklefilename = os.path.join('cache', filename)

	if os.path.isfile(picklefilename): # restore pickled processed data from /cache (if exists)
		print("mnist_processed: restoring pickled processed MNIST data")

		unpickled = np.load(picklefilename)
		train_images = unpickled['train_images']
		train_labels = unpickled['train_labels']
		test_images = unpickled['test_images']
		test_labels = unpickled['test_labels']

		num_train = train_images.shape[0]
		num_test = test_images.shape[0]
		
	else: # process the unpacked mnist data (if no pickled data exists)
		print("mnist_processed: getting unpacked MNIST data")

		# load raw images
		train_images, train_labels, test_images, test_labels = mnist()

		print("mnist_processed: processing unpacked MNIST data")

		# clip training and test counts to available number of images
		num_train = min(num_train,train_images.shape[0])
		num_test = min(num_test,test_images.shape[0])

		# select the number we want
		train_images = train_images[0:num_train]
		train_labels = train_labels[0:num_train]
		test_images = test_images[0:num_test]
		test_labels = test_labels[0:num_test]

		# binarize images?
		if binarize_img:
			train_images = np.rint(train_images / 255.0)
			test_images = np.rint(test_images / 255.0)

		# flatten images?
		if flatten_img:
			partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
			train_images = partial_flatten(train_images)
			test_images  = partial_flatten(test_images)

		# corrupt labels?
		if corrupt:
			for i, label in enumerate(train_labels):
				die = np.random.random_sample()
				if (die < p_c_one):
					initiallabel = label
					while True:
						train_labels[i] = np.random.randint(10)
						if not (train_labels[i] == initiallabel):
							break
				elif (die > p_c_one) and (die < (p_c_one + p_c_two)):
					alpha = 0.75
					newdie = np.random.random_sample()
					if (label == 1) and (newdie < alpha):
						train_labels[i] = 7
					if (label == 7) and (newdie < (1.0-alpha)):
						train_labels[i] = 1

					if (label == 3) and (newdie < alpha):
						train_labels[i] = 8
					if (label == 8) and (newdie < (1.0-alpha)):
						train_labels[i] = 3

					if (label == 4) and (newdie < alpha):
						train_labels[i] = 9
					if (label == 9) and (newdie < (1.0-alpha)):
						train_labels[i] = 4

		# one hot labels?
		if one_hot_label:
			one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
			train_labels = one_hot(train_labels, 10)
			test_labels = one_hot(test_labels, 10)

		# pickle processed data in /cache (if doesn't already exist)
		if not os.path.exists('cache'):
			print('mnist_processed: creating cache folder')
			os.makedirs('cache')
		if not os.path.isfile(picklefilename):
			print('mnist_processed: saving pickled MNIST data')
			np.savez(picklefilename, train_images=train_images, train_labels=train_labels, test_images=test_images, test_labels=test_labels)

	return num_train, num_test, train_images, train_labels, test_images, test_labels

# (Helper): 
# given a list of images (2D 0-255 arrays) and a pyplot axis, plots the images on the axis (?)
def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
				cmap=matplotlib.cm.binary, vmin=None, vmax=None):
	"""Images should be a (N_images x pixels) matrix."""
	N_images = images.shape[0]
	N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
	pad_value = np.min(images.ravel())
	concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
							 (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
	for i in range(N_images):
		cur_image = np.reshape(images[i, :], digit_dimensions)
		row_ix = i // ims_per_row
		col_ix = i % ims_per_row
		row_start = padding + (padding + digit_dimensions[0]) * row_ix
		col_start = padding + (padding + digit_dimensions[1]) * col_ix
		concat_images[row_start: row_start + digit_dimensions[0],
					  col_start: col_start + digit_dimensions[1]] = cur_image
	cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	return cax

# take an image and turn it into an "array of images" length 1: np.expand_dims(x, axis=0)

# Given a numpy array of images (2D 0-255 arrays) and a filename, collates them and saves the image
def save_images(images, filename, **kwargs):
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(111)
	plot_images(images, ax, **kwargs)
	fig.patch.set_visible(False)
	ax.patch.set_visible(False)
	plt.savefig(filename)
	fig.clf()

# Given a numpy array of images (2D 0-255 arrays) displays them
def display_images(images, **kwargs):
	fig = plt.figure()
	fig.clf()
	ax = fig.add_subplot(111)
	plot_images(images, ax, **kwargs)
	fig.patch.set_visible(False)
	ax.patch.set_visible(False)
	plt.show()

# given a flattened (1D array) square image, returned an unflattened (2D array) square image
def unflatten(img):
	side = int(np.sqrt(img.shape[0]))
	return np.reshape(img,(side, side))

# given an array (1D or 2D) of probabilities, return an array of grayscale int color values (0.0 => 0 (black), 0.5 => gray, 1.0 => white)
def prob2gray(probs):
	return np.clip(np.rint(probs * 255).astype(int),0,255)

# boost the visibility of a 2D image
def boost(img):
	return np.clip(     np.rint(((((img.astype('float64') / 255.0) ** 2) * 200) * 255)).astype(int)      ,0,255              )