from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import matplotlib.pyplot as plt
import matplotlib.image

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.optimizers import adam

from autograd.util import quick_grad_check
from builtins import range

from data import mnist_processed
from data import save_images
from data import display_images
from data import unflatten
from data import prob2gray
from data import boost

import os
import argparse

# run interactively: exec(open("noisemodel-simple.py"),globals())

# shotgun softmax
# input: list of rows
# output: list of softmaxed rows
# note: since e^x gets very large, this max trick improves numerical stability (tested)
def softmax(x):
	e_x = np.exp(x - np.max(x))
	return np.divide(e_x,np.sum(e_x,axis=1,keepdims=True))

# construct weight initialization objective/gradient
def build_init_objective(L2_VAR_2, NUM_TRAIN, train_images, train_labels, C, D, L):
	# shotgun predictions: matrix of probabilities of each datapoint being in each class
	# input weight shape: ( flattened weights ) flattened weights are indexed by class, then flattened pixels
	# ( reshaped weight shape: ( class [j], pixel [k]) )
	# input image shape: ( training sample [l], flattened pixel [k] )
	# output prediction shape: ( class [j], training sample [l] )
	def predict(weights, inputs):
		return softmax(np.einsum('jk,lk->jl',weights.reshape(C,D), inputs))

	# Specify an inference problem by its unnormalized log-density.
	# input weight shape: ( flattened weights ) flattened weights are indexed by class, then flattened pixels
	# input image shape: ( training sample , flattened pixel ) 
	# input label shape: ( training sample  , one-hot class encoding ) 
	# ( prediction shape: ( class , training sample ) )
	# output density shape: (scalar)
	def log_density(weights, t):
		return (np.einsum('jk,kj->', np.log(predict(weights, train_images)), train_labels) - (np.sum((weights**2))/(2*L2_VAR_2)) )

	def objective(params, t):
		return -log_density(params, t)

	gradient = grad(objective)
	return objective, gradient

# construct noise-aware optimization objective/gradient
def build_noise_objective(L2_VAR_2, NUM_TRAIN, train_images, train_labels, C, D, N, L):
	# our first C**2 = N weights are for P(c|r), our next C*D = L weights are for logistic regression
	def unpack_params(params):
		c_l_r = params[:N].reshape(C,C)
		weights = params[N:]
		return c_l_r, weights

	# shotgun predictions: matrix of probabilities of each datapoint being in each class with simple logistic regression
	# input weight shape: ( flattened weights ) flattened weights are indexed by class, then flattened pixels
	# ( reshaped weight shape: ( class [j], pixel [k]) )
	# input image shape: ( training sample [l], flattened pixel [k] )
	# output prediction shape: ( class [j], training sample [l] )
	def logisticpredict(weights, inputs):
		return softmax(np.einsum('jk,lk->jl',weights.reshape(C,D), inputs))

	# shotgun predictions: matrix of probabilities of each datapoint being in each class with bayes' rule
	def bayespredict(c_l_r, weights, inputs):
		numerator = np.einsum('cr,rs->cs', softmax(c_l_r), logisticpredict(weights, inputs))
		denominator = np.sum(numerator,axis=0)
		return numerator / denominator

	# calculate predictive accuracy on train set and test set
	def accuracies(params):
		# unpack logged parameters
		optimized_c_l_r, optimized_w = unpack_params(params)
		predicted_train_classes = np.argmax(bayespredict(optimized_c_l_r, optimized_w, train_images),axis=0)
		predicted_test_classes = np.argmax(bayespredict(optimized_c_l_r, optimized_w, test_images),axis=0)
		# compute real classes
		real_train_classes = np.argmax(train_labels,axis=1)
		real_test_classes = np.argmax(test_labels,axis=1)
		# compute accuracy
		train_accuracy = np.average(np.equal(predicted_train_classes,real_train_classes).astype(float))
		test_accuracy = np.average(np.equal(predicted_test_classes,real_test_classes).astype(float))
		# output accuracy
		return train_accuracy, test_accuracy

	# Specify an inference problem by its unnormalized log-density.
	# input weight shape: ( flattened weights ) flattened weights are indexed by class, then flattened pixels
	# input image shape: ( training sample , flattened pixel ) 
	# input label shape: ( training sample  , one-hot class encoding ) 
	# ( prediction shape: ( class , training sample ) )
	# output density shape: (scalar)
	def log_density(c_l_r, weights, t):
		return (np.einsum('jk,kj->', np.log(np.einsum('cr,rs->cs', softmax(c_l_r), logisticpredict(weights, train_images))), train_labels) - (np.sum((weights**2))/(2*L2_VAR_2)) )

	def objective(params, t):
		c_l_r, weights = unpack_params(params)
		return -log_density(c_l_r, weights, t)

	gradient = grad(objective)
	return accuracies, objective, gradient, unpack_params

if (__name__ == '__main__'):
	# determine the size of the train/test sets from command-line args
	parser = argparse.ArgumentParser(description='no description yet')
	parser.add_argument('-corruptionone', type=float, default=0.00)
	parser.add_argument('-corruptiontwo', type=float, default=0.00)
	parser.add_argument('-train', type=int, default=10000)
	parser.add_argument('-test', type=int, default=10000)
	parser.add_argument('--mini', action='store_true') # SMALL RUN: force train and test set to be small, and both iters to be tiny
	parser.add_argument('-var', type=float, default=0.1)
	parser.add_argument('-initers', type=int, default=200)
	parser.add_argument('-iters', type=int, default=100)
	parser.add_argument('--fullaccuracy', action='store_true')
	args = parser.parse_args()

	if not args.mini:
		ARGS_TRAIN = args.train
		ARGS_TEST = args.test
		INIT_ITERS = args.initers
		NUM_ITERATIONS = args.iters
	else:
		ARGS_TRAIN = 30
		ARGS_TEST = 30
		INIT_ITERS = 5
		NUM_ITERATIONS = 5

	L2_VAR_2 = args.var

	P_C_ONE = args.corruptionone
	P_C_TWO = args.corruptiontwo

	# load the data
	NUM_TRAIN, NUM_TEST, train_images, train_labels, test_images, test_labels = mnist_processed(num_train=ARGS_TRAIN, num_test=ARGS_TEST, binarize_img=True, flatten_img=True, one_hot_label=True, corrupt=True, p_c_one=P_C_ONE, p_c_two = P_C_TWO)

	C = 10 # # [C]lasses
	D = 784 # # per-class feature [D]imensions

	N = C**2 # # [N]oise weights
	L = C * D # # [L]ogistic weights

	###############################################################
    # PERFORM NAIVE LOGISTIC REGRESSION TO INITIALIZE THE WEIGHTS #
	###############################################################
	filename = "initialization_weights_" + str(P_C_ONE) + "_" + str(P_C_TWO) + "_" + str(L2_VAR_2) + "_" + str(NUM_TRAIN) + ".npz"
	picklefilename = os.path.join('cache', filename)
	if os.path.isfile(picklefilename): 
		# restore pickled weights from /cache (if exist)
		print("restoring pickled initialization weights")
		unpickled = np.load(picklefilename)
		init_weights = unpickled['init_weights']
	else:
		# Build initialization objective 
		objective, init_gradient = build_init_objective(L2_VAR_2, NUM_TRAIN, train_images, train_labels, C, D, L)

		# Build callback for ADAM optimizer
		def init_callback(params, t, g):
			lik = -objective(params, t)
			print("Initialization iteration {} log-likelihood {}".format(t, lik))

		# initialize weights
		pre_init_weights = np.ones(L)

		# optimize weights
		print("Initializing weights...")
		init_weights = adam(init_gradient, pre_init_weights, step_size=0.1, num_iters=INIT_ITERS, callback=init_callback)

		# pickle processed data in /cache (if doesn't already exist)
		if not os.path.exists('cache'):
			print('creating cache folder')
			os.makedirs('cache')
		if not os.path.isfile(picklefilename):
			print('saving pickled regression initalization data')
			np.savez(picklefilename, init_weights=init_weights)
	###############################################################

	###############################################################
    # OPTIMIZE NOISE-AWARE LIKELIHOOD #
	###############################################################
	# Build noise-aware logistic regression objective.
	accuracies, objective, gradient, unpack_params = build_noise_objective(L2_VAR_2, NUM_TRAIN, train_images, train_labels, C, D, N, L)

	# Build callback for ADAM optimizer
	liklog = np.zeros(NUM_ITERATIONS)
	param_log = np.zeros((NUM_ITERATIONS, N+L))
	def callback(params, t, g):
		lik = -objective(params, t)
		liklog[t] = lik
		param_log[t] = params
		if not args.fullaccuracy:
			print("Iteration {} log-likelihood {}".format(t, lik))

	# Initialize noise weights, carry over initialized logistic regression weights
	init_c_l_r = np.identity(C).reshape(-1) + 0.0001
	init_params = np.concatenate([init_c_l_r, init_weights])

	# Optimize all parameters
	print("Optimizing parameters...")
	optimized_params = adam(gradient, init_params, step_size=0.1, num_iters=NUM_ITERATIONS, callback=callback)
	###############################################################

	# output remaining accuracy info
	if args.fullaccuracy:
		for i in range(0,NUM_ITERATIONS):
			print("Iteration {} log-likelihood {}".format(i, liklog[i]))
			train_accuracy, test_accuracy = accuracies(param_log[i])
			print("Iteration {} train_accuracy: {}".format(i, train_accuracy))
			print("Iteration {} test_accuracy: {}".format(i, test_accuracy))
	else:
		train_accuracy, test_accuracy = accuracies(optimized_params)
		print("train_accuracy: {}".format(train_accuracy))
		print("test_accuracy: {}".format(test_accuracy))

	#return estimated convergence point (first log-likelihood in the log within <margin> of the average of the last <finalsamples> log-liklihoods)
	finalsamples = 5
	margin = 10
	rejectmargin = 50
	goal = np.average(liklog[-finalsamples:])
	convergenceindex = np.argmax(liklog > (goal - margin)) #argmax returns first index of a True in this case
	if (np.abs(goal - liklog[-1]) > rejectmargin):
		print("log-likelihood has not yet converged")
	else:
		print("log-likelihood  converges by iteration " + str(convergenceindex))

	# Plot weights
	optimized_c_l_r, optimized_w = unpack_params(optimized_params)
	print(softmax(optimized_c_l_r))
	weights = optimized_w.reshape(C,28,28)
	save_images(weights, "weights.jpg")