"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression.utils import loadDataset
from regression.logreg import LogisticRegressor
import numpy as np
# (you will probably need to import more things here)

def test_prediction():
	X, y = loadDataset()								#load dataset
	model = LogisticRegressor(num_feats=X.shape[1])		#load model with same feature # as in data

	X_pad = np.hstack([X, np.ones((X.shape[0], 1))])	#pad data with ones (bias term)

	z = X_pad @ model.W									#linear model (Y = WX)
	expected = 1 / (1 + np.exp(-z))						#sigmoid function (manual = expected)

	y_pred = model.make_prediction(X_pad)				#my code's prediction

	assert np.allclose(y_pred, expected)				#assert prediction is close to expected

def test_loss_function():
	X, y = loadDataset()								#load dataset
	model = LogisticRegressor(num_feats=X.shape[1])		#load model with same feature # as in data

	X_pad = np.hstack([X, np.ones((X.shape[0], 1))])	#pad data with ones (bias term)
	y_pred = model.make_prediction(X_pad)				#make prediction

	eps = 1e-9											#avoid log(0) -> undefined
	y_pred_clip = np.clip(y_pred, eps, 1 - eps)			#bound the min/max range of y_pred 

	expected_loss = -np.mean(y * np.log(y_pred_clip) + (1 - y) * np.log(1 - y_pred_clip))	#manual loss function = expected
	loss = model.loss_function(y, y_pred)				#my code's binary cross entropy loss

	assert np.isclose(loss, expected_loss)				#assert loss is close to expected

def test_gradient():
	X, y = loadDataset()								#load dataset
	model = LogisticRegressor(num_feats=X.shape[1])		#load model with same feature # as in data

	X_pad = np.hstack([X, np.ones((X.shape[0], 1))])	#pad data with ones (bias term)
	y_pred = model.make_prediction(X_pad)				#make prediction

	expected_grad = (X_pad.T @ (y_pred - y)) / X_pad.shape[0]	#manual gradient equation = expected
	grad = model.calculate_gradient(y, X_pad)			#my code's gradient function

	assert np.allclose(grad, expected_grad)				#assert gradient is close to expected

def test_training():
	X_train, X_test, y_train, y_test = loadDataset(split_percent=0.8)	#load dataset (split 80% train and 20% test)
	model = LogisticRegressor(num_feats=X_train.shape[1], max_iter=5)	#load model with same feature # as in data

	initial_weights = model.W.copy()						#save my initial model weights
	model.train_model(X_train, y_train, X_test, y_test)		#train the model

	assert not np.allclose(initial_weights, model.W)		#when model trains, the weights should change significantly!

	assert len(model.loss_hist_train) > 0				#the loss history should not be empty
	assert len(model.loss_hist_val) > 0