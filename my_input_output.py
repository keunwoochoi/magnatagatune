import sys
import os
import h5py
import numpy as np
from environments import *
from constants import *
sys.path.append(PATH_EMBEDDING)
import hdf5matrix

def load_x_hdf5matrix(tf_type=None):
	'''using hdf. perhaps you should set PATH_HDF_LOCAL for the machine you're using.
	tf_type: cqt, stft, mfcc, chroma. ''
	task = 'reg', 'cla'
	for any tf_type, any post-processing is not required except standardization.
	'''
	def normalizer_cqt(input_data):
		global_mean = -28.3472 # computed from the whole data for cqt #all this nums from ilm10k, so should be updated.
		global_std  = 6.59574
		return (input_data - global_mean) / global_std

	def normalizer_stft(input_data):	
		global_mean = -2.01616 # should be mended with STFT values
		global_std  = 9.23697
		return (input_data - global_mean) / global_std

	def normalizer_mfcc(input_data):
		global_mean = 2.1356969
		global_std = 16.260582
		return (input_data - global_mean) / global_std
	
	def normalizer_melgram(input_data):
		global_mean = -1.65182
		global_std = 21.5
		return (input_data - global_mean) / global_std

	if tf_type is None:
		tf_type = 'cqt'
	
	if tf_type == 'stft':
		normalizer = normalizer_stft
	elif tf_type == 'cqt':
		normalizer = normalizer_cqt
	elif tf_type == 'mfcc':
		normalizer = normalizer_mfcc
	elif tf_type == 'melgram':
		normalizer = normalizer_melgram		
	else:
		normalizer = None

	train_x = hdf5matrix.HDF5Matrix(PATH_HDF_LOCAL + 'magna_train.hdf', tf_type, None, None, normalizer=normalizer)
	valid_x = hdf5matrix.HDF5Matrix(PATH_HDF_LOCAL + 'magna_valid.hdf', tf_type, None, None, normalizer=normalizer)	
	test_x  = hdf5matrix.HDF5Matrix(PATH_HDF_LOCAL + 'magna_test.hdf',  tf_type, None, None, normalizer=normalizer)
	
	return train_x, valid_x, test_x

def load_x(tf_type):
	print 'Load x will load a standardised %s' % tf_type
	ret = []
	for i in range(15):
		ret.append(h5py.File(PATH_HDF_LOCAL + 'magna_%d.hdf' % i)[tf_type])
	
	return ret

def load_y(top_n=50, merged=True):
	print 'Load y will load top-%d labels. is it merged? %s' % (top_n, str(merged))
	if merged:
		name = 'y_merged'
	else:
		name = 'y_original'
	ret = []
	for i in range(15):
		ret.append(h5py.File(PATH_HDF_LOCAL + 'magna_%d.hdf' % i)[name])
		
	return ret
"""
def load_x(tf_type):
	print 'Load x will load a standardised %s' % tf_type
	train_x = h5py.File(PATH_HDF_LOCAL + 'magna_train_stdd.hdf')[tf_type]
	valid_x = h5py.File(PATH_HDF_LOCAL + 'magna_valid_stdd.hdf')[tf_type]
	test_x  = h5py.File(PATH_HDF_LOCAL + 'magna_test_stdd.hdf')[tf_type]

	return train_x, valid_x, test_x


def load_y(top_n=50):
	train_y = h5py.File(PATH_HDF_LOCAL + 'magna_train.hdf')['y'][:, :top_n]
	valid_y = h5py.File(PATH_HDF_LOCAL + 'magna_valid.hdf')['y'][:, :top_n]
	test_y  = h5py.File(PATH_HDF_LOCAL + 'magna_test.hdf')['y'][:, :top_n]

	return train_y, valid_y, test_y
"""
