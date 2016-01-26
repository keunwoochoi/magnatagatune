import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
It deals with 'data' (or 'x') only!
For labels (or 'y'), see main_prepare.y.py

It prepares stft and cqt representation.
It is recommended to rather use this file independetly than import -- because it's clearer!
"""

import os
import sys
import cPickle as cP
import numpy as np
import pdb
import librosa
import time
from multiprocessing import Pool
import h5py

from environments import *
from constants import *
import my_utils
import file_manager

def check_if_done(path):
	if os.path.exists(path):
		if os.path.getsize(path) != 0: 
			return True
	return False

def get_conventional_set():
	''' Get conventional 12:1:3 validation setting as other did.
	'''
	if os.path.exists(PATH_DATA + FILE_DICT['conventional_set_idxs']):
		return np.load(PATH_DATA + FILE_DICT['conventional_set_idxs'])
	print 'Will create conventional set'
	fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))
	train_idxs = []
	valid_idxs = []
	test_idxs = []
	train_pres = [str(ele) for ele in range(10)] + ['a', 'b']
	valid_pres = ['c']
	test_pres = ['d','e','f']
	for path_idx, path in enumerate(fm.paths):
		if path[0] in train_pres:
			train_idxs.append(path_idx)
		elif path[0] in valid_pres:
			valid_idxs.append(path_idx)
		elif path[0] in test_pres:
			test_idxs.append(path_idx)
		else:
			raise RuntimeError('Path seems strange: %d, %s' % (path_idx, path))
	np.save(PATH_DATA + FILE_DICT['conventional_set_idxs'], [train_idx, valid_idx, test_idx])
	print 'done done done.'
	return [train_idx, valid_idx, test_idx]

#------------------------------------------#

def create_hdf(set_indices=None):
	'''create hdf file that has cqt, stft, mfcc, melgram of train/valid/test set.
	For my original 6:1:1 (8-fold) cross-validation, input arg == None.
	Otherwise, specify indices, [list_of_train_id, list_of_val_id, list_of_test_id]
	'''
	
	fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))
	if set_indices == None:
		set_indices = fm.shuffle(n_fold=8) # train, valid, test indices
	set_name = ['train', 'valid', 'test']
	dataset_names = ['stft', 'melgram', 'cqt', 'mfcc']
	print '='*60
	print '====== create_hdf ======'
	print '='*60
	for set_name_idx, indices in enumerate(set_indices): # e.g. For Train set, it's (0, [random indices])
		# dataset file
		filename = 'magna_'+set_name[set_name_idx] + '.hdf'
		if os.path.exists(PATH_HDF_LOCAL + filename):
			file_write = h5py.File(PATH_HDF_LOCAL + filename, 'r+')
			print 'loading hdf file that exists already there.'
		else:
			file_write = h5py.File(PATH_HDF_LOCAL + filename, 'w')
			print 'creating new hdf file.'
		#
		num_datapoints = NUM_SEG * len(indices)
		print 'number of data point in %s is %d' % (filename, num_datapoints)
		# dataset name
		for dataset_name in dataset_names: # e.g. For cqt,
			test_tf = fm.load_file(file_type=dataset_name, clip_idx=0, seg_idx=0)


			tf_height = test_tf.shape[0]
			tf_width = test_tf.shape[1]
			if dataset_name in file_write:
				data_to_store = file_write[dataset_name]
			else:
				data_to_store = file_write.create_dataset(dataset_name, (num_datapoints, 1, tf_height, tf_width))
			# fill the dataset
			done_idx_file_path = PATH_HDF_LOCAL + filename + '_' +dataset_name + '_done_idx.npy'
			if os.path.exists(done_idx_file_path):
				done_idx = np.load(done_idx_file_path)
			else:
				done_idx = -1
			for write_idx, clip_idx in enumerate(indices): # e.g. For a clip, clip_idx is randomly permutted here. 
				if write_idx <= done_idx:
					continue
				for seg_idx in range(NUM_SEG):  # e.g. For a segment 
					tf_here = fm.load_file(file_type=dataset_name, clip_idx=clip_idx, seg_idx=seg_idx)
					if test_tf is None:
						os.remove('%s%d_%d.npy' % (PATH_TF['dataset_name'], fm.clip_ids[clip_idx], seg_idx))
						process_all_features((fm.clip_ids[clip_idx], fm.paths[clip_idx]))
						tf_here = fm.load_file(file_type=dataset_name, clip_idx=clip_idx, seg_idx=seg_idx)	
					try:
						data_to_store[write_idx + seg_idx*len(indices)] = tf_here
					except ValueError:
						pdb.set_trace()
				if write_idx % 100 == 0:
					print 'write-th clip at %d, %s, %s is done' % (write_idx, dataset_name, filename)
					np.save(done_idx_file_path, write_idx)
		print 'Done: %s, %s ' % (dataset_name, filename)

		dataset_name = 'y' # label
		done_idx_file_path = PATH_HDF_LOCAL + filename + '_' +dataset_name + '_done_idx.npy'
		label_matrix = np.load(PATH_DATA + FILE_DICT['sorted_merged_label_matrix'])
		if dataset_name in file_write:
			data_to_store = file_write[dataset_name]
			if data_to_store.shape != (num_datapoints, label_matrix.shape[1]):
				del file_write[dataset_name]
				data_to_store = file_write.create_dataset(dataset_name, (num_datapoints, label_matrix.shape[1]))
		else:
			data_to_store = file_write.create_dataset(dataset_name, (num_datapoints, label_matrix.shape[1]))
		# fill it.

		for write_idx, clip_idx in enumerate(indices): # e.g. For a clip, clip_idx is randomly permutted here. 
			if write_idx <= done_idx:
					continue
			for seg_idx in range(NUM_SEG):
				data_to_store[write_idx + seg_idx*len(indices)] = label_matrix[clip_idx,:]
			if write_idx % 100 == 0:
				np.save(done_idx_file_path, write_idx)
		print 'Labels are DONE as well! for %s' % dataset_name
	
	print 'HDF for train, valid, test and for all feature is created and stored.'
	print 'Now copy it from %s to c4dm server.' % PATH_HDF_LOCAL
	print '='*60


#------------------------------------------#

def prepare_y():

	# create a file manager
	if os.path.exists(PATH_DATA + FILE_DICT["file_manager"]):
		fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))
	else:
		fm = file_manager.File_Manager()
		fm.fill_from_csv()
		fm.create_label_matrix()
		cP.dump(fm, open(PATH_DATA + FILE_DICT["file_manager"], 'w'))

	# refine tags
	my_utils.refine_label_matrix()

#------------------------------------------#

def do_cqt(src, clip_id, seg_idx):
	if check_if_done('%s%d_%d.npy'%(PATH_CQT,clip_id,seg_idx)):
		return
	np.save('%s%d_%d.npy'%(PATH_CQT,clip_id,seg_idx) ,
				 librosa.logamplitude(librosa.cqt(y=src, 
												sr=SR, 
												hop_length=HOP_LEN, 
												bins_per_octave=BINS_PER_OCTAVE, 
												n_bins=N_CQT_BINS)**2, 
										ref_power=1.0))
	return

def do_melgram(src, clip_id, seg_idx):
	if check_if_done('%s%d_%d.npy'%(PATH_MELGRAM,clip_id,seg_idx)):
		return
	np.save('%s%d_%d.npy'%(PATH_MELGRAM,clip_id,seg_idx) ,
				 librosa.logamplitude(librosa.feature.melspectrogram(
				 								y=src, 
												sr=SR, 
												hop_length=HOP_LEN, 
												)**2, 
										ref_power=1.0))
	return

def do_stft(src, clip_id, seg_idx):
	if check_if_done('%s%d_%d.npy'%(PATH_STFT,clip_id,seg_idx)):
		return
	np.save('%s%d_%d.npy'%(PATH_STFT,clip_id,seg_idx) ,
				 librosa.logamplitude(np.abs(librosa.stft(
				 								y=src, 
												hop_length=HOP_LEN, 
												n_fft=N_FFT)
				 							)**2, 
										ref_power=1.0))
	return

def do_mfcc(src, clip_id, seg_idx):
	def augment_mfcc(mfcc):
		'''concatenate d-mfcc and dd-mfcc.
		mfcc: numpy 2d array.'''
		def get_derivative_mfcc(mfcc):
			'''return a same-sized, derivative of mfcc.'''
			len_freq, num_fr = mfcc.shape
			mfcc = np.hstack((np.zeros((len_freq, 1)), mfcc))
			return mfcc[:, 1:] - mfcc[:, :-1]
		d_mfcc = get_derivative_mfcc(mfcc)
		return np.vstack((mfcc, d_mfcc, get_derivative_mfcc(d_mfcc)))
	
	if check_if_done('%s%d_%d.npy'%(PATH_MFCC,clip_id,seg_idx)):
		return
	
	mfcc = librosa.feature.mfcc(y=src, 
								sr=SR, 
								n_mfcc=31,
								hop_length=HOP_LEN, 
								n_fft=N_FFT)
	mfcc = mfcc[1:, :] # remove the first one.

	np.save('%s%d_%d.npy'%(PATH_MFCC,clip_id,seg_idx), augment_mfcc(mfcc))			 							
										
	return
		
def process_all_features(args):
	''' args = (clip_id, mp3_path)
	'''
	clip_id, mp3_path = args # unpack
	if mp3_path == '':
		return
	# constants
	num_segments = NUM_SEG # 7
	len_segments = LEN_SEG # 4.0
	sp_per_seg = int(len_segments * SR)

	if os.path.exists(PATH_MAGNA + 'audio/' + mp3_path):
		try:
			src, sr = librosa.load(PATH_MAGNA + 'audio/' + mp3_path, sr=SR)
		except EOFError:
			print args
			pdb.set_trace()
	else:
		print 'NO mp3 for %d, %s' % (clip_id, mp3_path)
		return
	for seg_idx in range(NUM_SEG):
		src_here = src[seg_idx*sp_per_seg : (seg_idx+1)*sp_per_seg]
		do_mfcc(src_here, clip_id, seg_idx)
		do_melgram(src_here, clip_id, seg_idx)
		do_cqt(src_here, clip_id, seg_idx)
		do_stft(src_here, clip_id, seg_idx)

	print 'All features are done for all segments of clip_id:%d' % clip_id
	return

def prepare_x():
	'''It spawns process.
	'''
	fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))
	
	clip_ids_to_process = fm.clip_ids
	paths_to_process = fm.paths

	args = zip(clip_ids_to_process, paths_to_process)
	# print '%d file clips will be done' % len(clip_ids_to_process)
	# print 'process idx is %d' % process_idx
	# for idx, arg in enumerate(args):
	# 	if idx % 16 == process_idx:
	# 		process_all_features(arg)
	# 	else:
	# 		pass
	
	p = Pool(48)
	p.map(process_all_features, args)

	return
#------------------------------------------#
def standardise():
	'''load all hdf file and standardise them'''

	tfs = ['cqt', 'melgram', 'stft', 'mfcc']
	nb_subset = NUM_SEG
	f_train = h5py.File(PATH_HDF_LOCAL + 'magna_train.hdf','r')
	f_valid = h5py.File(PATH_HDF_LOCAL + 'magna_valid.hdf','r')
	f_test = h5py.File(PATH_HDF_LOCAL + 'magna_test.hdf','r')

	f_train_std = h5py.File(PATH_HDF_LOCAL + 'magna_train_stdd.hdf','w')
	f_valid_std = h5py.File(PATH_HDF_LOCAL + 'magna_valid_stdd.hdf','w')
	f_test_std = h5py.File(PATH_HDF_LOCAL + 'magna_test_stdd.hdf','w')	

	for tf in tfs:
		raw_data = f_train[tf][:30000]
		mean = np.mean(raw_data)
		std = np.std(raw_data)
		mean = np.mean([mean, np.mean(f_train[tf][30000:60000])])
		std = np.mean([std, np.std(f_train[tf][30000:60000])])
		
		print '%s, mean %f, std %f' % (tf, mean, std)
		
		write_train = f_train_std.create_dataset(tf, f_train[tf].shape)
		write_valid = f_valid_std.create_dataset(tf, f_valid[tf].shape)
		write_test = f_test_std.create_dataset(tf, f_test[tf].shape)
		
		for idx, sp in enumerate(f_train[tf]):
			write_train[idx] = (sp - mean) / std
		for idx, sp in enumerate(f_valid[tf]):
			write_valid[idx] = (sp - mean) / std
		for idx, sp in enumerate(f_test[tf]):
			write_test[idx] = (sp - mean) / std

	f_train.close()
	f_valid.close()
	f_test.close()

	f_train_std.close()
	f_valid_std.close()
	f_test_std.close()

	print 'standaridse - done.'


if __name__ == '__main__':
	'''
	First, remove things in data/ to restart. Then execute as follows:
	
	prepare_y()
	prepare_x()
	create_hdf()
	standardise()
	'''
	# prepare_y()
	# prepare_x()
	list_of_list = get_conventional_set()
	create_hdf(list_of_list)
	# standardise()
	
