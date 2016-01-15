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

from environments import *
from constants import *
import my_utils
import file_manager

def check_if_done(path):
	if os.path.exists(path):
		if os.path.getsize(path) != 0: 
			return True
	return False
#------------------------------------------#

def create_hdf():
	'''create hdf file that has cqt, stft, mfcc, melgram of train/valid/test set.'''
	fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))
	set_indices = fm.shuffle(n_fold=8) # train, valid, test indices
	set_name = ['train', 'valid', 'test']
	dataset_names = ['stft', 'melgram', 'cqt', 'mfcc']
	print '='*60
	print '====== create_hdf ======'
	print '='*60
	for idx, indices in enumerate(set_indices): # e.g. For Train set, 
		# dataset file
		filename = 'magna_'+set_name[idx] + '.hdf'
		if os.path.exists(PATH_HDF_LOCAL + filename):
			file_write = h5py.File(PATH_HDF_LOCAL + filename, 'r+')
			print 'loading hdf file that exists already there.'
		else:
			file_write = h5py.File(PATH_HDF_LOCAL + filename, 'w')
			print 'creating new hdf file.'
		#
		num_datapoints = NUM_SEG * len(indices)
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
					data_to_store[write_idx + seg_idx*num_datapoints] = tf_here
				if write_idx % 100 == 0:
					print '%d-th clip at %s, %s is done' % (clip_idx, dataset_name, filename)
			np.save(done_idx_file_path, write_idx)
		print 'Done: %s, %s ' % (dataset_name, filename)

		dataset_name = 'y' # label
		label_matrix = np.load(PATH_DATA + FILE_DICT['sorted_merged_label_matrix'])
		if dataset_name in file_write:
			data_to_store = file_write[dataset_name]
		else:
			data_to_store = file_write.create_dataset(dataset_name, (num_datapoints, fm.num_tags))
		# fill it.
		for write_idx, clip_idx in enumerate(indices): # e.g. For a clip, clip_idx is randomly permutted here. 
			for seg_idx in range(NUM_SEG):
				data_to_store[write_idx + seg_idx*num_datapoints] = label_matrix[clip_idx,:]
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

	src, sr = librosa.load(PATH_MAGNA + 'audio/' + mp3_path, sr=SR)
	for seg_idx in range(NUM_SEG):
		src_here = src[seg_idx*sp_per_seg : (seg_idx+1)*sp_per_seg]
		do_mfcc(src_here, clip_id, seg_idx)
		do_melgram(src_here, clip_id, seg_idx)
		do_cqt(src_here, clip_id, seg_idx)
		do_stft(src_here, clip_id, seg_idx)

	print 'All features are done for all segments of clip_id:%d' % clip_id
	return

def prepare_x():
	'''It spawns process'''
	fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))
	idx_to_process = [idx for idx in xrange(fm.num_songs) if idx not in fm.idx_no_audio]

	clip_ids_to_process = [fm.clip_ids[idx] for idx in idx_to_process]
	paths_to_process = [fm.paths[idx] for idx in idx_to_process]
	print len(clip_ids_to_process)

	args = zip(clip_ids_to_process, paths_to_process)
	
	# for arg in args:
	# 	process_all_features(arg)

	p = Pool(48)
	p.map(process_all_features, args)

	return
#------------------------------------------#

if __name__ == '__main__':
	'''
	Remove things in data/ to start over, and then,
	
	prepare_y()
	prepare_x()
	create_hdf()
	
	'''

	prepare_y()
	prepare_x()
	create_hdf()
	