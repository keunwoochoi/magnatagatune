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
import pprint
from random import shuffle

from environments import *
from constants import *
import my_utils
import file_manager


def get_permutation(num):
	'''get list shuffled numbers. load from npy if exists '''
	permutation_file = 'permutation_for_all_%d.npy' % num
	if os.path.exists(PATH_DATA + permutation_file):
		permutation_list = np.load(PATH_DATA+permutation_file)
	else:
		permutation_list = np.random.permutation(num)
		np.save(PATH_DATA+permutation_file, permutation_list)
	return permutation_list


def check_if_done(path):
	'''check if file at the path exists. 
	also check if the size of file is not zero.
	'''
	if os.path.exists(path):
		if os.path.getsize(path) != 0: 
			return True
	return False

def get_start_end_points(seg_idx, sp_per_seg):
		'''
		adding +1 is heuristic to make it equal if it's done by frame...
		so if it's about sample, use it.
		if it's about TF frames, add +1 at the 'frame_to' value.
		'''
		if seg_idx < 7:
			return seg_idx*sp_per_seg, (seg_idx+1)*sp_per_seg
		else:
			return int((seg_idx-6.5)*sp_per_seg), int((seg_idx-5.5)*sp_per_seg)

def get_conventional_set():
	''' Get conventional 12:1:3 validation setting as other did.
	returns three lists, all of which consists of indices of songs 
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
	np.save(PATH_DATA + FILE_DICT['conventional_set_idxs'], [train_idxs, valid_idxs, test_idxs])
	print 'done done done.'
	return [train_idxs, valid_idxs, test_idxs]

#------------------------------------------#
def process_hdf(set_name_idx):
	'''sub process that will be spawned'''

	fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))
	
	label_matrices = {}
	label_matrices['y_original'] = np.load(PATH_DATA + FILE_DICT['sorted_label_matrix'])
	label_matrices['y_merged'] = np.load(PATH_DATA + FILE_DICT['sorted_merged_label_matrix'])

	set_names = [str(ele) for ele in range(16)] # ['0','1','2','3',..'15']
	folder_names = set_names[:10] + ['a','b','c','d','e','f']
	dataset_names = ['cqt', 'stft', 'melgram', 'mfcc']
	dataset_label_names=['y_merged','y_original']

	print '='*60
	print '====== prepare_hdf, %d ======' % set_name_idx
	print '='*60
	means = {'cqt':-69.8194, 'melgram':-15.5739, 'stft':-24.2885, 'mfcc':1.14238}
	stds  = {'cqt':16.7193,  'melgram':21.1379,  'stft':20.6936, 'mfcc':18.7942}
	set_name = set_names[set_name_idx]
	filename = 'magna_%s.hdf' % set_name 
	if os.path.exists(PATH_HDF_LOCAL + filename): # read+ or create hdf file.
		file_write = h5py.File(PATH_HDF_LOCAL + filename, 'r+')
	else:
		file_write = h5py.File(PATH_HDF_LOCAL + filename, 'w')

	# x: get number of data by file_manager path imformation.
	num_datapoints = NUM_SEG*len([path for path in fm.paths if path[0] == folder_names[set_name_idx]])
	for dataset_name in dataset_names: # e.g. 'cqt', 'stft',..
		if not dataset_name in file_write: # create dataset 
			test_tf = fm.load_file(file_type=dataset_name, clip_idx=0, seg_idx=0)
			tf_height, tf_width = test_tf.shape
			file_write.create_dataset(dataset_name, (num_datapoints, 1, tf_height, tf_width))
	# y: get number of data by file_manager path imformation.
	for dataset_label_name in dataset_label_names:
		if not dataset_label_name in file_write:
			file_write.create_dataset(dataset_label_name, (num_datapoints, label_matrices[dataset_label_name].shape[1]))
	
	file_write_idx = set_name_idx

	# load files and put them into corresponding hdf files.
	folder_name = folder_names[file_write_idx]
	#paths_in = [path for path in fm.paths if path[0] == folder_name]
	clip_ids = [clip_id for clip_id in fm.clip_ids if fm.id_to_paths[str(clip_id)][0] == folder_name] # [2,6,...
	shuffle(clip_ids)
	print '  paths_in[0]: %s' % fm.id_to_paths[str(clip_ids[0])]
	print '  paths_in[-1]: %s' % fm.id_to_paths[str(clip_ids[-1])]
	print '  len clip_ids: %d' % len(clip_ids)
	# for data (x)
	for dataset_name in dataset_names: # e.g. 'cqt', 'stft',..
		print '    process %s' % dataset_name
		data_to_store = file_write[dataset_name]
		print '    size of dataset: ', data_to_store.shape
		for write_idx, clip_id in enumerate(clip_ids): # shuffled clip ids for this folder.--> no, it's not..?
			clip_idx = fm.id_to_idx[str(clip_id)]
			for seg_idx in range(NUM_SEG):
				tf_here = fm.load_file(file_type=dataset_name, clip_idx=clip_idx, seg_idx=seg_idx)
				try:
					data_to_store[write_idx + seg_idx*len(clip_ids)] = (tf_here - means[dataset_name])/stds[dataset_name]
				except TypeError:
					process_all_features((clip_id, fm.paths(clip_idx), True))
					tf_here = fm.load_file(file_type=dataset_name, clip_idx=clip_idx, seg_idx=seg_idx)
					data_to_store[write_idx + seg_idx*len(clip_ids)] = (tf_here - means[dataset_name])/stds[dataset_name]
					# raise RuntimeError('Error on loaded tf:%s, clip_idx:%d, seg_idx:%d'%(dataset_name, clip_idx,seg_idx))
					
	# for labels (y)
	for dataset_label_name in dataset_label_names:
		print '    process %s' % dataset_label_name
		data_to_store = file_write[dataset_label_name]
		print '    size: ', data_to_store.shape
		for write_idx, clip_id in enumerate(clip_ids): # shuffled clip ids for this folder.--> no, too. 
			clip_idx = fm.id_to_idx[str(clip_id)]
			for seg_idx in range(NUM_SEG):
				data_to_store[write_idx + seg_idx*len(clip_ids)] = label_matrices[dataset_label_name][clip_idx,:]

	print 'Labels are done as well! for %d/%d' %(file_write_idx, len(set_names)) 

def prepare_hdf():
	'''create hdf file that has cqt, stft, mfcc, melgram of 16 sets in MagnaTatATune.
	This function includes standardisation of tf values, but not shuffling.
	'''
	
	# for set_name_idx, set_name in enumerate(set_names): # for every folder,
	# 	process_hdf(set_name_idx)
	p = Pool(16)
	set_name_indices = range(16)
	p.map(process_hdf, set_name_indices[9:])
	
	print 'ALL DONE.'
	print 'Now shuffle and copy it from %s to c4dm server.' % PATH_HDF_LOCAL
	print '='*60


#------------------------------------------#
def get_LDA(X, num_components=10, show_topics=True):
	''' Latent Dirichlet Allication by NMF.
	21 Nov 2015, Keunwoo Choi

	LDA for a song-tag matrix. The motivation is same as get_LSI. 
	With NMF, it is easier to explain what each topic represent - by inspecting 'H' matrix,
	where X ~= X' = W*H as a result of NMF. 
	It is also good to have non-negative elements, straight-forward for both W and H.

	'''

	from sklearn.decomposition import NMF
	if X == None:
		print 'X is omitted, so just assume it is the mood tag mtx w audio.'
		X = np.load(PATH_DATA + FILE_DICT["mood_tags_matrix"]) #np matrix, 9320-by-100

	nmf = NMF(init='nndsvd', n_components=num_components, max_iter=400) # 400 is too large, but it doesn't hurt.
	W = nmf.fit_transform(X)
	H = nmf.components_
	print '='*60
	print "NMF done with k=%d, average error:%2.4f" % (num_components, nmf.reconstruction_err_/(X.shape[0]*X.shape[1]))

	term_rankings = []
	moodnames = cP.load(open(PATH_DATA + FILE_DICT['sorted_tags'], 'r')) #list, 100
	for topic_index in range( H.shape[0] ):
		top_indices = np.argsort( H[topic_index,:] )[::-1][0:10]
		term_ranking = [moodnames[i] for i in top_indices]
		term_rankings.append(term_ranking)
		if show_topics:	
			print "Topic %d: %s" % ( topic_index, ", ".join( term_ranking ) )
	print '='*60
	cP.dump(term_rankings, open(PATH_DATA + ('topics_strings_%d_components.cP' % num_components), 'w'))
	for row_idx, row in enumerate(W):
		if np.max(row) != 0:
			W[row_idx] = row / np.max(row)
	return W / np.max(W) # return normalised matrix, [0, 1]
	''''''


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

	# also, create sorted label matrix.	
	label_matrix = np.load(PATH_DATA + FILE_DICT['label_matrix'])
	sum_labels = np.sum(label_matrix, axis=0)
	sort_args = np.argsort(sum_labels[::-1])
	sorted_label_matrix = np.zeros(label_matrix.shape, dtype=np.int)

	for read_idx, write_idx in enumerate(sort_args):
		sorted_label_matrix[:, write_idx] = label_matrix[:, read_idx]

	np.save(PATH_DATA + FILE_DICT['sorted_label_matrix'], sorted_label_matrix)
	# and LDA-50 version of sorted label matrix.
#------------------------------------------#

def do_cqt(src, clip_id, seg_idx):
	'''see do_mfcc'''
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
	'''see do_mfcc'''
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
	'''see do_mfcc'''	
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

def get_mfcc(src):
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
	mfcc = librosa.feature.mfcc(y=src, 
								sr=SR, 
								n_mfcc=31,
								hop_length=HOP_LEN, 
								n_fft=N_FFT)
	mfcc = mfcc[1:, :] # remove the first one.
	return augment_mfcc(mfcc)

def do_mfcc(src, clip_id, seg_idx):
	'''src: audio samples of a segment.
	clip_id: (usually) integer clip id, an element of file_manager.clip_ids
	seg_idx: integer segment index, use to be range(7). now in range(13). 
	          range(7): samples from [4s x seg_idx] to [4s x seg_idx++]
	          range(7,13): samples from [2s + 4sx(seg_idx-6)] to .. --> 2-second overlaps
	'''	
	if check_if_done('%s%d_%d.npy'%(PATH_MFCC,clip_id,seg_idx)):
		return

	np.save('%s%d_%d.npy'%(PATH_MFCC,clip_id,seg_idx), get_mfcc(src))			 							
										
	return

		

def process_all_features(args):
	''' args = (clip_id, mp3_path, force)
	force: boolean to force or not
	'''
	def get_tf_representation(src, tf_type):
		if tf_type == 'cqt':
			return librosa.logamplitude(librosa.cqt(y=src, 
													sr=SR, 
													hop_length=HOP_LEN, 
													bins_per_octave=BINS_PER_OCTAVE, 
													n_bins=N_CQT_BINS)**2, 
										ref_power=1.0)
		elif tf_type == 'stft':
			return librosa.logamplitude(np.abs(librosa.stft(y=src, 
															hop_length=HOP_LEN, 
															n_fft=N_FFT))**2, 
										ref_power=1.0)
		elif tf_type == 'mfcc':
			return get_mfcc(src)
		elif tf_type == 'melgram':
			return librosa.logamplitude(librosa.feature.melspectrogram(y=src, 
																		sr=SR, 
																		hop_length=HOP_LEN)**2, 
										ref_power=1.0)
		else:
			raise RuntimeError('Wrong tf type I guess: %s' % tf_type)

	def check_if_they_are_done(clip_id, path):	
		ret = True
		for seg_idx in range(NUM_SEG):
			ret = ret * check_if_done('%s%d_%d.npy'%(path, clip_id, seg_idx))
			if ret == False:
				return ret
		return ret
	
	''''''
	clip_id, mp3_path, force = args # unpack
	if mp3_path == '':
		return
	tf_types = ['cqt', 'mfcc', 'melgram', 'stft']
	paths = [PATH_CQT, PATH_MFCC, PATH_MELGRAM, PATH_STFT]
	for tf_type, path in zip(tf_types, paths):
		if not force:
			if check_if_they_are_done(clip_id, path):
				#  print '  clip_id:%d, everything is done for %s' % (clip_id, tf_type)
				continue
		try:
			src_full, sr = librosa.load(PATH_MAGNA + 'audio/' + mp3_path, sr=SR)
		except:
			print 'AudioRead error', path, tf_type, mp3_path, clip_id
			raise RuntimeError("STOP!")

		for seg_idx in range(NUM_SEG):
			full_filepath_out = '%s%d_%d.npy'%(path,clip_id,seg_idx)
			if not force:
				if check_if_done(full_filepath_out):
					# print '  -- clip_id:%d, tf_type:%s, seg_idx:%d done already' % (clip_id, tf_type, seg_idx)	
					continue

			SRC_full = get_tf_representation(src_full, tf_type)
			fr_from, fr_to = get_start_end_points(seg_idx, int(4*FRAMES_PER_SEC))
			fr_to = fr_to + 1 # to make it 251 rathe rthan 250. 

			np.save(full_filepath_out, SRC_full[:, fr_from:fr_to])
			
	print '  -- ALL done : clip_id:%d' % (clip_id)
	return

	# for all types of tf-representation:
	# check if THEY are done first. (check all of the output files.)
	#    get tf_representation for the whole signal (29.1 seconds).
	#    for every sub_segment:
	#        get sp_from:sp_to,
	#        get segments and save them (check_if_done first! ). 
	# 		

	'''
	# constants
	num_segments = NUM_SEG # 7
	len_segments = LEN_SEG # 4.0
	sp_per_seg = int(len_segments * SR)
	 it is already done in file_manager
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
		sp_from, sp_to = get_start_end_samples(seg_idx, sp_per_seg)

		src_here = src[sp_from : sp_to]
		do_mfcc(src_here, clip_id, seg_idx)
		do_melgram(src_here, clip_id, seg_idx)
		do_cqt(src_here, clip_id, seg_idx)
		do_stft(src_here, clip_id, seg_idx)

	print 'All features are done for all segments of clip_id:%d' % clip_id
	return
	'''

def prepare_x(num_pc=None, idx_pc=None):
	'''It spawns process to generate all numpy files for all songs.
	It does NOT do something with HDF.
	num_pc : number of pc using
	idx_pc : idx of pc
	'''
	if not num_pc and not idx_pc:
		num_pc = 1
		idx_pc = 0
	print 'num_pc:%d, idx_pc:%d' % (num_pc, idx_pc)
	fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))
	
	clip_ids_to_process = fm.clip_ids
	paths_to_process = fm.paths

	args = zip(clip_ids_to_process, paths_to_process)
	force = False
	args = [ele + (force,) for idx,ele in enumerate(args) if idx%num_pc == idx_pc]
	

	p = Pool(48)
	p.map(process_all_features, args)

	return
"""
ALL shuffle should be done thoroughly. 

def shuffle_hdf_process(set_idx):
	''''''
	print 'Start shuffle hdf process: %d' % set_idx
	dataset_names = ['cqt', 'stft', 'melgram', 'mfcc','y_merged', 'y_original']
	filename_hdf = 'magna_%d.hdf' % set_idx

	f = h5py.File(PATH_HDF_LOCAL+filename_hdf, 'r+')
	num_datapoints = f['cqt'].shape[0]
	num_clips = num_datapoints / NUM_SEG

	print '%d. total data point:%d, num_clip:%d. and this should zero.-->%d' % (set_idx, num_datapoints, num_clips, num_datapoints%num_clips)
	permutation_file = 'permutation_%d_%d.npy' % (set_idx, num_clips)
	
	if os.path.exists(PATH_DATA + permutation_file):
		permutation_list = np.load(PATH_DATA+permutation_file)
	else:
		permutation_list = np.random.permutation(num_clips)
		np.save(PATH_DATA+permutation_file, permutation_list)

	if 'shuffled' in f.attrs:
		if f.attrs['shuffled'] == True:
			print "it is already shuffled, %d set" % set_idx
			return
	else:
		f.attrs.create(name='shuffled', data=0.0, dtype=np.bool)
		print 'create shuffled value'	
	f.attrs.create(name='permutation_list', data=permutation_list)

	for dataset_name in dataset_names:
		temp_shuffled = []
		for seg_idx in range(NUM_SEG):
			shuffled_minibatch = [f[dataset_name][seg_idx*num_clips + permutation_list[i]] for i in xrange(num_clips)]
			temp_shuffled = temp_shuffled + shuffled_minibatch
		temp_shuffled = np.array(temp_shuffled)
		print 'shuffling done; ', f[dataset_name].shape, temp_before_shuffleded.shape
		f[dataset_name][:] = temp_shuffled
	f.attrs['shuffled'] = True
	f.close()
	return

def shuffle_hdfs():
	'''
	shuffle magna_0.hdf - magna_15.hdf
	and save the permutation.
	'''
	for i in range(16):
		shuffle_hdf_process(i)
		print 'shuffle done for %d' % i
	return
"""
def merge_shuffle_save_hdfs(file_read_ptrs, file_write):
	'''input: h5py file objects to read,
		      h5py file object to write, usually a temporary one.
	'''
	dataset_names = file_read_ptrs[0].keys()
	
	num_datapoints = sum([f[dataset_names[0]].shape[0] for f in file_read_ptrs])
	num_clips = num_datapoints/NUM_SEG
	# get permutation
	permutation_list = get_permutation(num_clips)
	
	# read and merge into a temp file
	for dataset_name in dataset_names:
		print '  dataset name: %s size of %d' % (dataset_name, num_datapoints)
		shape_write = (num_datapoints,) +  file_read_ptrs[0][dataset_name].shape[1:]
		# create a single and BIG temp hdf file
		file_temp = h5py.File(PATH_HDF_LOCAL + 'magna_temporary.hdf', 'w')
		file_temp.create_dataset(dataset_name, shape_write)
		# put everything into the temp hdf file.
		write_idx = 0
		for seg_idx in range(NUM_SEG):
			print '    seg index: %d/%d' % (seg_idx, NUM_SEG)
			for file_read in file_read_ptrs:
				num_clips_to_add = file_read[dataset_name].shape[0]/NUM_SEG

				data_from = num_clips_to_add*seg_idx
				data_to   = num_clips_to_add*(seg_idx+1)

				file_temp[dataset_name][write_idx:write_idx+num_clips_to_add] = file_read[dataset_name][data_from:data_to]
				write_idx += num_clips_to_add
		# in temp_before_shuffled is concatenated of all data, but sorted by segments 
		# [songs of segment 0][songs of segment 1]....[songs of segment 6]
		# write it.
		file_write.create_dataset(dataset_name, shape_write)
		
		print '  shuffle it - and write it per segment.'
		write_idx = 0
		for seg_idx in range(NUM_SEG):
			shuffled_minibatch = [file_temp[dataset_name][seg_idx*num_clips + permutation_list[i]] for i in xrange(num_clips)]
			num_data_added = len(shuffled_minibatch)
			file_write[dataset_name][write_idx:write_idx+num_data_added] = np.array(shuffled_minibatch)
			write_idx += num_data_added
		print '  shuffle done for %s .' % (dataset_name)

		file_temp.close()
		os.remove(PATH_HDF_LOCAL + 'magna_temporary.hdf')
		
	print 'All done.'

"""
def merge_shuffle_train_hdfs():
	'''
	train set: 0-11 (12 sets)
	shuffle within a folder.
	'''
	train_filenames = ['magna_%d.hdf'%idx for idx in range(12)]
	file_read_ptrs = [h5py.File(PATH_HDF_LOCAL+train_filenames[i], 'r') for i in range(12)]
	file_write = h5py.File(PATH_HDF_LOCAL+'magna_train_12set.hdf', 'w')
	file_temp = h5py.File(PATH_HDF_LOCAL+'magna_train_12set_temporary.hdf', 'w')
	dataset_names = ['cqt', 'stft', 'melgram', 'mfcc','y_merged', 'y_original']

	num_datapoints = sum([f['cqt'].shape[0] for f in file_read_ptrs])
	num_clips = num_datapoints/NUM_SEG
	# get permutation
	permutation_file = 'permutation_for_all_%d.npy' % num_clips
	if os.path.exists(PATH_DATA + permutation_file):
		permutation_list = np.load(PATH_DATA+permutation_file)
	else:
		permutation_list = np.random.permutation(num_clips)
		np.save(PATH_DATA+permutation_file, permutation_list)
	print 'will do some work now.'
	# do the work.
	for dataset_name in dataset_names:
		print '  dataset name: %s' % dataset_name
		shape_write = (num_datapoints,) +  file_read_ptrs[0][dataset_name].shape[1:]
		file_temp.create_dataset(dataset_name, shape_write)
		# temp_before_shuffled = np.zeros(shape_write)
		write_idx = 0
		for seg_idx in range(NUM_SEG):
			print '    seg index: %d/7' % seg_idx
			for file_read in file_read_ptrs:
				num_clips_to_add = file_read[dataset_name].shape[0]/7

				data_from = num_clips_to_add*seg_idx
				data_to   = num_clips_to_add*(seg_idx+1)

				file_temp[dataset_name][write_idx:write_idx+num_clips_to_add] = file_read[dataset_name][data_from:data_to]
				write_idx += num_clips_to_add
		# in temp_before_shuffled is concatenated of all data, but sorted by segments 
		# [songs of segment 0][songs of segment 1]....[songs of segment 6]
		# now it's in the temp_before_shuffled
		
		# write it.
		file_write.create_dataset(dataset_name, shape_write)
		
		print '  shuffle it - and write it per segment.'
		write_idx = 0
		for seg_idx in range(NUM_SEG):
			shuffled_minibatch = [file_temp[dataset_name][seg_idx*num_clips + permutation_list[i]] for i in xrange(num_clips)]
			num_data_added = len(shuffled_minibatch)
			file_write[dataset_name][write_idx:write_idx+num_data_added] = np.array(shuffled_minibatch)
			write_idx += num_data_added
		print '  shuffle done.'
		
		print '  merge Done: %s' % dataset_name
	file_write.close()
	print 'All done.'
"""

def prepare_divide_merge_shuffle_per_set():
	'''shuffling within folder was not enough,
	so shuffle it across folders, per sets (train, valid, test)'''
	# train_filenames = ['magna_shuffled_%d.hdf'%idx for idx in range(12)]
	# dataset_names = h5py.File(PATH_HDF_LOCAL + 'magna_0.hdf', 'r').keys()

	sets_numbers = [range(12), [12], [13,14,15]] # number of sets of train/valid/data.
	for set_nums_idx, set_nums in enumerate(sets_numbers): # trains, valids, tests
		
		if set_nums_idx == 0: # because it;s done. 
			continue

		print '#'*50
		print 'set nums:', set_nums
		print '#'*50
		num_datapoints_total = 0
		num_datapoints_sets = []
		file_read_ptrs = []
		for set_num in set_nums: # each folder in this set.
			f = h5py.File(PATH_HDF_LOCAL + 'magna_%d.hdf' % set_num, 'r')
			file_read_ptrs.append(f)
			num_datapoints_total += f['melgram'].shape[0]
			num_datapoints_sets.append(f['melgram'].shape[0])
		
		num_datapoints_each = num_datapoints_total / len(set_nums)
		
		f_read_example = f
		shuffled_idx_list = get_permutation(num_datapoints_total / NUM_SEG)

		dataset_names = f_read_example.keys()

		# make a merged set for temporary. (also freq normalised)
		

		f_merged = h5py.File(PATH_HDF_LOCAL + 'magna_temp_merged.hdf', 'w')

		# shuffle everything into a temp file.
		merge_shuffle_save_hdfs(file_read_ptrs, f_merged)
		
		# freq-based normalisation..?

		# put them into each, new (shuffled) set.
		for set_idx, set_num in enumerate(set_nums): # each folder in this set.
			# f = h5py.File(PATH_HDF_LOCAL + 'magna_shuffled_%d.hdf' % set_num, 'w')
			filename_out = 'magna_shuffled_%d.hdf' % set_num

			print '  - write idx:%d, %s, datapoints_each:%d/%d' % (set_num, filename_out, num_datapoints_each, num_datapoints_total)
			
			f_write = h5py.File(PATH_HDF_LOCAL + filename_out, 'w')
			
			for dataset_name in dataset_names:
				print '  -- dataset_name:%s' % dataset_name
				shape_write = (num_datapoints_each,) +  f_read_example[dataset_name].shape[1:]
				f_write.create_dataset(dataset_name, shape_write)
				print '    pick up data, [%d:%d] ' % (set_idx*num_datapoints_each, (set_idx+1)*num_datapoints_each)
				
				f_write[dataset_name][:] = f_merged[dataset_name][set_idx*num_datapoints_each: (set_idx+1)*num_datapoints_each]

			print 'done:%d, %s' % (set_idx, dataset_name)
			f_write.close()
		os.remove(PATH_HDF_LOCAL + 'magna_temp_merged.hdf')

	print 'ALL DONE: shuffle and merge'

	return

def freq_normalise_dataset(hdf_path):
	'''load hdf path, which has datasets, and do the work - normalize for each frequency.'''
	f = h5py.File(hdf_path, 'r+')
	dataset_names = ['melgram', 'stft', 'cqt', 'mfcc']
	print '-'*40
	print 'start normalise for %s ' % hdf_path
	for dataset_name in dataset_names:
		if dataset_name not in f:
			continue
		# dataset = f[dataset_name]
		freq_mean = np.mean(np.mean(np.mean(f[dataset_name], axis=0), axis=0), axis=1)
		freq_mean = freq_mean.reshape((1,1,-1,1))

		freq_var  = np.mean(np.mean(np.var(f[dataset_name], axis=0), axis=0), axis=1)
		freq_var  = freq_var.reshape((1,1,-1,1))
		f[dataset_name][:] = (f[dataset_name] - freq_mean) / np.sqrt(freq_var)
		print '%s, %s: done.' % (hdf_path, dataset_name)
	return

def freq_normalise_all():
	hdf_files = os.listdir(PATH_HDF_LOCAL)
	hdf_paths = [PATH_HDF_LOCAL + filename for filename in hdf_files if filename.split('.')[-1] == 'hdf']
	print 'paths are:'
	pprint.pprint(hdf_paths)

	for hdf_path in hdf_paths:
		freq_normalise_dataset(hdf_path)

	print 'ALL DONE'
	return

def get_tags_list():
	'''sorted tags for merged/not merged'''
	return



if __name__ == '__main__':
	'''
	First, remove things in data/ to restart. Then execute as follows:
	
	python main_prepare.py 2 0 # in pc 1
	python main_prepare.py 2 1 # in pc 2


	prepare_y()
	prepare_x()
	prepare_hdf()
	standardise()
	'''
	# if sys.argv < 3:
	# 	prepare_x()
	# else:
	# 	num_pc = int(sys.argv[1])
	# 	idx_pc = int(sys.argv[2])
	# 	prepare_x(num_pc, idx_pc)
	mtx = np.load(PATH_DATA + 'sorted_label_matrix.npy')

	reduced_mtx = get_LDA(mtx, num_components=50, show_topics=True)
	np.save(PATH_DATA + 'LDA_50_label_matrix.npy', reduced_mtx)

	sys.exit()
	prepare_y()
	prepare_x()
	prepare_hdf() # put numpy files into hdf without shuffling
	prepare_divide_merge_shuffle_per_set() # shuffles within each set (training/valid/test)
	
