from environments import *
from constants import *
import os
import cPickle as cP
import time
import sys
import numpy as np
import h5py
import pdb

class File_Manager():
	def __init__(self):
		self.clip_ids = [] # integer. sequence aligned. valid (i.e. mp3 exists) only
		self.clip_ids_no_audio = [] # invalids are appended here.
		self.paths = [] #string. sequence aligned.
		self.idx_permutation = []
		self.id_permutation = []
		self.id_to_paths = {}
		self.feat_to_paths = {} # string : list, e.g. 'cqt' : blah..
		self.id_to_idx = {}
		# self.np_whole_label_matrix = None # will be numpy array
		self.num_songs = 0
		self.num_tags = 0
		self.tags = []
	
	def fill_from_csv(self):
		''' This function fills
		- self.clip_ids
		- self.idx_no_audio
		- self.paths
		- self.id_to_paths
		- self.id_to_idx
		- self.id_to_paths
		- self.num_songs
		- self.num_tags
		- self.tags
		
		'''
		with open(PATH_MAGNA+'clip_info_final.csv', 'r') as f:
			labels = f.readline() # first line is labels.
			write_idx = 0
			for line_idx, line in enumerate(f): # line_idx == read_idx
				if line_idx % 5000 == 0:
					print 'Line idx : %d loaded.' % line_idx
				values = [value.rstrip('\r\n').strip('"') for value in line.split('\t')]
				# [0]:clip_id, track_no (in album), title, artist, album, url, seg_start, seg_end, original_url, [9]:mp3_path
				
				if values[9] == '':
				 	self.clip_ids_no_audio.append(values[0])
				elif not os.path.exists(PATH_MAGNA + 'audio/' + values[9]):
				 	self.clip_ids_no_audio.append(values[0])
				else:
					self.clip_ids.append(int(values[0]))
					self.paths.append(values[9])
					self.id_to_paths[values[0]] = values[9]
					self.id_to_idx[values[0]] = write_idx
					write_idx += 1
				
			print 'All info from clip_info_final.csv loaded.'
		self.num_songs = len(self.clip_ids)
		
		print 'Now will identify number of tags'
		with open(PATH_MAGNA + 'annotations_final.csv', 'r') as f:
			tag_names = f.readline() # clip_id, 188 tags, mp3_path
			tags = [value.rstrip('\r\n').strip('"') for value in tag_names.split('\t')]
			tags = tags[1:-1]
			self.num_tags  = len(tags)
			self.tags = tags
		return
	
	def load_label_matrix(self):
		if os.path.exists(PATH_DATA + FILE_DICT['label_matrix']):
			return np.load(PATH_DATA + FILE_DICT['label_matrix'])
		
		return self.create_label_matrix()

	def create_label_matrix(self):
		print 'Now will read annotations_final.csv'
		if self.num_songs == 0 or self.num_tags ==0:
			raise RuntimeError('self.num_songs:%d, self.num_tags:%d, stop here. Execute self.fill_from_csv() first.' % (self.num_songs, self.num_tags))
			return
		with open(PATH_MAGNA + 'annotations_final.csv', 'r') as f:
			tag_names = f.readline() # clip_id, 188 tags, mp3_path
			label_matrix = np.zeros((self.num_songs, self.num_tags), dtype=np.bool_)
			for line_idx, line in enumerate(f):
				if line_idx % 2000 == 0:
					print 'Line idx : %d loaded.' % line_idx
				values = [value.rstrip('\r\n').strip('"') for value in line.split('\t')]
				if values[0] in self.clip_ids_no_audio:
					pass
				else:
					labels = [int(ele) for ele in values[1:-1]]
					label_matrix[self.id_to_idx[values[0]], : ] = np.array(labels, dtype=np.int)
				

		print 'label matrix created'
		np.save(PATH_DATA + FILE_DICT['label_matrix'], label_matrix)
		return label_matrix

	def shuffle(self, n_fold=8):
		'''load permutation file if exists, create otherwise.
		Then get random permutations of range(self.num_songs).
		...which will be done until I get a balanced train, valid, and test sets.
		Then get shuffled array of song_ids.'''

		'''For temporary, however, simple permutation will be used.  '''
		rand_filename = PATH_DATA +("balanced_sets_%d_%d.npy" % (n_folds, self.num_songs))
		if os.path.exists(rand_filename):
			print 'File manager will use a previously made random permutation file'

			train_idx, valid_idx, test_idx = np.load(rand_filename)
		else:
			print 'File manager will use a new random permutation file'
			rand_inds = np.random.permutation(self.filenum)
			num_valid = self.num/n_fold

			train = rand_inds[:num_valid*(n_fold-2)]
			valid = rand_inds[num_valid*(n_fold-2):num_valid*(n_fold-1)]
			test  = rand_inds[num_valid*(n_fold-1):]

			np.save(rand_filename, [train, valid, test])
		return train, valid, test

 	def load_file(self, file_type, clip_idx, seg_idx):
 		'''for file tyle (cqt, stft, mel,..) 
 		file_type : string, 'cqt', 'stft',..
 		clip_idx  : integer. not ID, just number from 0 to self.num_songs
 		seg_idx   : integer in range(7): 
 		
 		return: corresponding numpy array, 2d, for 4-seconds.
 		'''
 		if file_type.lower() == 'cqt':
 			path = PATH_CQT
 		elif file_type.lower() == 'stft':
 			path = PATH_STFT
 		elif file_type.lower() == 'mfcc':
 			path = PATH_MFCC
 		elif file_type.lower() == 'melgram':
 			path = PATH_MELGRAM
 		else:
 			print 'wrong file type in filer_manager.fload_file'

 		return np.load('%s%d_%d.npy'%(path, self.clip_ids[clip_idx], seg_idx))

 	def get_labels(self, n_fold=8, top_n=50):
 		'''returns train_y, valid_y, test_y '''
 		label_matrix = np.load(PATH_DATA + FILE_DICT['sorted_merged_label_matrix'])
		train_idx, valid_idx, test_idx = fm.shuffle(n_fold=8) # train, valid, test indices






