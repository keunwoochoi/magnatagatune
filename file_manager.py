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
		self.clip_ids = [] # integer.
		self.paths = [] #string
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
			for line_idx, line in enumerate(f):
				values = [value.rstrip('\r\n').strip('"') for value in line.split('\t')]
				# [0]:clip_id, track_no (in album), title, artist, album, url, seg_start, seg_end, original_url, [9]:mp3_path
				self.clip_ids.append(int(values[0]))
				if values[9] == '':
					pdb.set_trace()
				self.paths.append(values[9])
				self.id_to_paths[values[0]] = values[9]
				self.id_to_idx[values[0]] = line_idx
				if line_idx % 5000 == 0:
					print 'Line idx : %d loaded.' % line_idx
			print 'All info from clip_info_final.csv loaded.'
		self.num_songs = len(self.clip_ids)
		
		print 'Now will read annotations_final.csv'
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
			raise RuntimeError('self.num_songs:%d, self.num_tags:%d, stop here' % (self.num_songs, self.num_tags))
			return
		with open(PATH_MAGNA + 'annotations_final.csv', 'r') as f:
			tag_names = f.readline() # clip_id, 188 tags, mp3_path
			tags = [value.rstrip('\r\n').strip('"') for value in tag_names.split('\t')]
			self.num_tags  = len(tags) - 2 
			label_matrix = np.zeros((self.num_songs, self.num_tags), dtype=np.bool_)
			for line_idx, line in enumerate(f):
				values = [value.rstrip('\r\n').strip('"') for value in line.split('\t')]
				labels = [int(ele) for ele in values[1:-1]]
				label_matrix[self.id_to_idx[values[0]], : ] = np.array(labels, dtype=np.int)
				if line_idx % 2000 == 0:
					print 'Line idx : %d loaded.' % line_idx

		print 'label matrix created'
		np.save(PATH_DATA + FILE_DICT['label_matrix'], label_matrix)
		return label_matrix

 	def load_file(self, file_type, clip_id, seg_idx):
 		'''for file tyle (cqt, stft, mel,..) 
 		file_type : string, 'cqt', 'stft',..
 		clip_id   : integer. 
 		seg_idx   : integer in range(7): 
 		
 		return: corresponding numpy array, 2d, for 4-seconds.
 		'''
 		pass

 	# def load_mp3(self, clip_id):
 	# 	return librosa.load(PATH_MAGNA + self.id_to_paths[clip_id], sr=SR)






