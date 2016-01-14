from environments import *
import os
import cPickle as cP
import time
import sys
import numpy as np
import h5py
import pdb

class File_Manager():
	def __init__(self):
		self.clip_ids = []
		self.paths = []
		self.idx_permutation = []
		self.id_permutation = []
		self.id_to_paths = {}
		self.feat_to_paths = {} # string : list, e.g. 'cqt' : blah..
		self.id_to_idx = {}
		self.np_whole_label_matrix = None # will be numpy array
		self.num_songs = 0
		self.num_tags = 0
	
	def fill_from_csv(self):
		with open(PATH_MAGNA+'clip_info_final.csv', 'r') as f:
			labels = f.readline() # first line is labels.
			for line_idx, line in enumerate(f):
				values = [value.rstrip('\r\n').strip('"') for value in line.split('\t')]
				# [0]:clip_id, track_no (in album), title, artist, album, url, seg_start, seg_end, original_url, [9]:mp3_path
				self.clip_ids.append(values[0])
				self.paths.append(values[9])
				self.id_to_paths[values[0]] = values[9]
				self.id_to_idx[values[0]] = line_idx
				if line_idx % 50 == 0:
					print 'Line idx : %d loaded.' % line_idx
			print 'All info from clip_info_final.csv loaded.'
		self.num_songs = len(self.clip_ids)
		
		print 'Now will read annotations_final.csv'
		with open(PATH_MAGNA + 'annotations_final.csv', 'r') as f:
			tag_names = f.readline() # clip_id, 188 tags, mp3_path
			tags = [value.rstrip('\r\n').strip('"') for value in tags.split('\t')]
			self.num_tags  = len(tags) - 2 
			self.np_whole_label_matrix = np.zeros((num_songs, num_tags), dtype=np.bool_)
			for line_idx, line in enumerate(f):
				values = [value.rstrip('\r\n').strip('"') for value in line.split('\t')]
				labels = [int(ele) for ele in values[1:-1]]
				self.np_whole_label_matrix[self.id_to_idx[values[0]], : ] = np.array(labels, dtype=np.int)
				if line_idx % 50 == 0:
					print 'Line idx : %d loaded.' % line_idx
			print 'All info from annotations_final.csv loaded.'
				
 	def squeeze_label_matrix(self, reduced_num):
 		'''pick top-N popular tags and return the song-tag matrix.'''
 		pass

 	def load_file(self, file_type, clip_id, seg_id):
 		'''for file tyle (cqt, stft, mel,..) 
 		file_type : string, 'cqt', 'stft',..
 		clip_id   : integer. 
 		seg_id    : integer in range(7): 
 		
 		return: corresponding numpy array, 2d, for 4-seconds.
 		'''
 		pass







