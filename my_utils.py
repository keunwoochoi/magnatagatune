import numpy as np
import sys
import os
from environments import *
from constants import *
import cPickle as cP


def refine_label_matrix():
	''' It load label matrix and refine it by
	- merge tags that means the same.
	- pick top 50 tags
	- remove strange tags
	'''

	weird_tags = ['girl', 'lol', 'different', 'english', 'not english']
	synonyms = [['beat', 'beats'],
				['chant', 'chanting'],
				['choir', 'choral'],
				['classical', 'clasical', 'classic'],
				['drum', 'drums'],
				['electro', 'electronic', 'electronica', 'electric'],
				['fast', 'fast beat', 'quick'],
				['female', 'female singer', 'female singing', 'female vocals', 'female voice', 'woman', 'woman singing', 'women'],
				['flute', 'flutes'],
				['guitar', 'guitars'],
				['hard', 'hard rock'],
				['harpsichord', 'harpsicord'],
				['heavy', 'heavy metal', 'metal'],
				['horn', 'horns'],
				['india', 'indian'],
				['jazz', 'jazzy'],
				['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
				['no beat', 'no drums'],
				['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
				['opera', 'operatic'],
				['orchestra', 'orchestral'],
				['quiet', 'silence'],
				['singer', 'singing'],
				['space', 'spacey'],
				['string', 'strings'],
				['synth', 'synthesizer'],
				['violin', 'violins'],
				['vocal', 'vocals', 'voice', 'voices'],
				['strange', 'weird']]

	fm = cP.load(open(PATH_DATA + FILE_DICT["file_manager"], 'r'))

	whole_label_matrix = fm.load_label_matrix()
	tags_to_data = {}
	for tag_idx, tag in enumerate(fm.tags): # for 188 tags,
		tags_to_data[tag] = whole_label_matrix[:, tag_idx]

	new_tags_to_data = {}
	# merge
	for syn_list in synonyms:
		new_name = syn_list[0] + '_merged'
		new_tags_to_data[new_name] = 0
		for syn in syn_list:
			new_tags_to_data[new_name] += tags_to_data[syn]
	# add the others
	synonyms_flat = [tag for sublist in synonyms for tag in sublist]
	for tag in tags_to_data:
		if tag not in synonyms_flat:
			new_tags_to_data[tag] = tags_to_data[tag]
	# sort by total counts
	new_tags = new_tags_to_data.keys()
	num_new_tags = len(new_tags)
	merged_label_matrix = np.zeros((fm.num_songs , num_new_tags))
	
	for tag_idx, tag in enumerate(new_tags):
		merged_label_matrix[:, tag_idx] = new_tags_to_data[tag]

	total_counts = np.sum(merged_label_matrix, axis=0)
	tag_args = total_counts.argsort()[::-1] # descending order 
	sorted_merged_label_matrix = np.zeros((fm.num_songs , num_new_tags))
	sorted_tags = []
	for new_idx, tag_idx in enumerate(tag_args):
		sorted_merged_label_matrix[:,new_idx] = merged_label_matrix[:,tag_idx]
		sorted_tags.append(new_tags[tag_idx])

	# trim at 1.0 
	sorted_merged_label_matrix =  np.minimum(sorted_merged_label_matrix, np.ones(sorted_merged_label_matrix.shape))
	np.save(PATH_DATA + FILE_DICT['sorted_merged_label_matrix'], sorted_merged_label_matrix)
	cP.dump(sorted_tags, open(PATH_DATA + FILE_DICT['sorted_tags'], 'w'))
	return



