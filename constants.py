# -*- coding: utf-8 -*-
SR = 16000
N_FFT = 512 # 257 freq bin for stft
WIN_LEN = 512
HOP_LEN = 256 # 11 sec --> 512 frames
FRAMES_PER_SEC = float(SR) / HOP_LEN

NUM_SEG = 7 # segment per clip
LEN_SEG = 4.0

BINS_PER_OCTAVE = 36
NUM_OCTAVE = 7 # 252 freq bins for cqt
N_CQT_BINS = NUM_OCTAVE*BINS_PER_OCTAVE

FILE_DICT = {}
FILE_DICT['label_matrix'] = 'label_matrix.npy'
FILE_DICT['file_manager'] = 'file_manager.cP'

FILE_DICT['sorted_label_matrix'] = 'sorted_label_matrix.npy'
FILE_DICT['sorted_tags'] = 'sorted_tags.cP'

FILE_DICT['LDA_50_label_matrix'] = 'LDA_50_label_matrix.npy'

FILE_DICT['sorted_merged_label_matrix'] = 'sorted_merged_label_matrix.npy'
FILE_DICT['sorted_merged_tags'] = 'sorted_merged_tags.cP'
FILE_DICT['conventional_set_idxs'] = 'conventional_set_idxs.npy'
FILE_DICT["hyperparam_manager"] = "hyperparam_manager.cP"

