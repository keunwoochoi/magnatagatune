SR = 22050
N_FFT = 1024
WIN_LEN = 1024
HOP_LEN = 512 # 11 sec --> 512 frames
FRAMES_PER_SEC = float(SR) / HOP_LEN

FILE_DICT = {}
FILE_DICT['label_matrix'] = 'label_matrix.npy'
FILE_DICT['file_manager'] = 'file_manager.cP'
FILE_DICT['sorted_merged_label_matrix'] = 'sorted_merged_label_matrix.npy'

