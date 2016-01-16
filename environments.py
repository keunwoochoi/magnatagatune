import platform
import os
import sys

device_name = platform.node()

if device_name.startswith('ewert-server'):
	isMacPro = True
	isServer = False
	isMacbook= False
	isDT1     = False
	isDT2 	 = False
elif device_name in ["KChoiMBPR2013.local", "KChoi.MBPR.2013.home", "lt91-51", 'lt91-51.eecs.qmul.ac.uk','lt91-47']:
	print 'macbook pro'
	isMacPro = False
	isServer = False
	isMacbook = True
	isDT1 = False
	isDT2 = False

elif device_name in ['octave', 'big-bird','lincoln','frank','jazz']:
	isMacPro = False
	isServer = True
	isMacbook= False
	isDT1 	 = False
	isDT2 	 = False

elif device_name.startswith('keunwoo'):
	isMacPro = False
	isServer = False
	if device_name == "keunwoo-dt-ubuntu":
		isMacbook = False
		isDT1 = True
		isDT2 = False
	elif device_name == "keunwoo-dt2":
		isMacbook= False
		isDT1 = False
		isDT2 = True
else:
	isMacPro = False
	isServer = False
	isMacbook = True
	isDT1 = False
	isDT2 = False

if isMacPro:
	print "This is MacPro in CS.319"
	PATH_IMPORT = '/Users/keunwoo/mnt/c4dm/'
	PATH_HOME   = '/Users/keunwoo/mnt/kc306home/'

elif isServer:
	print "THIS IS A SERVER NAMED %s" % device_name

	PATH_IMPORT = '/import/'	
	PATH_HOME = "/homes/kc306/"

elif isDT1:
	print "You are using Ubuntu Desktop"
	PATH_IMPORT = '/mnt/c4dm/'
	PATH_HOME   = '/mnt/kc306home/'
elif isMacbook:
	print "Do not use MacbookPro for computation!...I hope."
	PATH_IMPORT = '/Users/gnu/mnt/c4dm/'
	PATH_HOME   = '/Users/gnu/Gnubox/'

if isMacPro:
	PATH_HDF_LOCAL = '/Users/keunwoo/data/hdf_magna/'
elif isDT1:
	PATH_HDF_LOCAL = '/home/keunwoo/data/hdf_magna/'
elif isServer:
	PATH_HDF_LOCAL = '/import/c4dm-04/keunwoo/magnatagatune/hdf/'
elif isMacbook:
	PATH_HDF_LOCAL = '/Users/gnu/Gnubox/Srcs/magnatagatune/'

if isMacbook:
	PATH_EMBEDDING = PATH_HOME + "embedding_tag/"
else:
	PATH_EMBEDDING = PATH_HOME + "embedding/"
	
PATH_WORK = PATH_HOME + "magnatagatune/"
PATH_DATA = PATH_WORK + 'data/'
PATH_RESULTS= PATH_WORK + 'results/'
PATH_RESULTS_W= PATH_WORK + 'results_w/'

PATH_MAGNA = PATH_IMPORT + 'c4dm-datasets/MagnaTagATune/'

PATH_MELGRAM = PATH_IMPORT + 'c4dm-04/keunwoo/magnatagatune/melgram/'
PATH_CQT = PATH_IMPORT + 'c4dm-04/keunwoo/magnatagatune/cqt/'
PATH_STFT = PATH_IMPORT + 'c4dm-04/keunwoo/magnatagatune/stft/'
PATH_MFCC = PATH_IMPORT + 'c4dm-04/keunwoo/magnatagatune/mfcc/'

PATH_TF = {}
PATH_TF['cqt'] = PATH_CQT
PATH_TF['stft']= PATH_STFT
PATH_TF['mfcc']= PATH_MFCC
PATH_TF['melgram']=PATH_MELGRAM

for path in [PATH_DATA, PATH_MELGRAM, PATH_CQT, PATH_STFT, PATH_MFCC, PATH_HDF_LOCAL, PATH_RESULTS, PATH_RESULTS_W]:
	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			print "Can't make the directory: %s" % path
			pass

if isMacbook:
	pass
else:

	for path in []:
		if not os.path.exists(path):
			os.mkdir(path)

	# sys.path.append(PATH_HOME + 'modules/' + 'librosa/')
