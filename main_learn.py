import numpy as np
import sys
import os
import time
import h5py
import keras
from keras.utils.visualize_util import plot as keras_plot

import my_input_output as io
from environments import *
from constants import *

sys.path.append(PATH_EMBEDDING)
from training_settings import *
import my_utils
import my_keras_models
import my_keras_utils
import my_plots
import hyperparams_manager


def run_with_setting(hyperparams, argv=None):
	# pick top-N from label matrix
	dim_labels = hyperparams['dim_labels']	
	# label_matrix = np.load(PATH_DATA + FILE_DICT['sorted_merged_label_matrix'])
	# label_matrix = label_matrix[:, :dim_labels]
	train_x, valid_x, test_x = io.load_x(hyperparams['tf_type'])
	train_y, valid_y, test_y = io.load_y(dim_labels)
	if hyperparams['is_test']:
		train_x = train_x[:96]
		valid_x = valid_x[:96]
		test_x = test_x[:96]
		train_y = train_y[:96]
		valid_y = valid_y[:96]
		test_y = test_y[:96]
		
	hyperparams['height_image'] = train_x.shape[2]
	hyperparams["width_image"]  = train_x.shape[3]
	
	hp_manager = hyperparams_manager.Hyperparams_Manager()


	# name, path, ...
	nickname = hp_manager.get_name(hyperparams)
	timename = time.strftime('%m-%d-%Hh%M')
	if hyperparams["is_test"]:
		model_name = 'test_' + nickname
	else:
		model_name = timename + '_' + nickname
	hp_manager.save_new_setting(hyperparams)
	print '-'*60
	print 'model name: %s' % model_name
	model_name_dir = model_name + '/'
	model_weight_name_dir = 'w_' + model_name + '/'
	fileout = model_name + '_results'
	
	if not os.path.exists(PATH_RESULTS + model_name_dir):
		os.mkdir(PATH_RESULTS + model_name_dir)
		os.mkdir(PATH_RESULTS + model_name_dir + 'images/')
		os.mkdir(PATH_RESULTS + model_name_dir + 'plots/')
		os.mkdir(PATH_RESULTS_W + model_weight_name_dir)
	hp_manager.write_setting_as_texts(PATH_RESULTS + model_name_dir, hyperparams)
 	hp_manager.print_setting(hyperparams)
 	# build model
 	model = my_keras_models.build_convnet_model(setting_dict=hyperparams)
	# prepare callbacks
	keras_plot(model, to_file=PATH_RESULTS + model_name_dir + 'images/'+'graph_of_model_'+hyperparams["!memo"]+'.png')
	checkpointer = keras.callbacks.ModelCheckpoint(filepath=PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5", 
													verbose=1, 
								             		save_best_only=True)
	weight_image_monitor = my_keras_utils.Weight_Image_Saver(PATH_RESULTS + model_name_dir + 'images/')
	patience = 3
	if hyperparams["is_test"] is True:
		patience = 99999999
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
														patience=patience, 
														verbose=0)
	# other constants
	if hyperparams["tf_type"] == 'cqt':
		batch_size = 48
	elif hyperparams["tf_type"] == 'stft':
		batch_size = 24
	elif hyperparams["tf_type"] == 'mfcc':
		batch_size = 96
	elif hyperparams["tf_type"] == 'melgram':
		batch_size = 96
	else:
		raise RuntimeError('batch size for this? %s' % hyperparams["tf_type"])
	if hyperparams['model_type'] == 'vgg_original':
		batch_size = (batch_size * 3)/5
	# ready to run
	predicted = model.predict(test_x, batch_size=batch_size)
	if hyperparams['debug'] == True:
		pdb.set_trace()
	print 'mean of target value:'
	print np.mean(test_y, axis=0)
	print 'mean of predicted value:'
	print np.mean(predicted, axis=0)
	print 'mse with just predicting average is %f' % np.mean((test_y - np.mean(test_y, axis=0))**2)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_init.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])
	print '--- train starts. Remove will_stop.keunwoo to continue learning after %d epochs ---' % hyperparams["num_epoch"]
	f = open('will_stop.keunwoo', 'w')
	f.close()
	total_history = {}
	num_epoch = hyperparams["num_epoch"]
	total_epoch = 0
	if hyperparams['is_test']:
		callbacks = [weight_image_monitor]
	else:
		callbacks = [weight_image_monitor, early_stopping, checkpointer]

 	# run
	while True:	
		batch_size = batch_size / 2
		history=model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
										batch_size=batch_size, 
										nb_epoch=num_epoch, 
										show_accuracy=True, 
										verbose=1, 
										callbacks=callbacks,
										shuffle='batch')
		total_epoch += num_epoch
		print '%d-th epoch is complete' % total_epoch
		my_utils.append_history(total_history, history.history)
		
		if os.path.exists('will_stop.keunwoo'):	
			loss_testset = model.evaluate(test_x, test_y, show_accuracy=True, batch_size=batch_size)
			break
		else:
			num_epoch = 1
			print ' *** will go for another one epoch. '
			print ' *** $ touch will_stop.keunwoo to stop at the end of this, otherwise it will be endless.'
	#
	best_batch = np.argmin(total_history['val_acc'])+1
	predicted = model.predict(test_x, batch_size=batch_size)
	print predicted[:10]

	if hyperparams["debug"] == True:
		pdb.set_trace()
	if not hyperparams['is_test']:
		model.load_weights(PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5") 
	predicted = model.predict(test_x, batch_size=batch_size)
	print predicted[:10]
	
	#save results
	np.save(PATH_RESULTS + model_name_dir + fileout + '_history.npy', [total_history['acc'], total_history['val_acc']])
	np.save(PATH_RESULTS + model_name_dir + fileout + '_loss_testset.npy', loss_testset)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_result.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])
	np.save(PATH_RESULTS + model_name_dir + 'weights_changes.npy', np.array(weight_image_monitor.weights_changes))

	# ADD weight change saving code	
	my_plots.export_history(total_history['loss'], total_history['val_loss'], 
												acc=total_history['acc'], 
												val_acc=total_history['val_acc'], 
												out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
	
	min_loss = np.min(total_history['val_acc'])
	best_batch = np.argmin(total_history['val_acc'])+1
	num_run_epoch = len(total_history['val_acc'])
	oneline_result = '%6.4f, %d_of_%d, %s' % (min_loss, best_batch, num_run_epoch, model_name)
	with open(PATH_RESULTS + model_name_dir + oneline_result, 'w') as f:
		pass
	f = open( (PATH_RESULTS + '%s_%s_%06.4f_at_(%d_of_%d)_%s'  % \
		(timename, hyperparams["loss_function"], min_loss, best_batch, num_run_epoch, nickname)), 'w')
	f.close()
	with open('one_line_log.txt', 'a') as f:
		f.write('%6.4f, %d/%d, %s' % (min_loss, best_batch, num_run_epoch, model_name))
		f.write(' ' + ' '.join(argv) + '\n')
	print '========== DONE: %s ==========' % model_name
	return min_loss

	
if __name__ == '__main__':

	TR_CONST['isClass'] = True
	TR_CONST['isRegre'] = False
	TR_CONST["clips_per_song"] = 7
	# TR_CONST['loss_function'] = 'categorical_crossentropy'
	# TR_CONST["output_activation"] = 'softmax'
	TR_CONST['loss_function'] = 'binary_crossentropy'
	TR_CONST["output_activation"] = 'sigmoid'
	TR_CONST["dropouts"] = [0.5]*TR_CONST["num_layers"]
	TR_CONST["BN"] = False
	TR_CONST["regulariser"] = [('l2', 0.)]*TR_CONST["num_layers"] # use [None] not to use.


	TR_CONST["BN_fc_layers"] = True 
	TR_CONST["dropouts_fc_layers"] = [0.5]*TR_CONST["num_fc_layers"]

	TR_CONST["nums_units_fc_layers"] = [128]*TR_CONST["num_fc_layers"]
	TR_CONST["activations_fc_layers"] = ['lrelu']*TR_CONST["num_fc_layers"]
	TR_CONST["regulariser_fc_layers"] = [('l2', 0.), ('l2', 0.)]
	TR_CONST["BN_fc_layers"] = True 



	
	run_with_setting(TR_CONST, sys.argv)

