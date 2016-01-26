import numpy as np
import sys
import os
import time
import h5py
import argparse
import pprint
from sklearn import metrics
import pdb
import keras
from keras.utils.visualize_util import plot as keras_plot
import cPickle as cP
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

def evaluate_result(y_true, y_pred):
	ret = {}
	ret['roc_auc_micro'] = metrics.roc_auc_score(y_true, y_pred, average='micro')
	ret['roc_auc_macro'] = metrics.roc_auc_score(y_true, y_pred, average='macro')
	
	print '.'*60
	for key in ret:
		print key, ret[key]
	print '.'*60
	return ret


def update_setting_dict(setting_dict):

	setting_dict["num_feat_maps"] = [setting_dict["num_feat_maps"][0]]*setting_dict["num_layers"]
	setting_dict["activations"] = [setting_dict["activations"][0]] *setting_dict["num_layers"]
	setting_dict["dropouts"] = [setting_dict["dropouts"][0]]*setting_dict["num_layers"]
	setting_dict["regulariser"] = [setting_dict["regulariser"][0]]*setting_dict["num_layers"]

	setting_dict["dropouts_fc_layers"] = [setting_dict["dropouts_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["nums_units_fc_layers"] = [setting_dict["nums_units_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["activations_fc_layers"] = [setting_dict["activations_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["regulariser_fc_layers"] = [setting_dict["regulariser_fc_layers"][0]]*setting_dict["num_fc_layers"]

	return

def append_history(total_history, local_history):
	'''local history is a dictionary,
	key:value == string:dictionary.

	key: loss, vall_loss, batch, size
	Therefore total_history has the same keys and append the values.
	'''

	for key in local_history:
		if key not in total_history:
			total_history[key] = []
		total_history[key] = total_history[key] + local_history[key]

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

def run_with_setting(hyperparams, argv=None, batch_size=None):
	f = open('will_stop.keunwoo', 'w')
	f.close()
	if os.path.exists('stop_asap.keunwoo'):
		os.remove('stop_asap.keunwoo')
	# pick top-N from label matrix
	dim_labels = hyperparams['dim_labels']	
	shuffle = 'batch'
	num_sub_epoch = 5
	# label_matrix = np.load(PATH_DATA + FILE_DICT['sorted_merged_label_matrix'])
	# label_matrix = label_matrix[:, :dim_labels]
	train_x, valid_x, test_x = io.load_x(hyperparams['tf_type'])
	train_y, valid_y, test_y = io.load_y(dim_labels)
	if hyperparams['is_test']:
		pdb.set_trace()
		num_data_in_test = 256
		train_x = train_x[:num_data_in_test]
		valid_x = valid_x[:num_data_in_test]
		test_x  = test_x[:num_data_in_test]
		train_y = train_y[:num_data_in_test]
		valid_y = valid_y[:num_data_in_test]
		test_y  = test_y[:num_data_in_test]
		shuffle = False
		num_sub_epoch = 1
		
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
	# checkpointer = keras.callbacks.ModelCheckpoint(filepath=PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5", 
	# 												 monitor='val_acc',
	# 												verbose=1, 
	# 							             		save_best_only=True)
	weight_image_monitor = my_keras_utils.Weight_Image_Saver(PATH_RESULTS + model_name_dir + 'images/')
	patience = 100
	if hyperparams["is_test"] is True:
		patience = 99999999
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
														patience=patience, 
														verbose=0)
	if batch_size == None:
		batch_size = 16
	
	if hyperparams['model_type'] == 'vgg_original':
		batch_size = (batch_size * 3)/5
	# ready to run
	if hyperparams['debug'] == True:
		pdb.set_trace()
	print '--- train starts. Remove will_stop.keunwoo to continue learning after %d epochs ---' % hyperparams["num_epoch"]
	
	num_epoch = hyperparams["num_epoch"]
	total_epoch = 0
	
	callbacks = [weight_image_monitor]

	total_history = {}
	total_label_count = train_y.shape[0]*train_y.shape[1]
	print 'With predicting all zero, acc is %0.6f' % ((total_label_count - np.sum(train_y))/float(total_label_count))

	if hyperparams['resume'] != '':
		if os.path.exists(PATH_RESULTS_W + 'w_' + hyperparams['resume']):
			model.load_weights(PATH_RESULTS_W + 'w_' + hyperparams['resume'] + '/weights_best.hdf5')
		if os.path.exists(PATH_RESULTS + hyperparams['resume'] + '/total_history.cP'):
			previous_history = cP.load(open(PATH_RESULTS + hyperparams['resume'] + '/total_history.cP', 'r'))
			print 'previously learned weight: %s is loaded ' % hyperparams['resume']
			append_history(total_history, previous_history)
	if not hyperparams['do_not_learn']:
		my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
											filename_prefix='local_INIT', 
											normalize='local', 
											mono=True)
		my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
											filename_prefix='global_INIT', 
											normalize='global', 
											mono=True)
	 	# run
	 	print 'TEST FLIGHT'
	 	model.fit(train_x[-256:], train_y[-256:], validation_data=(valid_x[:512], valid_y[:512]), 
															batch_size=batch_size, 
															nb_epoch=1, 
															show_accuracy=hyperparams['isClass'], 
															verbose=1, 
															callbacks=callbacks,
															shuffle=shuffle)
	 	print 'TEST FLIGHT DONE'
		while True:	
			for sub_epoch_idx in range(num_sub_epoch):
				if os.path.exists('stop_asap.keunwoo'):
					break
				seg_from = sub_epoch_idx * (train_x.shape[0]/num_sub_epoch)
				seg_to   = (sub_epoch_idx+1) * (train_x.shape[0]/num_sub_epoch)
				train_x_here = train_x[seg_from:seg_to]
				train_y_here = train_y[seg_from:seg_to]
				if sub_epoch_idx == (num_sub_epoch-1):
					valid_data = (valid_x, valid_y)
				else:
					valid_data = (valid_x[:2048], valid_y[:2048])
				# early_stop should watch overall AUC rather than val_loss or val_acc
				batch_size_applied = batch_size
				model.fit(train_x_here, train_y_here, validation_data=None, 
															batch_size=batch_size_applied, 
															nb_epoch=1, 
															show_accuracy=hyperparams['isClass'], 
															verbose=1, 
															callbacks=callbacks,
															shuffle=shuffle)
				# check with AUC
				predicted = model.predict(valid_x, batch_size=batch_size)
				val_result = evaluate_result(valid_y, predicted)
				history = {}
				history['auc'] = [val_result['roc_auc_macro']]
				# print 'history[auc]'
				# print history['auc']
				# print 'total_history'
				# print total_history
				append_history(total_history, history)

			print '%d-th of %d epoch is complete, auc:%f' % (total_epoch, num_epoch, val_result['roc_auc_macro'])
			total_epoch += 1

			if os.path.exists('stop_asap.keunwoo'):
				os.remove('stop_asap.keunwoo')
				# loss_testset = model.evaluate(test_x, test_y, show_accuracy=True, batch_size=batch_size)

				break
			
			if os.path.exists('will_stop.keunwoo'):	
				if total_epoch > num_epoch:
					# loss_testset = model.evaluate(test_x, test_y, show_accuracy=True, batch_size=batch_size)
					break
				else:
					print ' *** will go for %d epochs' % (num_epoch - total_epoch)
			else:
				print ' *** will go for another one epoch. '
				print ' *** $ touch will_stop.keunwoo to stop at the end of this, otherwise it will be endless.'
	
	if hyperparams["debug"] == True:
		pdb.set_trace()
	if not hyperparams['is_test']:
		model.load_weights(PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5") 
	predicted = model.predict(test_x, batch_size=batch_size)
	eval_result_final = evaluate_result(test_y, predicted)
	print '.'*60
	for key in sorted(eval_result_final.keys()):
		print key, eval_result_final[key]
	print '.'*60
	
	#save results
	cP.dump(total_history, open(PATH_RESULTS + model_name_dir + 'total_history.cP', 'w'))
	# np.save(PATH_RESULTS + model_name_dir + 'loss_testset.npy', loss_testset)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_result.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])
	np.save(PATH_RESULTS + model_name_dir + 'weights_changes.npy', np.array(weight_image_monitor.weights_changes))

	# ADD weight change saving code
	if total_history != {}:
		
		my_plots.export_list_png(total_history['auc'], out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png', title='AUC' )
		# my_plots.export_history(total_history['loss'], total_history['val_loss'], 
		# 											acc=total_history['acc'], 
		# 											val_acc=total_history['val_acc'], 
		# 											out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
		
		max_auc = np.max(total_history['auc'])
		best_batch = np.argmax(total_history['auc'])+1
		num_run_epoch = len(total_history['auc'])
		oneline_result = '%6.4f, auc %d_of_%d, %s' % (max_auc, best_batch, num_run_epoch, model_name)
		with open(PATH_RESULTS + model_name_dir + oneline_result, 'w') as f:
			pass
		f = open( (PATH_RESULTS + '%s_%s_auc_%06.4f_at_(%d_of_%d)_%s'  % \
			(timename, hyperparams["loss_function"], max_auc, best_batch, num_run_epoch, nickname)), 'w')
		f.close()
		with open('one_line_log.txt', 'a') as f:
			f.write(oneline_result)
			f.write(' ' + ' '.join(argv) + '\n')
	else:
		max_auc = 0.0
	print '========== DONE: %s ==========' % model_name
	return max_auc

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parser for input arguments')
	parser.add_argument('-ne', '--n_epoch', type=int, 
											help='set the number of epoch, \ndefault=30', 
											required=False)
	parser.add_argument('-tf', '--tf', help='whether cqt, stft, mfcc, melgram \ndefault=cqt.', 
								required=False)
	parser.add_argument('-m', '--model', help='set the model, \ndefault=vgg_simple.', 
								   		required=False)
	parser.add_argument('-l', '--layers', type=int,
								 		help='set the number(s) of layers, \ndefault=[5], set like 4, 5, 6',
										required=False)
	parser.add_argument('-lfc', '--num_fc_layers', type=int,
								 		help='set the number(s) of fc layers, \ndefault=[2], set like 1, 2, 3',
										required=False)
	parser.add_argument('-t', '--task', help='classification or regression, \ndefault=regre', 
									   required=False)
	parser.add_argument('-op', '--optimiser', help='optimiser - rmsprop, sgd, adagrad, adam, adadelta \ndefault=rmsprop', 
									   required=False)
	parser.add_argument('-lf', '--loss_function', help='loss function - binary_crossentropy, rmse\ndefault=binary_crossentropy', 
									   required=False)
	parser.add_argument('-act', '--activations', help='activations - relu, lrelu, prelu, elu \ndefault=relu', 
									   required=False)
	parser.add_argument('-cps', '--clips_per_song', type=int,
													help='set #clips/song, \ndefault=3',
													required=False)
	parser.add_argument('-dl', '--dim_labels', type=int,
												help='set dimension of label, \ndefault=3',
												required=False)
	parser.add_argument('-fm', '--feature_maps', type=int,
												help='set number of feature maps in convnet, \ndefault=48',
												required=False)
	parser.add_argument('-nu', '--number_units', type=int,
												help='set number of units in fc layers, \ndefault=512',
												required=False)	
	parser.add_argument('-it', '--is_test', type=int,
												help='say if it is test \ndefault=0 (False)',
												required=False)
	parser.add_argument('-memo', '--memo', 	help='short memo \ndefault=""',
											required=False)
	parser.add_argument('-do', '--dropout', type=float,
											help='dropout value that is applied to conv',
											required=False)
	parser.add_argument('-do_fc', '--dropout_fc', type=float,
												help='dropout value that is applied to FC layers',
												required=False)
	parser.add_argument('-reg', '--regulariser', type=float,
												help='regularise coeff that is applied to conv',
												required=False)
	parser.add_argument('-reg_fc', '--regulariser_fc', type=float,
														help='regularise coeff that is applied to fc layer',
														required=False)
	parser.add_argument('-bn', '--batch_normalization', type=str,
														help='BN for conv layers',
														required=False)
	parser.add_argument('-bn_fc', '--batch_normalization_fc', type=str,
															help='BN for fc layers',
															required=False)
	parser.add_argument('-mo', '--maxout', type=str,
											help='Maxout true or false',
											required=False)
	parser.add_argument('-debug', '--debug', type=str,
											help='if debug',
											required=False)
	parser.add_argument('-lr', '--learning_rate', type=float,
													help='learning_rate',
													required=False)
	parser.add_argument('-ol', '--output_layer', type=str,
												help='sigmoid, linear',
												required=False )
	parser.add_argument('-rs', '--resume', type=str,
										help='model name with date, without w_, to load.',
										required=False )
	parser.add_argument('-dnl', '--do_not_learn', type=str,
										help='model name with date, without w_, to load.',
										required=False )
	parser.add_argument('-bs', '--batch_size', type=int,
										help='batch size',
										required=False )
	parser.add_argument('-gn', '--gaussian_noise', type=str,
										help='add noise? true or false',
										required=False )
	parser.add_argument('-gn_sigma', '--gn_sigma', type=float,
										help='sigma of gaussian noise',
										required=False )
	
	
	args = parser.parse_args()
	#------------------- default setting --------------------------------#
	TR_CONST["dim_labels"] = 50
	TR_CONST["num_layers"] = 4

	TR_CONST['isClass'] = True
	TR_CONST['isRegre'] = False
	TR_CONST["clips_per_song"] = 7
	TR_CONST['loss_function'] = 'binary_crossentropy'
	TR_CONST["optimiser"] = 'adam'
	TR_CONST['learning_rate'] = 1e-2
	TR_CONST["output_activation"] = 'sigmoid'

	TR_CONST["num_epoch"] = 3
	TR_CONST["dropouts"] = [0.0]*TR_CONST["num_layers"]
	TR_CONST["num_feat_maps"] = [32]*TR_CONST["num_layers"]
	TR_CONST["activations"] = ['elu']*TR_CONST["num_layers"]
	TR_CONST["BN"] = True
	TR_CONST["regulariser"] = [('l2', 0.0)]*TR_CONST["num_layers"] # use [None] not to use.
	TR_CONST["model_type"] = 'vgg_modi_3x3'
	TR_CONST["tf_type"] = 'melgram'

	TR_CONST["num_fc_layers"] = 3

	TR_CONST["BN_fc_layers"] = True
	TR_CONST["dropouts_fc_layers"] = [0.5]*TR_CONST["num_fc_layers"]

	TR_CONST["nums_units_fc_layers"] = [512]*TR_CONST["num_fc_layers"]
	TR_CONST["activations_fc_layers"] = ['elu']*TR_CONST["num_fc_layers"]
	TR_CONST["regulariser_fc_layers"] = [('l2', 0.0)] *TR_CONST["num_fc_layers"]
	TR_CONST["BN_fc_layers"] = True
	TR_CONST["maxout"] = True
	TR_CONST["gaussian_noise"] = False
	#--------------------------------------------------------#
	if args.layers:
		TR_CONST["num_layers"] = args.layers
	if args.num_fc_layers:
		TR_CONST["num_fc_layers"] = args.num_fc_layers
	if args.n_epoch:
		TR_CONST["num_epoch"] = args.n_epoch
	if args.tf:
		TR_CONST["tf_type"] = args.tf
		print 'tf-representation type is input by: %s' % TR_CONST["tf_type"]
	if args.optimiser:
		TR_CONST["optimiser"] = args.optimiser
	if args.loss_function:
		TR_CONST["loss_function"] = args.loss_function
	if args.model:
		TR_CONST["model_type"] = args.model
	if args.activations:
		TR_CONST["activations"] = [args.activations] * TR_CONST["num_layers"]
		TR_CONST["activations_fc_layers"] = [args.activations] * TR_CONST["num_fc_layers"]
	if args.task:
		if args.task in['class', 'cla', 'c', 'classification']:
			TR_CONST["isClass"] = True
			TR_CONST["isRegre"] = False
		else:
			TR_CONST["isClass"] = False
			TR_CONST["isRegre"] = True
	if args.clips_per_song:
		TR_CONST["clips_per_song"] = args.clips_per_song
	if args.dim_labels:
		TR_CONST["dim_labels"] = args.dim_labels
	if args.feature_maps:
		TR_CONST["num_feat_maps"] = [args.feature_maps]*TR_CONST["num_layers"]
	if args.number_units:
		TR_CONST["nums_units_fc_layers"] = [args.number_units]*TR_CONST["num_fc_layers"]
	if args.is_test:
		TR_CONST["is_test"] = bool(int(args.is_test))
	if args.memo:
		TR_CONST["!memo"] = args.memo
	else:
		TR_CONST["!memo"] = ''
	if args.dropout or args.dropout == 0.0:
		TR_CONST["dropouts"] = [args.dropout]*TR_CONST["num_layers"]
	if args.dropout_fc or args.dropout_fc == 0.0:
		TR_CONST["dropouts_fc_layers"] = [args.dropout_fc]*TR_CONST["num_fc_layers"]
	if args.regulariser or args.regulariser == 0.0:
		TR_CONST["regulariser"] = [(TR_CONST["regulariser"][0][0], args.regulariser)]*TR_CONST["num_layers"]
	if args.regulariser_fc or args.regulariser == 0.0:
		TR_CONST["regulariser_fc_layers"] = [(TR_CONST["regulariser_fc_layers"][0][0], args.regulariser_fc)]*TR_CONST["num_fc_layers"]
	if args.batch_normalization:
		TR_CONST["BN"] = str2bool(args.batch_normalization)
	if args.batch_normalization_fc:
		TR_CONST["BN_fc_layers"] = str2bool(args.batch_normalization_fc)
	if args.learning_rate:
		TR_CONST["learning_rate"] = args.learning_rate
	if args.maxout:
		TR_CONST["maxout"] = str2bool(args.maxout)
	if args.debug:
		TR_CONST["debug"] = str2bool(args.debug)
	if args.output_layer:
		TR_CONST["output_activation"] = args.output_layer
	if args.resume:
		TR_CONST["resume"] = args.resume
	else:
		TR_CONST["resume"] = ''
	if args.do_not_learn:
		TR_CONST["do_not_learn"] = args.do_not_learn
	if args.batch_size:
		batch_size = args.batch_size
	else:
		batch_size = 16
	if args.gaussian_noise:
		TR_CONST["gaussian_noise"] = str2bool(args.gaussian_noise)
	if args.gn_sigma:
		TR_CONST["gn_sigma"] = args.gn_sigma

 	#----------------------------------------------------------#
	# 1. vanilla setting: not learning. 
	# 2. regularise with 5e-4, 5e-4: 01-18-14h48_silly_pup, predicting means
	# 2-1 with 1e-1, 5e-7... similar result. 
	# 3. no reg, dropout (0.25, 0.25) only: learning stops after 1 subepoch.. (01_18-16h31_rage_gryph)
	# may be it's not about regularise. how about lrelu then.
	# 4. without 

	
	if False:
		# 01-18-18h04_sharp_dog
		# 27148/27148 [==============================] - 274s - loss: 0.1584 - acc: 0.9477 - val_loss: 0.1589 - val_acc: 0.9466
		# 27148/27148 [==============================] - 245s - loss: 0.1533 - acc: 0.9484
		# leark rely works! 
		TR_CONST["activations"] = ['lrelu'] # alpha was 0.1 this time
		TR_CONST["activations_fc_layers"] = ['lrelu']
		TR_CONST["!memo"] = 'vanilla_with_leaky_relu'
		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)
		# 01-18-18h30_pink_lynx
		# roc_auc_micro 0.5 0.57511327831
		# roc_auc_none 0.5 0.553722154374
		# f1_binary 0.0 0.204950506406
		# log_loss_after 10.6407992873 36.7952879927
		# precision 0.527200247219 0.406661829929
		# f1_macro 0.0 0.151255938067
		# log_loss_before 10.6407992873 7.84873996332
		# f1_micro 0.0 0.247900891155
		# roc_auc_macro 0.5 0.553722154374
		# 27148/27148 [==============================] - 222s - loss: 0.1435 - acc: 0.9500 - val_loss: 0.1484 - val_acc: 0.9480
		TR_CONST["activations"] = ['relu']
		TR_CONST["activations_fc_layers"] = ['relu']
		TR_CONST["BN"] = True
		TR_CONST["BN_fc_layers"] = True
		TR_CONST["!memo"] = 'vanilla_w_bn_all'
		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)
		# blue_bunny
		# 27148/27148 [==============================] - 318s - loss: 0.1499 - acc: 0.9485 - val_loss: 0.1837 - val_acc: 0.9454
		# overfits.
		TR_CONST["BN"] = False
		TR_CONST["BN_fc_layers"] = True
		TR_CONST["!memo"] = 'vanilla_bn_fc_only'
		run_with_setting(TR_CONST, sys.argv)
		# bear_wing
		# 27148/27148 [==============================] - 1220s - loss: 0.1591 - acc: 0.9475 - val_loss: 0.1613 - val_acc: 0.9464
		# this time there were 64 features per layer, which seems overkill
		TR_CONST["BN"] = True # requires very long time -- but why??
		TR_CONST["BN_fc_layers"] = False
		TR_CONST["!memo"] = 'vanilla_bn_conv_only'
		run_with_setting(TR_CONST, sys.argv)
	
		# also kinda working - red_orca
		# 27148/27148 [==============================] - 167s - loss: 0.0427 - acc: 0.9483 - val_loss: 0.0451 - val_acc: 0.9463
		# 27148/27148 [==============================] - 165s - loss: 0.0392 - acc: 0.9510 - val_loss: 0.0408 - val_acc: 0.9492
		TR_CONST["BN"] = False
		TR_CONST["!memo"] = 'mse_loss_function_w_sigmoid'
		TR_CONST["loss_function"] = 'mse'
		run_with_setting(TR_CONST, sys.argv)

		# BN with conve and mse, linear output - wet_doge
		# 27148/27148 [==============================] - 358s - loss: 0.0434 - acc: 0.9477 - val_loss: 0.0442 - val_acc: 0.9464
		TR_CONST["loss_function"] = 'mse'
		TR_CONST["output_activation"] = 'linear'
		TR_CONST["!memo"] = 'mse_loss_function_w_linear'
		run_with_setting(TR_CONST, sys.argv)	

		# BN for both, and lrelu - bad_orca
		TR_CONST["activations"] = ['lrelu']
		TR_CONST["activations_fc_layers"] = ['lrelu']
		TR_CONST["BN"] = True
		TR_CONST["BN_fc_layers"] = True
		# 01-19-00h37spotty_cat 
		# when combined BN and lrelu(0.1) together, it seems similar or even better with #layer=4
		# 27148/27148 [==============================] - 354s - loss: 0.1482 - acc: 0.9490 - val_loss: 0.1679 - val_acc: 0.9465
		# 27148/27148 [==============================] - 352s - loss: 0.1387 - acc: 0.9509 - val_loss: 0.1478 - val_acc: 0.9483
		# 27148/27148 [==============================] - 352s - loss: 0.1341 - acc: 0.9518 - val_loss: 0.1447 - val_acc: 0.9488
		# 27148/27148 [==============================] - 351s - loss: 0.1307 - acc: 0.9526 - val_loss: 0.1447 - val_acc: 0.9489
		# 27148/27148 [==============================] - 350s - loss: 0.1279 - acc: 0.9532 - val_loss: 0.1461 - val_acc: 0.9490
		# which is, roc_auc_none 0.5 0.554170834687
		TR_CONST["!memo"] = 'bn on and on, lrelu and lrelu, keep 32 per layer'
		TR_CONST["num_layers"] = 4
		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)

		# then try dropout on FC only first. 01-19-00h53_red_wolf ####### THIS IS PERHAPS ONE OF THE BEST!
		# 27148/27148 [==============================] - 352s - loss: 0.1541 - acc: 0.9480 - val_loss: 0.1529 - val_acc: 0.9474
		# 27148/27148 [==============================] - 350s - loss: 0.1433 - acc: 0.9497 - val_loss: 0.1472 - val_acc: 0.9482
		# roc_auc_none 0.5 0.553300884607
		# ...probably better generalisation? (and should be.)
		# no time consumption added! So if data allows it would be [0.5] rather than [0.25]
		# just quickly go with vgg original.
		TR_CONST["activations"] = ['lrelu'] # alpha is 0.3 now
		TR_CONST["activations_fc_layers"] = ['lrelu']
		TR_CONST["BN"] = True
		TR_CONST["BN_fc_layers"] = True
		TR_CONST["num_layers"] = 4
		TR_CONST["!memo"] = 'bn on and on, 4layer, dropout on fc only, lrelu and lrelu, keep 32 per layer'
		TR_CONST["dropouts_fc_layers"] = [0.25]
		TR_CONST["nums_units_fc_layers"] = [682] # with 0.25 this is equivalent to 512 units
		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)

		# vgg original. rage_deer
		# after 6 epochs, (all night)
		# 27148/27148 [==============================] - 737s - loss: 0.1317 - acc: 0.9521 - val_loss: 0.1388 - val_acc: 0.9499
		# roc_auc_none 0.5 0.585801977209
		TR_CONST["activations"] = ['lrelu'] # alpha is 0.3 now
		TR_CONST["activations_fc_layers"] = ['lrelu']
		TR_CONST["BN"] = True
		TR_CONST["BN_fc_layers"] = True
		TR_CONST["num_layers"] = 2
		TR_CONST["!memo"] = 'bn on and on, 4layer, dropout on fc only, lrelu and lrelu, keep 32 per layer'
		TR_CONST["dropouts_fc_layers"] = [0.5]
		TR_CONST["nums_units_fc_layers"] = [1024] # with 0.25 this is equivalent to 512 units
		TR_CONST["model_type"] = 'vgg_original'

		#----until here, only 3rd mini batch was using validation, but now 3 and 5 th will do.
		# vgg_simple, BN -true,true, num_layer in [3 and 6]
		# with three layers,
		# 27148/27148 [==============================] - 341s - loss: 0.1563 - acc: 0.9476 - val_loss: 0.1613 - val_acc: 0.9457
		# 27148/27148 [==============================] - 338s - loss: 0.1532 - acc: 0.9482 - val_loss: 0.1515 - val_acc: 0.9476
		# with large lr=3e-2, but no difference with adagrad.

		# with six layers,
		# 27148/27148 [==============================] - 360s - loss: 0.1619 - acc: 0.9472 - val_loss: 0.1613 - val_acc: 0.9464
		# 27148/27148 [==============================] - 360s - loss: 0.1584 - acc: 0.9476 - val_loss: 0.1587 - val_acc: 0.9467
		# So it is not always good. 


		# with four layers,
		# BN BN, dr 0.5, lrelu lrelu + 64 -> 4
		# 27148/27148 [==============================] - 606s - loss: 0.1774 - acc: 0.9466 - val_loss: 0.2442 - val_acc: 0.9220
		# It takes too long time per epoch. still performance is not good. 

		# same, batch size of 24
		# 27148/27148 [==============================] - 383s - loss: 0.1580 - acc: 0.9474 - val_loss: 0.1624 - val_acc: 0.9442
		# 27148/27148 [==============================] - 380s - loss: 0.1559 - acc: 0.9478 - val_loss: 0.1720 - val_acc: 0.9412
		
		# okay, back to B=64.
		# now lrelu + lrelu, 4layers, BN on/on, dropout 0.5/0.5 go.
		# 01-19-12h08_tiny_fox
		# 27148/27148 [==============================] - 364s - loss: 0.1743 - acc: 0.9466 - val_loss: 0.1853 - val_acc: 0.9454
		# 27148/27148 [==============================] - 357s - loss: 0.1725 - acc: 0.9467 - val_loss: 0.1845 - val_acc: 0.9449
		# 27148/27148 [==============================] - 359s - loss: 0.1680 - acc: 0.9467 - val_loss: 0.1945 - val_acc: 0.9426
		# no, at least 0.5 for convnet is not good.

		TR_CONST["activations"] = ['lrelu'] # alpha is 0.3 now
		TR_CONST["activations_fc_layers"] = ['lrelu']
		TR_CONST["BN"] = True
		TR_CONST["BN_fc_layers"] = True
		
		TR_CONST["!memo"] = 'batch size is 1, it is a stochastic gradient descent.'
		TR_CONST["dropouts_fc_layers"] = [0.5]
		TR_CONST["dropouts"] = [0.5]
		TR_CONST["nums_units_fc_layers"] = [1024] # with 0.25 this is equivalent to 512 units
		TR_CONST["num_layers"] = 4

		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)
		sys.exit()

	
		# now lrelu + prelu, 4layers, BN on/on, dropout ??/?? go.
		# 01-19-13h43_spotty_deer # be ware! it was logged as 'lrelu'
		# 27148/27148 [==============================] - 356s - loss: 0.1656 - acc: 0.9471 - val_loss: 0.1651 - val_acc: 0.9459
		# 27148/27148 [==============================] - 355s - loss: 0.1608 - acc: 0.9474 - val_loss: 0.1605 - val_acc: 0.9463
		# 27148/27148 [==============================] - 355s - loss: 0.1523 - acc: 0.9482 - val_loss: 0.1611 - val_acc: 0.9466
		# 27148/27148 [==============================] - 354s - loss: 0.1431 - acc: 0.9498 - val_loss: 0.1442 - val_acc: 0.9490
		# 27148/27148 [==============================] - 354s - loss: 0.1407 - acc: 0.9503 - val_loss: 0.1424 - val_acc: 0.9491
		# 27148/27148 [==============================] - 354s - loss: 0.1436 - acc: 0.9499 - val_loss: 0.1424 - val_acc: 0.9494
		# roc_auc_none 0.5 0.58110036599 --> best so far, similar to lrelu with vgg original for all night. 
		# so generalsation is not bad.

		# 01-19-16h28_red_wolf
		# maybe, only lrelu(0.03) was better at the beginning, but then...
		# 27148/27148 [==============================] - 356s - loss: 0.1647 - acc: 0.9471 - val_loss: 0.1664 - val_acc: 0.9461
		# 27148/27148 [==============================] - 355s - loss: 0.1602 - acc: 0.9475 - val_loss: 0.1635 - val_acc: 0.9451
		# 27148/27148 [==============================] - 355s - loss: 0.1519 - acc: 0.9483 - val_loss: 0.1551 - val_acc: 0.9470
		# 27148/27148 [==============================] - 354s - loss: 0.1528 - acc: 0.9483 - val_loss: 0.1726 - val_acc: 0.9437
		# 27148/27148 [==============================] - 355s - loss: 0.1466 - acc: 0.9493 - val_loss: 0.1558 - val_acc: 0.9467
		# 27148/27148 [==============================] - 354s - loss: 0.1485 - acc: 0.9492 - val_loss: 0.1491 - val_acc: 0.9481
		# 27148/27148 [==============================] - 354s - loss: 0.1435 - acc: 0.9498 - val_loss: 0.1467 - val_acc: 0.9481
		# 27148/27148 [==============================] - 354s - loss: 0.1460 - acc: 0.9495 - val_loss: 0.1516 - val_acc: 0.9474
		# roc_auc_none 0.5 0.565670673522
		# so lrelu(0.03) is not as good as lrelu(0.3)+prelu or lrelu(0.1)+same_lrelu

		# result: need to look into lrelu(0.1)+prelu, or elu, or lrelu(0.1) only.
		TR_CONST["activations"] = ['lrelu'] 
		TR_CONST["activations_fc_layers"] = ['lrelu']
		TR_CONST["BN"] = True
		TR_CONST["BN_fc_layers"] = True
		TR_CONST["!memo"] = 'lrelu. with alpha=0.03 again.'
		TR_CONST["dropouts_fc_layers"] = [0.5]
		TR_CONST["nums_units_fc_layers"] = [1024] # with 0.25 this is equivalent to 512 units
		TR_CONST["num_layers"] = 4

		# 01-19-18h50_blue_shibe --> no activation in conv layer
		# 27148/27148 [==============================] - 30s - loss: 0.1506 - acc: 0.9488 - val_loss: 0.1574 - val_acc: 0.9471
		# 27148/27148 [==============================] - 29s - loss: 0.1244 - acc: 0.9545 - val_loss: 0.1500 - val_acc: 0.9489
		# Okay, not bad. bit overfitting because there was no dropout. and it's very fast so I can do something more
		# roc_auc_none 0.501975403804 0.547490792132

		# with dropout and BN,
		# 27148/27148 [==============================] - 41s - loss: 0.1372 - acc: 0.9510 - val_loss: 0.1411 - val_acc: 0.9500
		# roc_auc_none 0.5 0.569980667246
		TR_CONST["!memo"] = 'go_mfcc_convnet_model, ignore all other info here'
		TR_CONST['model_type'] = 'gnu_mfcc'
		TR_CONST['tf_type'] = 'mfcc'
		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)

		# 01-19-19h09_horse_paw --> no activation in conv layer
		# 01-19-20h44_aqua_koala --> new! and more complex.
		# it's too slow, perhaps because there is not much pooling. 
		# 27148/27148 [==============================] - 959s - loss: 0.1569 - acc: 0.9479 - val_loss: 0.1563 - val_acc: 0.9471
		# 27148/27148 [==============================] - 957s - loss: 0.1530 - acc: 0.9485 - val_loss: 0.1613 - val_acc: 0.9442
		# roc_auc_none 0.5 0.527866826088

		TR_CONST["!memo"] = 'design_gnu_convnet_model, ignore all other info here'
		TR_CONST['model_type'] = 'gnu_1d'
		TR_CONST['tf_type'] = 'melgram'
		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)
		sys.exit()



	
	# default - BN(y,n), dropout(n,y), 6-layer, with elu.01-19-23h09_tiny_horse
	# 27148/27148 [==============================] - 400s - loss: 0.1651 - acc: 0.9468 - val_loss: 0.1619 - val_acc: 0.9465

	# 01-20-00h13_aqua_shep
	# elu, vgg_modi_1x1, l=2 (effectively 4 layers), dropout(n,y), BN(y,n)
	# + lr=0.1 !!!! (shit, even worse than original lr = 1e-2)
	# 27148/27148 [==============================] - 649s - loss: 0.1662 - acc: 0.9467 - val_loss: 0.1640 - val_acc: 0.9463
	# 27148/27148 [==============================] - 649s - loss: 0.1609 - acc: 0.9474 - val_loss: 0.1538 - val_acc: 0.9477
	# 27148/27148 [==============================] - 648s - loss: 0.1514 - acc: 0.9483 - val_loss: 0.1509 - val_acc: 0.9480

	# Mistake above: to omit BN.
	# do it again with BN!
	# 01-20-01h48_musky_roo, 1024 units, fc dropouts, bn (y,y), l=2, vgg_modi_1x1, elu
	# 27148/27148 [==============================] - 662s - loss: 0.1511 - acc: 0.9483 - val_loss: 0.1499 - val_acc: 0.9477
	
	# shit it was so fast. probably because it doesn't have dropouts in conv layers.
	# 01-20-05h03_musky_bat
	# 27148/27148 [==============================] - 662s - loss: 0.1499 - acc: 0.9487 - val_loss: 0.1462 - val_acc: 0.9485
	# 27148/27148 [==============================] - 662s - loss: 0.1401 - acc: 0.9503 - val_loss: 0.1429 - val_acc: 0.9489
	# 27148/27148 [==============================] - 661s - loss: 0.1424 - acc: 0.9503 - val_loss: 0.1420 - val_acc: 0.9494
	# 27148/27148 [==============================] - 661s - loss: 0.1336 - acc: 0.9518 - val_loss: 0.1414 - val_acc: 0.9489
	# 27148/27148 [==============================] - 661s - loss: 0.1336 - acc: 0.9518 - val_loss: 0.1414 - val_acc: 0.9489
	# ...
	# ... after 7 iterations
	# 27148/27148 [==============================] - 662s - loss: 0.1514 - acc: 0.9482 - val_loss: 0.1452 - val_acc: 0.9494
	# roc_auc_none 0.499981096408 0.584840045702
	# and resume! with adam from now.		
	# pooling only by time for the first two.

	# Okay, with less dropout. 0.2
	# 27148/27148 [==============================] - 1357s - loss: 0.1684 - acc: 0.9456 - val_loss: 0.1578 - val_acc: 0.9472
	# 27148/27148 [==============================] - 1356s - loss: 0.1640 - acc: 0.9462 - val_loss: 0.1564 - val_acc: 0.9474
	# 27148/27148 [==============================] - 1356s - loss: 0.1576 - acc: 0.9468 - val_loss: 0.1479 - val_acc: 0.9493
	# result: no dropout at convnet. yeah......
	# roc_auc_none 0.507071232823 0.561989803816


	# 01-20-16h57_shark_wing
	# elu, batch(y,y), dropout(n,y), 3x2 conv layers (vgg_modi_1x1), 3(256-256-256) fc layers
	# from now, just once in a whole epoch.
	# 27148/27148 [==============================] - 691s - loss: 0.1533 - acc: 0.9479 - val_loss: 0.1464 - val_acc: 0.9488
	# 27148/27148 [==============================] - 643s - loss: 0.1489 - acc: 0.9484 - val_loss: 0.1426 - val_acc: 0.9502
	# 27148/27148 [==============================] - 701s - loss: 0.1421 - acc: 0.9501 - val_loss: 0.1388 - val_acc: 0.9499
	# 27148/27148 [==============================] - 700s - loss: 0.1388 - acc: 0.9506 - val_loss: 0.1350 - val_acc: 0.9509``````````````
	# 27148/27148 [==============================] - 725s - loss: 0.1391 - acc: 0.9507 - val_loss: 0.1360 - val_acc: 0.9502
	# roc_auc_none 0.5 0.607832063532
	

	# 01-20-22h18_gryph_paw - maxout.  #### PROMISING.... ###
	# Bn (y,y) - not sure if it's working right with maxout.
	# dropout(n,y), all elu, 3-layer vgg_modi_1x1
	# 27148/27148 [==============================] - 984s - loss: 0.1452 - acc: 0.9497 - val_loss: 0.1432 - val_acc: 0.9480
	# 27148/27148 [==============================] - 983s - loss: 0.1265 - acc: 0.9536 - val_loss: 0.1400 - val_acc: 0.9484
	# roc_auc_none 0.5 0.611699031992


	# 01-21-03h04_rage_roo
	# with more complex setting, which results in overfitting.
	# roc_auc_none 0.5 0.614563063192
	# at the end,
	# 27148/27148 [==============================] - 983s - loss: 0.1265 - acc: 0.9536 - val_loss: 0.1400 - val_acc: 0.9484
	

	# 01-21-12h22_blue_mare
	# rage_roo was with wrong convnet BN axis, so do it again + additionally with l1-reg in fc layer.
	# 27148/27148 [==============================] - 804s - loss: 0.1328 - acc: 0.9519 - val_loss: 0.1335 - val_acc: 0.9514
	# 27148/27148 [==============================] - 708s - loss: 0.1267 - acc: 0.9532 - val_loss: 0.1362 - val_acc: 0.9497
	# 27148/27148 [==============================] - 804s - loss: 0.1236 - acc: 0.9541 - val_loss: 0.1354 - val_acc: 0.9501
	# 27148/27148 [==============================] - 789s - loss: 0.1098 - acc: 0.9585 - val_loss: 0.1503 - val_acc: 0.9441
	# WE NEED REGULARIZATION. Dropout on convnet again!
	# roc_auc_none 0.5 0.60138259003

	# 01-21-17h03_big_puma
	# with dropout in convnet,
	# 27148/27148 [==============================] - 936s - loss: 0.1442 - acc: 0.9500 - val_loss: 0.1961 - val_acc: 0.9323
	# 27148/27148 [==============================] - 933s - loss: 0.1368 - acc: 0.9512 - val_loss: 0.1427 - val_acc: 0.9494
	# 27148/27148 [==============================] - 933s - loss: 0.1221 - acc: 0.9548 - val_loss: 0.1502 - val_acc: 0.9455
	# 27148/27148 [==============================] - 933s - loss: 0.1128 - acc: 0.9578 - val_loss: 0.1564 - val_acc: 0.9432
	
	# 01-22-01h30_dark_bear
	# put a very strong l1 regulariser for maxout and conv. (0.01), and dropout at conv:0.25 -> 0.30
	# not learning well. Probably dropout is wrong..

	# 01-22-02h31_sharp_shark
	# remove dropout.
	# 27148/27148 [==============================] - 1092s - loss: 5.9497 - acc: 0.9462 - val_loss: 0.1930 - val_acc: 0.9432
	# 27148/27148 [==============================] - 1090s - loss: 5.8483 - acc: 0.9464 - val_loss: 0.1724 - val_acc: 0.9461
	# 27148/27148 [==============================] - 1090s - loss: 5.8221 - acc: 0.9463 - val_loss: 0.1745 - val_acc: 0.9453
	# 27148/27148 [==============================] - 1094s - loss: 5.7915 - acc: 0.9464 - val_loss: 0.1725 - val_acc: 0.9451

	#okay, stop l1 reg on convnet..
	# should find something better than gryph_paw and blue_mare!

	# 01-22-09h40_spotty_lynx
	# mae doesn't work well.


	# mse 01-22-10h45_wolfy_wolf
	# stop at the similar point. 
	# 27148/27148 [==============================] - 721s - loss: 0.0417 - acc: 0.9488 - val_loss: 0.0411 - val_acc: 0.9483
	#...
	# 27148/27148 [==============================] - 717s - loss: 0.0366 - acc: 0.9542 - val_loss: 0.0396 - val_acc: 0.9490
	# roc_auc_none 0.604942623177
	# should look into predict value, how unique vector does it predict. 

	# 01-22-17h02_musky_orca
	# l2 on conv only,
	# after 3 iteration,
	# 27148/27148 [==============================] - 714s - loss: 0.1427 - acc: 0.9503 - val_loss: 0.1503 - val_acc: 0.9491
	# roc_auc_none 0.597105533848

	# hinge loss
	# It sucks. Even pre-learned weights are getting worse. 

	# l2 conve and mse,
	# 01-22-23h17_pup_wing


	# add gaussian noise of sgm=0.1 to input of every conv layer?
	# for sigma in [0.05, 0.3, 1.0]:
	# 	TR_CONST['gn_sigma'] = sigma
	# 	update_setting_dict(TR_CONST)
	# 	run_with_setting(TR_CONST, sys.argv)

	# 01-23-13h31_shark_toy sgm=0.01
	# result: still overfit after 0.945x. 

	# 01-23-13h31_shark_toy
	# have very large fc layer - probably this is the bottleneck.
	# .. what?
	# roc_auc_none 0.560626304588 : it's not bad. will resume more 
	# e.g. one 4096 layers. 
	# TR_CONST['gn_sigma'] = 0.01

	# with 0.01 noise and single 4096 layer
	# memory error.

	# vgg_modi_3x3: more deeper. with 0.01 noise. with l2 conv(shit, mistake), maxout. dropout only fc. All BN. 
	# 01-23-17h21_doge_wing	
	# 27148/27148 [==============================] - 1101s - loss: 0.1546 - acc: 0.9480 - val_loss: 0.1704 - val_acc: 0.9401
	# 27148/27148 [==============================] - 1101s - loss: 0.1429 - acc: 0.9503 - val_loss: 0.2163 - val_acc: 0.9319
	# roc_auc_none 0.599799698513


	# no noise, no l2
	# 01-24-00h31_red_pig
	# 27148/27148 [==============================] - 1102s - loss: 0.1528 - acc: 0.9482 - val_loss: 0.1450 - val_acc: 0.9484
	# 27148/27148 [==============================] - 1099s - loss: 0.1460 - acc: 0.9495 - val_loss: 0.1415 - val_acc: 0.9486
	# 27148/27148 [==============================] - 1098s - loss: 0.1415 - acc: 0.9505 - val_loss: 0.1373 - val_acc: 0.9490
	# 27148/27148 [==============================] - 1099s - loss: 0.1389 - acc: 0.9510 - val_loss: 0.1356 - val_acc: 0.9501
	# roc_auc_macro 0.615940685831 --> best?

	results = {}
	nl = 5
	# TR_CONST["!memo"] = '4 layers, 4096 fc, 2 fc layers'
	# TR_CONST["num_layers"] = 4
	# TR_CONST['gaussian_noise'] = False
	# TR_CONST["regulariser"] = [('l2', 0.0)]*TR_CONST["num_layers"] # use [None] not to use.
	# TR_CONST['nums_units_fc_layers'] = [4096]
	# TR_CONST['num_fc_layers'] = 2


	#-----
	# from here, I change MaxPooling size, increased it overall so that final image would be very small (4x2 or 4x3)

	# 01-24-19h43_cyan_puma
	# 01-24-21h09_wet_cat
	# 16968/16968 [==============================] - 1690s - loss: 0.1652 - acc: 0.9465 - val_loss: 0.2980 - val_acc: 0.9034
	# 2x4096,..
	# with 3 layers: acc<0.9.
	# very bad. and very very slow. (over 2k sec under 5-sub-epoch sysrtem)

	# 01-24-22h38_rage_gryph
	# 3x512 wth new setting of MP units. 
	# large MP size on low layer may go wrong.
	# even it's not working well. so this new feature map width is not working? or problem is on MP?

	# 01-25-02h23_sharp_shep
	# give up 'almost fully convolutioanal', go back to red_pig.
	update_setting_dict(TR_CONST)
	acc = run_with_setting(TR_CONST, argv=sys.argv, batch_size=batch_size)	
	results[TR_CONST["!memo"]] = acc
	pprint.pprint(results)

	# 01-25-04h55_tiny_dobie (bit richer than red_pig, because in red_pig intermediat 3x3 conv feature map width was accidently set to 32 for all.)
	# + 2x1024 isntead of 3x512
	# OMG there was noise!! for the whole day !!! from 01-24- except red_pig!

	# again without noise


	sys.exit(0)

	for nl in [4]:
		if nl == 5:
			TR_CONST["num_layers"] = nl
			TR_CONST['gaussian_noise'] = False
			TR_CONST["regulariser"] = [('l2', 0.0)]*TR_CONST["num_layers"] # use [None] not to use.
			update_setting_dict(TR_CONST)
			acc = run_with_setting(TR_CONST, argv=sys.argv, batch_size=batch_size)	
			
			

		# with noise, no l2.
		TR_CONST["!memo"] = '4 layer, small noise, 0.005'
		TR_CONST["num_layers"] = nl
		TR_CONST['gaussian_noise'] = True
		TR_CONST['gn_sigma'] = 0.005
		TR_CONST["regulariser"] = [('l2', 0.0)]*TR_CONST["num_layers"] # use [None] not to use.
		update_setting_dict(TR_CONST)
		acc=run_with_setting(TR_CONST, argv=sys.argv, batch_size=batch_size)	
		results[TR_CONST["!memo"]] = acc
		pprint.pprint(results)
		
		# dropout on conv
		TR_CONST["!memo"] = '4 layer, 0.2 dropout on conv as well'
		TR_CONST["num_layers"] = nl
		TR_CONST['gaussian_noise'] = False
		TR_CONST['dropouts'] = 0.2
		update_setting_dict(TR_CONST)
		acc=run_with_setting(TR_CONST, argv=sys.argv, batch_size=batch_size)	
		results[TR_CONST["!memo"]] = acc
		

		# l2 on conv
		TR_CONST["!memo"] = '4 layer, l2 reg.'
		TR_CONST['dropouts'] = 0.0
		TR_CONST["num_layers"] = nl
		TR_CONST['gaussian_noise'] = False
		TR_CONST["regulariser"] = [('l2', 0.0001)]*TR_CONST["num_layers"] # use [None] not to use.
		update_setting_dict(TR_CONST)
		acc=run_with_setting(TR_CONST, argv=sys.argv, batch_size=batch_size)	
		results[TR_CONST["!memo"]] = acc
		pprint.pprint(results)


	# TODO : major voting to evaluate more accurately.

	# and 3x3s2 mp? as sander did in plankton work.

	# Shit I want to use Lincoln's gpu. 
	# Rule of thumb:
	# batch_size = 32 or 64
	# dropout = True for only fc_layers.
	# mfcc + some tf_type make sense.
	# lrelu > prelu, relu. 