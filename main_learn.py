# -*- coding: utf-8 -*-

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

def get_fit_dict(train_x, train_y, dim_labels):
	fit_dict = {}
	fit_dict['input'] = train_x
	for dense_idx in xrange(dim_labels):
		output_node_name = 'output_%d' % dense_idx
		fit_dict[output_node_name] = train_y[:, dense_idx:dense_idx+1]

	return fit_dict

def merge_multi_outputs(predicted_dict):
	dim_label = len(predicted_dict.keys())
	num_data = predicted_dict[predicted_dict.keys()[0]].shape[0]
	predicted = np.zeros((num_data, dim_label))
	for i in range(dim_label):
		predicted[:, i] = predicted_dict['output_%d'%i][:, 0]
	return predicted

def update_setting_dict(setting_dict):

	setting_dict["num_feat_maps"] = [setting_dict["num_feat_maps"][0]]*setting_dict["num_layers"]
	setting_dict["activations"] = [setting_dict["activations"][0]] *setting_dict["num_layers"]
	setting_dict["dropouts"] = [setting_dict["dropouts"][0]]*setting_dict["num_layers"]
	setting_dict["regulariser"] = [setting_dict["regulariser"][0]]*setting_dict["num_layers"]

	setting_dict["dropouts_fc_layers"] = [setting_dict["dropouts_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["nums_units_fc_layers"] = [setting_dict["nums_units_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["activations_fc_layers"] = [setting_dict["activations_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["regulariser_fc_layers"] = [setting_dict["regulariser_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["act_regulariser_fc_layers"] = [setting_dict["act_regulariser_fc_layers"][0]]*setting_dict["num_fc_layers"]

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
	
	
	best_auc = 0.0
	# label_matrix = np.load(PATH_DATA + FILE_DICT['sorted_merged_label_matrix'])
	# label_matrix = label_matrix[:, :dim_labels]
	hdf_xs = io.load_x(hyperparams['tf_type'], is_test=hyperparams['is_test'])
	hdf_ys = io.load_y(dim_labels, is_test=hyperparams['is_test'], merged=hyperparams['merged'])
	hdf_train_xs = hdf_xs[:12]
	hdf_valid_xs = hdf_xs[12:13]
	hdf_test_xs = hdf_xs[13:]
	hdf_train_ys = hdf_ys[:12]
	hdf_valid_ys = hdf_ys[12:13]
	hdf_test_ys = hdf_ys[13:]

	# train_x, valid_x, test_x = io.load_x(hyperparams['tf_type'])
	# train_y, valid_y, test_y = io.load_y(dim_labels)
	if hyperparams['is_test']:
		pdb.set_trace()
		# num_data_in_test = 256
		# train_x = train_x[:num_data_in_test]
		# valid_x = valid_x[:num_data_in_test]
		# test_x  = test_x[:num_data_in_test]
		# train_y = train_y[:num_data_in_test]
		# valid_y = valid_y[:num_data_in_test]
		# test_y  = test_y[:num_data_in_test]
		# shuffle = False
		# num_sub_epoch = 1
		
	hyperparams['height_image'] = hdf_train_xs[0].shape[2]
	hyperparams["width_image"]  = hdf_train_xs[0].shape[3]
	
	hp_manager = hyperparams_manager.Hyperparams_Manager()

	# name, path, ...
	nickname = hp_manager.get_name(hyperparams)
	timename = time.strftime('%m-%d-%Hh%M')
	if hyperparams["is_test"]:
		model_name = 'test_' + nickname
	else:
		model_name = timename + '_' + nickname
	if hyperparams['resume'] != '':
		model_name = model_name + '_from_' + hyperparams['resume']
	hp_manager.save_new_setting(hyperparams)
	print '-'*60
	print 'model name: %s' % model_name
	model_name_dir = model_name + '/'
	model_weight_name_dir = 'w_' + model_name + '/'
	fileout = model_name + '_results'
	
 	# build model
 	model = my_keras_models.build_convnet_model(setting_dict=hyperparams)
 	if not os.path.exists(PATH_RESULTS + model_name_dir):
		os.mkdir(PATH_RESULTS + model_name_dir)
		os.mkdir(PATH_RESULTS + model_name_dir + 'images/')
		os.mkdir(PATH_RESULTS + model_name_dir + 'plots/')
		os.mkdir(PATH_RESULTS_W + model_weight_name_dir)
	hp_manager.write_setting_as_texts(PATH_RESULTS + model_name_dir, hyperparams)
 	hp_manager.print_setting(hyperparams)
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
	print '--- %s train starts. Remove will_stop.keunwoo to continue learning after %d epochs ---' % (model_name, hyperparams["num_epoch"])
	
	num_epoch = hyperparams["num_epoch"]
	total_epoch = 0
	
	callbacks = [weight_image_monitor]

	total_history = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}

	# total_label_count = np.sum([hdf_train.shape[0]*hdf_train.shape[1] for hdf_train in hdf_train_ys]) 
	# total_zeros = 
	# print 'With predicting all zero, acc is %0.6f' % ((total_label_count - np.sum(train_y))/float(total_label_count))

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
	 	print '--TEST FLIGHT--'
	 	if hyperparams['model_type'] in ['multi_task']:
	 		
	 		fit_dict = get_fit_dict(hdf_train_xs[-1][-256:], hdf_train_ys[-1][-256:], hyperparams['dim_labels'])
 			# pdb.set_trace()
 			model.fit(fit_dict,	batch_size=batch_size,	nb_epoch=1, shuffle='batch')
	 	else:
	 		model.fit(hdf_train_xs[-1][-256:], hdf_train_ys[-1][-256:], 
			 		validation_data=(hdf_valid_xs[0][:512], hdf_valid_ys[0][:512]), 
					batch_size=batch_size, 
					nb_epoch=1, 
					show_accuracy=hyperparams['isClass'], 
					callbacks=callbacks,
					shuffle='batch')
	 	print '--TEST FLIGHT DONE: %s--' % model_name
	 	total_epoch_count = 0
		while True:
			for sub_epoch_idx, (train_x, train_y) in enumerate(zip(hdf_train_xs, hdf_train_ys)):
				total_epoch_count += 1
				if os.path.exists('stop_asap.keunwoo'):
					break
				# early_stop should watch overall AUC rather than val_loss or val_acc
				# [run]
			 	if hyperparams['model_type'] in ['multi_task']:
 					fit_dict = get_fit_dict(train_x, train_y, hyperparams['dim_labels'])
 					loss_history = model.fit(fit_dict,
 							batch_size=batch_size,
 							nb_epoch=1,
 							shuffle='batch')
			 	else:
					loss_history = model.fit(train_x, train_y, validation_data=(hdf_valid_xs[0][:2048], hdf_valid_ys[0][:2048]), 
											batch_size=batch_size,
											nb_epoch=1, 
											show_accuracy=hyperparams['isClass'], 
											verbose=1, 
											callbacks=callbacks,
											shuffle='batch')
				# [validation]
				if not sub_epoch_idx in [0, 6]: # validation with subset
					
					if hyperparams['model_type'] in ['multi_task']:
						fit_dict = get_fit_dict(hdf_valid_xs[-1][:], hdf_valid_ys[-1][:], hyperparams['dim_labels'])
						predicted_dict = model.predict(fit_dict, batch_size=batch_size)
						predicted = merge_multi_outputs(predicted_dict)
					else:
						valid_x, valid_y = (hdf_valid_xs[0][:2048], hdf_valid_ys[0][:2048])
						predicted = model.predict(valid_x, batch_size=batch_size)
				else: # validation with all
					print ' * Compute AUC with full validation data for model: %s.' % model_name
					if hyperparams['model_type'] in ['multi_task']:
						valid_y = hdf_valid_ys[0][:] # I know I'm using only one set for validation.
						fit_dict = get_fit_dict(hdf_valid_xs[-1][:], hdf_valid_ys[-1][:], hyperparams['dim_labels'])
						predicted_dict = model.predict(fit_dict, batch_size=batch_size)
						predicted = merge_multi_outputs(predicted_dict)
						val_loss_here = model.evaluate(fit_dict, batch_size=batch_size)
					else:
						predicted = np.zeros((0, dim_labels))
						valid_y = np.zeros((0, dim_labels))
						for valid_x_partial, valid_y_partial in zip(hdf_valid_xs, hdf_valid_ys):
							predicted = np.vstack((predicted, model.predict(valid_x_partial, batch_size=batch_size)))
							valid_y = np.vstack((valid_y, valid_y_partial))

				# [check if should stop]
				val_result = evaluate_result(valid_y, predicted)
				history = {}
				history['auc'] = [val_result['roc_auc_macro']]
				if hyperparams['model_type'] in ['multi_task']:
					history['val_loss'] = [val_loss_here]
				print '[%d] AUC: %f' % (total_epoch_count, val_result['roc_auc_macro'])
				if val_result['roc_auc_macro'] > best_auc:
					print ', which is new record! it was %f btw (%s)' % (best_auc, model_name)
					best_auc = val_result['roc_auc_macro']
					model.save_weights(filepath=PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5", 
										overwrite=True)
				else:
					print 'Keep old auc record, %f' % best_auc
				append_history(total_history, history)
				append_history(total_history, loss_history.history)

				my_plots.export_list_png(total_history['auc'], out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'auc_plots.png', title=model_name + 'AUC' + '\n'+hyperparams['!memo'] )
				my_plots.export_history(total_history['loss'], total_history['val_loss'], 
													acc=total_history['acc'], 
													val_acc=total_history['val_acc'], 
													out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'loss_plots.png')
		

			print '[%d], %d-th of %d epoch is complete, auc:%f' % (total_epoch_count, total_epoch, num_epoch, val_result['roc_auc_macro'])
			total_epoch += 1

			if os.path.exists('stop_asap.keunwoo'):
				os.remove('stop_asap.keunwoo')
				break			
			if os.path.exists('will_stop.keunwoo'):	
				if total_epoch > num_epoch:
					break
				else:
					print ' *** will go for %d epochs' % (num_epoch - total_epoch)
			else:
				print ' *** will go for another one epoch. '
				print ' *** $ touch will_stop.keunwoo to stop at the end of this, otherwise it will be endless.'
	# [summarise]
	if hyperparams["debug"] == True:
		pdb.set_trace()
	if not hyperparams['is_test']:
		if not best_auc == val_result['roc_auc_macro']: # load weights only it's necessary
			model.load_weights(PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5") 
	
	predicted = np.zeros((0, dim_labels))
	test_y = np.zeros((0, dim_labels))

	for test_x_partial, test_y_partial in zip(hdf_test_xs, hdf_test_ys):
		if hyperparams['model_type'] in ['multi_task']:
			fit_dict = get_fit_dict(test_x_partial[:], test_y_partial[:], hyperparams['dim_labels'])
			predicted_dict = model_predict(fit_dict, bath_size=batch_size)
			predicted = np.vstack((predicted, merge_multi_outputs(predicted_dict)))
		else:
			predicted = np.vstack((predicted, model.predict(test_x_partial, batch_size=batch_size)))
		test_y = np.vstack((test_y, test_y_partial))
	eval_result_final = evaluate_result(test_y, predicted)
	print '.'*60
	for key in sorted(eval_result_final.keys()):
		print key, eval_result_final[key]
	print '.'*60
	
	#save results

	cP.dump(total_history, open(PATH_RESULTS + model_name_dir + 'total_history.cP', 'w'))
	# np.save(PATH_RESULTS + model_name_dir + 'loss_testset.npy', loss_testset)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_result.npy', [predicted, test_y])
	np.save(PATH_RESULTS + model_name_dir + 'weights_changes.npy', np.array(weight_image_monitor.weights_changes))

	# ADD weight change saving code
	if total_history != {}:
		
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
	parser.add_argument('-act_fc', '--activations_fc', help='activations - relu, lrelu, prelu, elu \ndefault=relu', 
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
	parser.add_argument('-merged', '--merged', type=str,
										help='merged labels (synonyms) or not',
										required=False )
	parser.add_argument('-num_mo', '--num_maxout_feature', type=int,
										help='number of maxout features',
										required=False )
	parser.add_argument('-act_reg_fc', '--act_regulariser_fc', type=float,
										help='activity regulariser',
										required=False)
	
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

	TR_CONST["num_epoch"] = 1
	TR_CONST["dropouts"] = [0.0]*TR_CONST["num_layers"]
	TR_CONST["num_feat_maps"] = [32]*TR_CONST["num_layers"]
	TR_CONST["activations"] = ['elu']*TR_CONST["num_layers"]

	TR_CONST["BN"] = True
	TR_CONST["regulariser"] = [('l2', 0.0)]*TR_CONST["num_layers"] # use [None] not to use.
	TR_CONST["model_type"] = 'vgg_modi_1x1'
	TR_CONST["tf_type"] = 'melgram'

	TR_CONST["num_fc_layers"] = 2

	TR_CONST["BN_fc_layers"] = True
	TR_CONST["dropouts_fc_layers"] = [0.5]*TR_CONST["num_fc_layers"]

	TR_CONST["nums_units_fc_layers"] = [2048]*TR_CONST["num_fc_layers"]
	TR_CONST["activations_fc_layers"] = ['elu']*TR_CONST["num_fc_layers"]
	TR_CONST["regulariser_fc_layers"] = [('l1', 0.0)] *TR_CONST["num_fc_layers"]
	TR_CONST["act_regulariser_fc_layers"] = [('activity_l1l2', 0.0)] *TR_CONST["num_fc_layers"]
	TR_CONST["BN_fc_layers"] = True
	TR_CONST["maxout"] = True
	TR_CONST["gaussian_noise"] = False
	TR_CONST['merged'] = False
	TR_CONST['nb_maxout_feature'] = 4
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
	if args.activations_fc:
		TR_CONST["activations_fc_layers"] = [args.activations_fc] * TR_CONST["num_fc_layers"]
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
	if not args.regulariser == 0.0:
		TR_CONST["regulariser"] = [(TR_CONST["regulariser"][0][0], args.regulariser)]*TR_CONST["num_layers"]
	if not args.regulariser_fc == 0.0:
		TR_CONST["regulariser_fc_layers"] = [(TR_CONST["regulariser_fc_layers"][0][0], args.regulariser_fc)]*TR_CONST["num_fc_layers"]
	if not args.act_regulariser_fc == 0.0:
		TR_CONST["act_regulariser_fc_layers"] = [(TR_CONST["act_regulariser_fc_layers"][0][0], args.act_regulariser_fc)]*TR_CONST["num_fc_layers"]
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
	if args.merged:
		TR_CONST["merged"] = str2bool(args.merged)
	if args.num_maxout_feature:
		TR_CONST['nb_maxout_feature'] = args.num_maxout_feature
 	#----------------------------------------------------------#
	
	update_setting_dict(TR_CONST)
	auc = run_with_setting(TR_CONST, argv=sys.argv, batch_size=batch_size)	

	#
	
	