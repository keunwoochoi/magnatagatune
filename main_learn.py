import numpy as np
import sys
import os
import time
import h5py
import argparse
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
	ret['log_loss_before'] = metrics.log_loss(y_true, y_pred)
	y_pred = np.round(y_pred)
	ret['precision'] = metrics.average_precision_score(y_true, y_pred)
	ret['f1_binary'] = metrics.f1_score(y_true, y_pred, average='binary')
	ret['f1_micro'] = metrics.f1_score(y_true, y_pred, average='micro')
	ret['f1_macro'] = metrics.f1_score(y_true, y_pred, average='macro')
	ret['log_loss_after'] = metrics.log_loss(y_true, y_pred)
	ret['roc_auc_micro'] = metrics.roc_auc_score(y_true, y_pred, average='micro')
	ret['roc_auc_macro'] = metrics.roc_auc_score(y_true, y_pred, average='macro')
	ret['roc_auc_none'] = metrics.roc_auc_score(y_true, y_pred)
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

def run_with_setting(hyperparams, argv=None):
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
		test_x = test_x[:num_data_in_test]
		train_y = train_y[:num_data_in_test]
		valid_y = valid_y[:num_data_in_test]
		test_y = test_y[:num_data_in_test]
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
	checkpointer = keras.callbacks.ModelCheckpoint(filepath=PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5", 
													verbose=1, 
								             		save_best_only=True)
	weight_image_monitor = my_keras_utils.Weight_Image_Saver(PATH_RESULTS + model_name_dir + 'images/')
	patience = 100
	if hyperparams["is_test"] is True:
		patience = 99999999
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
														patience=patience, 
														verbose=0)
	

	if hyperparams["tf_type"] == 'cqt':
		batch_size = 64
	elif hyperparams["tf_type"] == 'stft':
		batch_size = 32
	elif hyperparams["tf_type"] == 'mfcc':
		batch_size = 128
	elif hyperparams["tf_type"] == 'melgram':
		batch_size = 64
	else:
		raise RuntimeError('batch size for this? %s' % hyperparams["tf_type"])
	if hyperparams['model_type'] == 'vgg_original':
		batch_size = (batch_size * 3)/5
	# ready to run
	predicted = model.predict(test_x, batch_size=batch_size)
	eval_result_init = evaluate_result(test_y, predicted)
	if hyperparams['debug'] == True:
		pdb.set_trace()
	# print 'mean of target value:'
	# print np.mean(test_y, axis=0)
	# print 'mean of predicted value:'
	# print np.mean(predicted, axis=0)
	# print 'mse with just predicting average is %f' % np.mean((test_y - np.mean(test_y, axis=0))**2)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_init.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])
	print '--- train starts. Remove will_stop.keunwoo to continue learning after %d epochs ---' % hyperparams["num_epoch"]
	
	num_epoch = hyperparams["num_epoch"]
	total_epoch = 0
	if hyperparams['is_test']:
		callbacks = [weight_image_monitor]
	else:
		callbacks = [weight_image_monitor, early_stopping, checkpointer]

	total_history = {}
	if hyperparams['resume'] != '':
		if os.path.exists(PATH_RESULTS_W + 'w_' + hyperparams['resume']):
			model.load_weights(PATH_RESULTS_W + 'w_' + hyperparams['resume'] + '/weights_best.hdf5')
		if os.path.exists(PATH_RESULTS + hyperparams['resume'] + '/total_history.cP'):
			previous_history = cP.load(open(PATH_RESULTS + hyperparams['resume'] + '/total_history.cP', 'r'))
			print 'previously learned weight: %s is loaded ' % hyperparams['resume']
			append_history(total_history, previous_history)

	my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
										filename_prefix='local_INIT', 
										normalize='local', 
										mono=True)
	my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
										filename_prefix='global_INIT', 
										normalize='global', 
										mono=True)

 	# run
	while True:	
		
		for sub_epoch_idx in range(num_sub_epoch):
			if os.path.exists('stop_asap.keunwoo'):
				break
			seg_from = sub_epoch_idx * (train_x.shape[0]/num_sub_epoch)
			seg_to   = (sub_epoch_idx+1) * (train_x.shape[0]/num_sub_epoch)
			train_x_here = train_x[seg_from:seg_to]
			train_y_here = train_y[seg_from:seg_to]
			if sub_epoch_idx in [2, 4]:
				valid_data = (valid_x, valid_y)
			else:
				valid_data = None

			history=model.fit(train_x_here, train_y_here, validation_data=valid_data, 
														batch_size=batch_size, 
														nb_epoch=1, 
														show_accuracy=hyperparams['isClass'], 
														verbose=1, 
														callbacks=callbacks,
														shuffle=shuffle)
			append_history(total_history, history.history)

		print '%d-th of %d epoch is complete' % (total_epoch, num_epoch)
		total_epoch += 1
		if os.path.exists('stop_asap.keunwoo'):
			os.remove('stop_asap.keunwoo')
			loss_testset = model.evaluate(test_x, test_y, show_accuracy=True, batch_size=batch_size)
			break
		
		if os.path.exists('will_stop.keunwoo'):	
			if total_epoch > num_epoch:
				loss_testset = model.evaluate(test_x, test_y, show_accuracy=True, batch_size=batch_size)
				break				
		else:
			print ' *** will go for another one epoch. '
			print ' *** $ touch will_stop.keunwoo to stop at the end of this, otherwise it will be endless.'
	#
	best_batch = np.argmax(total_history['val_acc'])+1
	
	if hyperparams["debug"] == True:
		pdb.set_trace()
	if not hyperparams['is_test']:
		model.load_weights(PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5") 
	predicted = model.predict(test_x, batch_size=batch_size)
	eval_result_final = evaluate_result(test_y, predicted)
	print '.'*60
	for key in sorted(eval_result_final.keys()):
		print key, eval_result_init[key], eval_result_final[key]
	print '.'*60
	# print np.mean(test_y, axis=0)[:10]
	# print predicted[:2][:10]

	#save results
	cP.dump(total_history, open(PATH_RESULTS + model_name_dir + 'total_history.cP', 'w'))
	np.save(PATH_RESULTS + model_name_dir + 'loss_testset.npy', loss_testset)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_result.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])
	np.save(PATH_RESULTS + model_name_dir + 'weights_changes.npy', np.array(weight_image_monitor.weights_changes))

	# ADD weight change saving code	
	my_plots.export_history(total_history['loss'], total_history['val_loss'], 
												acc=total_history['acc'], 
												val_acc=total_history['val_acc'], 
												out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
	
	min_loss = np.max(total_history['val_acc'])
	best_batch = np.argmax(total_history['val_acc'])+1
	num_run_epoch = len(total_history['val_acc'])
	oneline_result = '%6.4f, acc %d_of_%d, %s' % (min_loss, best_batch, num_run_epoch, model_name)
	with open(PATH_RESULTS + model_name_dir + oneline_result, 'w') as f:
		pass
	f = open( (PATH_RESULTS + '%s_%s_acc_%06.4f_at_(%d_of_%d)_%s'  % \
		(timename, hyperparams["loss_function"], min_loss, best_batch, num_run_epoch, nickname)), 'w')
	f.close()
	with open('one_line_log.txt', 'a') as f:
		f.write(oneline_result)
		f.write(' ' + ' '.join(argv) + '\n')
	print '========== DONE: %s ==========' % model_name
	return min_loss

	
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
	
	
	args = parser.parse_args()
	#------------------- default setting --------------------------------#
	TR_CONST["dim_labels"] = 50
	TR_CONST["num_layers"] = 6

	TR_CONST['isClass'] = True
	TR_CONST['isRegre'] = False
	TR_CONST["clips_per_song"] = 7
	TR_CONST['loss_function'] = 'binary_crossentropy'
	TR_CONST["optimiser"] = 'adagrad'
	TR_CONST['learning_rate'] = 4e-1
	TR_CONST["output_activation"] = 'sigmoid'

	TR_CONST["num_epoch"] = 4
	TR_CONST["dropouts"] = [0.0]*TR_CONST["num_layers"]
	TR_CONST["num_feat_maps"] = [32]*TR_CONST["num_layers"]
	TR_CONST["activations"] = ['relu']*TR_CONST["num_layers"]
	TR_CONST["BN"] = True
	TR_CONST["regulariser"] = [('l2', 0.0)]*TR_CONST["num_layers"] # use [None] not to use.
	TR_CONST["model_type"] = 'vgg_simple'
	TR_CONST["tf_type"] = 'melgram'

	TR_CONST["num_fc_layers"] = 2

	TR_CONST["BN_fc_layers"] = False
	TR_CONST["dropouts_fc_layers"] = [0.0]*TR_CONST["num_fc_layers"]

	TR_CONST["nums_units_fc_layers"] = [512]*TR_CONST["num_fc_layers"]
	TR_CONST["activations_fc_layers"] = ['relu']*TR_CONST["num_fc_layers"]
	TR_CONST["regulariser_fc_layers"] = [('l2', 0.0), ('l2', 0.0)] 
	TR_CONST["BN_fc_layers"] = False 
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
	if args.debug:
		TR_CONST["debug"] = str2bool(args.debug)
	if args.output_layer:
		TR_CONST["output_activation"] = args.output_layer
	if args.resume:
		TR_CONST["resume"] = args.resume
	else:
		TR_CONST["resume"] = ''
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

		# then try dropout on FC only first. 01-19-00h53_red_wolf
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
		# after 6 epochs,
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

		# drop out of 0.5
		TR_CONST["activations"] = ['lrelu'] # alpha is 0.3 now
		TR_CONST["activations_fc_layers"] = ['lrelu']
		TR_CONST["BN"] = True
		TR_CONST["BN_fc_layers"] = True
		TR_CONST["num_layers"] = 4
		TR_CONST["!memo"] = 'bn on and on, 4layer, dropout on fc only, lrelu and lrelu, keep 32 per layer'
		TR_CONST["dropouts_fc_layers"] = [0.5]
		TR_CONST["nums_units_fc_layers"] = [1024] # with 0.25 this is equivalent to 512 units
		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)
	
	# vgg_simple, BN -true,true, num_layer in [3 and 6]
	TR_CONST["activations"] = ['lrelu'] # alpha is 0.3 now
	TR_CONST["activations_fc_layers"] = ['lrelu']
	TR_CONST["BN"] = True
	TR_CONST["BN_fc_layers"] = True
	
	TR_CONST["!memo"] = 'grid for layer number'
	TR_CONST["dropouts_fc_layers"] = [0.5]
	TR_CONST["nums_units_fc_layers"] = [1024] # with 0.25 this is equivalent to 512 units

	for num_l in [3,6]:
		TR_CONST["num_layers"] = num_l
		update_setting_dict(TR_CONST)
		run_with_setting(TR_CONST, sys.argv)
	sys.exit()
	# then small l2 weight decay on FC layers

	# then vgg original, conv-conv-mp

	# and 3x3s2 mp? as sander did in plankton work.

	# Shit I want to use Lincoln's gpu. 


	TR_CONST["num_layers"] = 3
	update_setting_dict(TR_CONST)
	run_with_setting(TR_CONST, sys.argv)

	# TR_CONST["num_layers"] = 4
	# update_setting_dict(TR_CONST)
	# run_with_setting(TR_CONST, sys.argv)
