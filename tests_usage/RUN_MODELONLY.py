import datetime as dt
import sys
import tensorflow as tf

from pymodconn.configs_init import get_configs
from pymodconn.model_gen_old import ModelClass

# takes argument as time_dt for all file saves and configuration json data
def runmain(time_dt, configs_data):
	tf.keras.backend.clear_session()
	model_class = ModelClass(configs_data, time_dt)
	model_class.build_model()
	print('model_class.model.inputs: ',model_class.model.inputs)
	print('model_class.model.outputs: ',model_class.model.outputs)


configs = get_configs('config_model.yaml')
configs['model']['if_model_image'] = 1
configs['model']['if_model_summary'] = 0

configs['model']['IF_GLU'] = 0
configs['model']['IF_ADDNORM'] = 0
configs['model']['IFFFN'] = 0
configs['model']['IFSELF_MHA'] = 0
configs['model']['IFCASUAL_MHA'] = 0
configs['model']['IFCROSS_MHA'] = 1
configs['rnn_units']['rnn_type'] = 'GRU'
configs['rnn_units']['input_enc_rnn_depth'] = 1
configs['rnn_units']['input_enc_rnn_bi'] = 1
configs['model']['control_future_cells'] = 1
configs['model']['all_layers_neurons'] = 64

ident = 'Decoders-%s_MHAfirst-%s_RNN-%s_Depth-%s_bi-%s_NNs-%s_' % (
		configs['model']['control_future_cells'],	
		configs['model']['MHA_RNN'],
		configs['rnn_units']['rnn_type'],
		configs['rnn_units']['input_enc_rnn_depth'],
		configs['rnn_units']['input_enc_rnn_bi'],
		configs['model']['all_layers_neurons'])

ident = 'test_'
current_run_dt = ident + str(dt.datetime.now().strftime('%d.%m.%Y-%H.%M.%S.%f')[:-3])
runmain(current_run_dt, configs)