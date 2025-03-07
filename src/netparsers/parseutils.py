'''All the parser functions is implemented here'''
import torch.nn as nn
import math
from src.layers.klmodules import *
from src.layers.Initializers import *

def parse_layer_opts(layer_string):
	'''input is a layer description string with name|p1:[v1],p2[v2]... convention'''
	layer_string.rstrip(' ')
	temp = layer_string.split('|')
	layer_name_str = temp[0]
	if len(temp)>1:
		layer_opts_string = temp[1]
		layer_opts_list = layer_opts_string.split(',')
	else:
		layer_opts_list =[]
	layer_opts = {}
	for param_value in layer_opts_list:
		param_value_list = param_value.split(':')
		if len(param_value_list)<2:
			raise Exception(param_value_list[0] + 'is not initialized')
		layer_opts[param_value_list[0]] = param_value_list[1]
	return layer_name_str,layer_opts

def evalpad(pad, ksize):
	if pad == 'same':
		totalpad = ksize - 1
		padding = int(math.floor(totalpad / 2))
	else:
		padding = 0
	return padding

def get_init(initstring:str)->Parameterizer:
	if 'stoch' in initstring:
		isstoch = True
	else:
		isstoch = False
	if 'unif' in initstring:
		isuniform = True
	else:
		isuniform = False
	if 'dirich' in initstring:
		isdirichlet = True
	else:
		isdirichlet = False
	if 'log' in initstring:
		if 'proj' in initstring:
			init = LogParameterProjector(isstoch=isstoch, isuniform=isuniform, isdirichlet=isdirichlet)
		else:
			init = LogParameter(isstoch=isstoch,isuniform=isuniform,isdirichlet=isdirichlet)
	elif 'sphere' in initstring:
		init = SphereParameter(isstoch=isstoch, isuniform=isuniform, isdirichlet=isdirichlet)
	return init

def parse_layer_string(layer_string,in_n_channel,in_icnum,blockidx_dict):
	out_n_channel = -1
	out_icnum = in_icnum
	layer_name_str,layer_opts = parse_layer_opts(layer_string)
	if layer_name_str not in blockidx_dict.keys():
		blockidx_dict[layer_name_str] = 1
	blockidx = blockidx_dict[layer_name_str]
	if layer_name_str == 'fin':
		return None,in_n_channel,out_icnum


	elif layer_name_str == 'conv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		coef = float(layer_opts['coef'])
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		pad = layer_opts['pad']
		layer = FConv(fnum=fnum,
		              input_ch=in_n_channel,
		              spsize=ksize,
		              stride=stride,
		              pad=pad,
		              init_coef=coef,
		              isbiased=isbiased,
		              blockidx=blockidx)
		out_n_channel = fnum

	elif layer_name_str == 'konvr':
		ksize = int(layer_opts['r'])
		out_states= int(layer_opts['f'])
		out_idcomp = int(layer_opts['icnum']) if 'icnum' in layer_opts.keys() else 1
		stride = int(layer_opts['stride'])
		init_type= (layer_opts['param'])
		coef = float(layer_opts['coef'])
		conc = float(layer_opts['conc'])
		train_conc = bool(layer_opts['trainconc'] in ['1', 'True'])
		cross = bool(layer_opts['cross'] in ['1', 'True']) if 'cross' in layer_opts.keys() else False
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		pad = layer_opts['pad']

		layer = Konv_R(out_states,
		               out_idcomp,
		               ksize,
		               in_n_channel,
		               in_icnum,
		               pad= pad,
		               stride = stride,
		               concentrate=conc,
		               train_conc=train_conc,
		               init_type=init_type,
		               init_coef=coef,
		               cross= cross,
		               is_biased=isbiased,
		              blockidx=blockidx)
		out_n_channel = out_states
		out_icnum = out_idcomp

	elif layer_name_str == 'konvs':
		ksize = int(layer_opts['r'])
		out_states= int(layer_opts['f'])
		out_idcomp = int(layer_opts['icnum']) if 'icnum' in layer_opts.keys() else 1
		stride = int(layer_opts['stride'])
		init_type= (layer_opts['param'])
		coef = float(layer_opts['coef'])
		conc = float(layer_opts['conc'])
		train_conc = bool(layer_opts['trainconc'] in ['1','True'])
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		pad = layer_opts['pad']

		layer = Konv_S(out_states,
		               out_idcomp,
		               ksize,
		               in_n_channel,
		               in_icnum,
		               pad= pad,
		               stride = stride,
		               init_type=init_type,
		               concentrate=conc,
		               train_conc=train_conc,
		               init_coef=coef,
		               is_biased=isbiased,
		              blockidx=blockidx)
		out_n_channel = out_states
		out_icnum = out_idcomp

	elif layer_name_str =='tofin':
		layer = ToFiniteProb(blockidx=blockidx)
		out_n_channel = 2
		out_icnum = in_n_channel
	elif layer_name_str == 'klconv_l':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		indpt_components = int(layer_opts['icnum']) if 'icnum' in layer_opts.keys() else 1
		stride = int(layer_opts['stride']) if 'stride' in layer_opts.keys() else 1
		coef = float(layer_opts['coef'])
		isbiased = ((layer_opts['bias']=='1')) if 'bias' in layer_opts.keys() else False
		isrelu = bool(int(layer_opts['isrelu']))
		drop_prob = float(layer_opts['droprate']) if 'droprate' in layer_opts.keys() else 0
		#stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		stoch = bool(layer_opts['stoch']=='1') if 'stoch' in layer_opts else False
		param = get_init(layer_opts['param'])
		layer = KLConv_LEGACY(fnum=fnum,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               inp_icnum= in_icnum,
		               icnum=indpt_components,
		               isbiased=isbiased,
		               isrelu=isrelu,
		               biasinit=None,
		               padding=pad,
		               paraminit=param,
		               isstoch=stoch,
		               coefinit=coef,
		               stride=stride,
		               drop_rate = drop_prob,
		               blockidx=blockidx)
		out_n_channel = fnum
		out_icnum= indpt_components



	elif layer_name_str == 'relu':
		layer = FReLU(blockidx=blockidx)
		out_n_channel = in_n_channel
	elif layer_name_str == 'hlnorm':
		alpha = int(layer_opts['alpha'] if 'alpha' in layer_opts.keys() else 1)
		layer = HyperLNorm(alpha,blockidx=blockidx)
		out_n_channel = in_n_channel
	elif layer_name_str == 'sigmoid':
		layer = nn.Sigmoid()
		out_n_channel = in_n_channel
	elif layer_name_str == 'idreg':
		alpha = float(layer_opts['alpha'])
		layer = IDReg(alpha, blockidx=blockidx)
		out_n_channel = in_n_channel
	elif layer_name_str == 'lsigmoid':
		layer = nn.LogSigmoid()
		out_n_channel = in_n_channel
	elif layer_name_str == 'maxpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'])
		pad = layer_opts['pad']
		pad = evalpad(pad,ksize)
		layer = nn.MaxPool2d(kernel_size=ksize, stride=stride, padding=pad)
		out_n_channel = in_n_channel
	elif layer_name_str == 'avgpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'])
		pad = layer_opts['pad']
		pad = evalpad(pad,ksize)
		layer = nn.AvgPool2d(kernel_size=ksize, stride=stride, padding=pad)
		out_n_channel = in_n_channel

	elif layer_name_str == 'klavgpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'])
		pad = layer_opts['pad']
		layer = KLAvgPool(ksize,stride, pad,blockidx= blockidx)
		out_n_channel = in_n_channel
		out_icnum= in_icnum
	elif layer_name_str =='dropout':
		prob = float(layer_opts['p'])
		exact = bool(layer_opts['exact']) if 'exact' in layer_opts.keys() else True
		layer = FDropOut(rate =prob,blockidx=blockidx,exact=exact)
		out_n_channel = in_n_channel
	elif layer_name_str =='bn':

		layer = nn.BatchNorm2d(in_n_channel,track_running_stats=True)
		out_n_channel = in_n_channel
	else:
		raise(Exception('Undefined Layer: ' + layer_name_str))
	if out_n_channel == -1:
		raise('Output Channel Num not assigned in :' + layer_name_str)

	blockidx_dict[layer_name_str] = blockidx_dict[layer_name_str]+1
	return layer, out_n_channel,out_icnum
