from src.netparsers.parseutils import *
import torch.tensor
import math
import torch.nn.functional as F





class StaticNet(MyModule):
	''' A static Module constructed from a model string. The model string specs are parsed with the static functions
	in the class.
	Constructor:

	StaticNet(modelstring, opts)

	modelstring: the model specification string with delimiter '->'
	opts: opts struct.
	'''
	def __init__(self,modelstring,inputchannels,weightinit=None,biasinit=None,sample_data=None):
		super(StaticNet, self).__init__(blockidx=0)
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.layerlist = self.parse_model_string(modelstring,inputchannels['chansz'],inputchannels['icnum'])
		for bloacknum,layer in enumerate(self.layerlist):
			if isinstance(layer,nn.Conv2d):
				weightinit(layer.weight.data)
				biasinit(layer.bias.data)

			self.add_module('block'+str(bloacknum),layer)


	def forward(self, x:Tensor,mode='likelihood',usemin=False,concentration=1.0,drop=True):
		# Max pooling over a (2, 2) window
		logprob= torch.zeros(1).type_as(x)
		maxlrob= None
		first_sampler = False
		for i,layer in enumerate(self.layerlist):

			x= layer(x)
		return (x,1)


	''' String Parsers'''
	def parse_model_string(self, modelstring:str, in_n_channel,in_icnum):
		layer_list_string = modelstring.split('->')
		layer_list = []
		out_n_channel = in_n_channel
		blockidx_dict= {}
		for blocknum, layer_string in enumerate(layer_list_string, 0):
			layer,out_n_channel,in_icnum= parse_layer_string(layer_string,out_n_channel,in_icnum,blockidx_dict)
			if layer is not None:
				layer_list += [layer]
		return layer_list




