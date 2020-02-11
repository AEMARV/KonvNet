from torch.nn import Module
from torch.nn import Parameter
from src.layers.klfunctions import *
from src.layers.Initializers import *
import torch.autograd.gradcheck
from torchvision.utils import save_image
import math
from definition import concentration as C
from definition import *


class MyModule(Module):
	def __init__(self, *args, blockidx=None, **kwargs):
		super(MyModule,self).__init__(*args, **kwargs)
		self.logprob = torch.zeros(1).to('cuda:0')
		self.regularizer = torch.zeros(1).to('cuda:0')
		self.scalar_dict = {}
		if blockidx is None:
			raise Exception('blockidx is None')
		self.blockidx = blockidx
		self.compact_name = type(self).__name__ + '({})'.format(blockidx)
		self.register_forward_hook(self.update_scalar_dict)

	def update_scalar_dict(self,self2,input,output):
		return
	def get_output_prior(self,inputprior):
		return inputprior
	def get_lrob_model(self, inputprior):
		return 0
	def get_log_prob(self):
		lprob = self.logprob
		for m in self.children():
			if isinstance(m,MyModule):
				lprob = lprob + m.get_log_prob()
		self.logprob = 0
		return lprob

	def get_reg_vals(self):
		reg = self.regularizer
		for m in self.children():
			if isinstance(m, MyModule):
				temp = m.get_reg_vals()
				if temp is not None:
					reg = reg + temp
		self.regularizer = 0
		return reg

	def print(self,inputs,epoch_num,batch_num):
		for m in self.children():


			inputs = m.forward(inputs)
			m.print_output(inputs, epoch_num, batch_num)

	def get_scalar_dict(self):

		for m in self.children():
			if isinstance(m, MyModule):
				self.scalar_dict.update(m.get_scalar_dict())

		return self.scalar_dict

	def max_prob(self,mother,model,dim):
		lrate = (mother - model)
		max_lrate= lrate.min(dim=dim , keepdim=True)[0]
		max_logical_ind = (lrate == max_lrate).float()

		max_logical_ind = max_logical_ind / max_logical_ind.sum(dim=dim,keepdim=True)
		max_lprob = (max_logical_ind * lrate).sum(dim=dim,keepdim=True)
		return max_lprob

	def print_filts(self,epoch,batch):
		try:
			probkernel = self.weight
		except:
			return
		sh = probkernel.shape
		probbias = self.get_log_bias().exp().view(sh[0], 1, 1, 1)
		probkernel = probkernel * probbias
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Filters' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/filt_' + str(epoch)+'_'+str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=False, scale_each=False, nrow=chans)

	def print_output(self, y,epoch,batch):
		probkernel = y
		probkernel= probkernel[0:,0:-1]
		chans = probkernel.shape[1]
		factors =probkernel.shape[4]
		probkernel = probkernel.permute((0,1,4,2,3))
		probkernel = probkernel.contiguous().view(
			[probkernel.shape[0] * probkernel.shape[1]*probkernel.shape[2], 1, probkernel.shape[3], probkernel.shape[4]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Output' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/output_' + str(epoch) + '_' + str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=False, scale_each=False, nrow=chans*factors)

	def prop_prior(self,output_prior):
		return output_prior

	def forward(self, *inputs):
		raise Exception("Not Implemented")
		pass

''' Deterministic Components'''

class FConv(MyModule):
	def __init__(self,*args,fnum=0,input_ch=0,spsize=0,stride=0,pad=0,init_coef= 1,isbiased= True, **kwargs):
		super(FConv,self).__init__(*args,**kwargs)
		self.fnum = fnum
		self.spsize= spsize
		self.input_ch = input_ch
		self.kernel = Parameter(torch.randn(fnum,input_ch,spsize,spsize)*init_coef)
		self.kernel.requires_grad=True
		self.register_parameter('weight',self.kernel)

		self.bias = Parameter(torch.zeros(fnum))
		self.bias.requires_grad = isbiased
		self.register_parameter('bias',self.bias)
		self.stride= stride
		self.pad = num_pad_from_symb_pad(pad,spsize)
	def print_output(self, y,epoch,batch):
		probkernel = y
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Output' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/output_' + str(epoch) + '_' + str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=probkernel.shape[1])

		return
	def forward(self, input):
		y = F.conv2d(input,self.kernel,self.bias,stride=self.stride,padding=self.pad)
		return y

class FReLU(MyModule):
	def forward(self, inputs:Tensor):
		return inputs.relu()

class HyperLNorm(MyModule):
	def __init__(self,alpha,*args,**kwargs):
		super(HyperLNorm,self).__init__(*args,**kwargs)
		self.alpha = alpha
	def forward(self, inputs:Tensor):
		return alpha_lnorm(inputs,1,self.alpha).exp()

'''Stochastic Component'''
class KLConv_Base(MyModule):
	def __init__(self,
	             *args,
	             fnum=None,
	             icnum=1,
	             inp_icnum=None,
	             kersize=None,
	             isbiased=False,
	             inp_chan_sz=0,
	             isrelu=True,
	             biasinit=None,
	             padding='same',
	             stride=1,
	             paraminit= None,
	             coefinit = 1,
	             isstoch=False,
	             **kwargs
	             ):
		super(KLConv_Base,self).__init__(*args,**kwargs)
		#TODO: Set isbinary switch in paraminit
		self.biasinit = biasinit
		self.padding = num_pad_from_symb_pad(padding,kersize)
		self.isbiased = isbiased
		self.isrelu = isrelu
		self.stride = stride
		self.isstoch= isstoch
		self.axisdim=-2
		self.icnum=icnum
		self.inp_icnum = inp_icnum
		self.spsz = kersize
		self.fnum = fnum
		self.chansz = inp_chan_sz
		self.coefinit  = coefinit
		self.kernel_shape = (fnum,self.icnum,)+(inp_chan_sz,inp_icnum,)+(kersize,kersize)
		self.paraminit = paraminit# type:Parameterizer
		self.input_is_binary= False
	'''Scalar Measurements'''
	def get_scalar_dict(self):
		#y = self.scalar_dict.copy()
		#self.scalar_dict = {}
		return self.scalar_dict

	def update_scalar_dict(self,self2,input,output):
		#Filter Entorpy
		if type(input) is tuple:
			input = input[0]

		temp_dict = {self.compact_name +'| Kernel Entropy' : self.expctd_ent_kernel().item(),
		             self.compact_name +'| Input Entropy' : self.expctd_ent_input(input).item(),
		             }
		for key in temp_dict.keys():
			if key in self.scalar_dict:
				self.scalar_dict[key] = self.scalar_dict[key]/2 + temp_dict[key]/2
			else:
				self.scalar_dict[key] = temp_dict[key]

	''' Build'''
	def build(self):
		self.paraminit.coef =  self.coefinit
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
		self.register_parameter('weight', self.kernel)
		if self.isbiased:
			self.paraminit.coef= 0
			self.bias = Parameter(data=self.paraminit((1,self.fnum,1,1,self.icnum),isbias=True))
			self.register_parameter('bias', self.bias)

	'''Kernel/Bias Getters'''
	def get_log_kernel(self,kernel=None,index=0):
		if kernel is None:
			k = self.kernel
		sp1 = k.shape[4]
		sp2 = k.shape[5]
		k = k.reshape((self.fnum*self.icnum,self.chansz,self.inp_icnum,sp1,sp2))
		k = self.paraminit.get_log_kernel(k)
		return k

	def get_log_kernel_conv(self,kernel=None):
		k = self.get_log_kernel(kernel=kernel)
		k = k.reshape((self.fnum * self.icnum, self.inp_icnum * self.chansz, self.kernel.shape[2], self.kernel.shape[3]))
		return k

	def get_kernel(self):
		return self.kernel

	def get_prob_kernel(self)-> torch.Tensor:

		return self.get_log_kernel().exp()

	def get_log_bias(self,index=0):
		return self.paraminit.get_log_kernel(self.bias)

	def get_prob_bias(self)-> torch.Tensor:
		return self.biasinit.get_prob_bias(self.bias)

	def convwrap(self,x:Tensor,w:Parameter):
		y = F.conv2d(x, w, bias=None,
		             stride=self.stride,
		             padding=self.padding)
		return y

	def reshape_input_for_conv(self,x:Tensor):
		if x.ndimension()<5:
			return x

		x=(x.permute(0,1,4,2,3))
		x = x.reshape((x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]))
		return x

	def reshape_input_for_nxt_layer(self,ly):
		ly = ly.view((ly.shape[0],self.fnum,self.icnum,ly.shape[2],ly.shape[3]))
		ly = ly.permute(0,1,3,4,2)
		return ly

	def add_ker_ent(self,y:torch.Tensor,x,pker,lker,mask=None):
		H = self.ent_per_spat(pker,lker)

		if mask is not None:
			H = self.convwrap((x[0:, 0:1, 0:, 0:]*0 + 1)*mask, H)
		else:
			H = self.convwrap(x[0:,0:1,0:,0:]*0 +1,H)
		hall = H.mean()
		return y + H,hall

	def project_params(self):
		self.kernel = self.paraminit.projectKernel(self.kernel)

	def get_log_prob(self):
		lprob = self.logprob
		self.logprob= 0
		return lprob

	'''Entorpy Functions'''
	def ent_per_spat(self,pker,lker):
		raise Exception(self.__class__.__name__ + ":Implement this function")

	def ent_kernel(self):
		return self.ent_per_spat(self.get_prob_kernel(),self.get_log_kernel()).sum()

	def expctd_ent_kernel(self):
		return self.ent_per_spat(self.get_log_kernel().exp(),self.get_log_kernel()).mean()/self.get_log_symbols()

	def ent_input_per_spat(self,x):
		ent = -x * x.exp()
		ent = ent.sum(dim=1, keepdim=True)
		return ent

	def expctd_ent_input(self,x):
		ent = self.ent_input_per_spat(x)
		return ent.mean()/self.get_log_symbols()

	def ent_input(self,x):
		ent = self.ent_input_per_spat(x)#type:Tensor
		e = ent.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return e

	def get_log_symbols(self):
		if self.input_is_binary:
			syms = self.chansz*math.log(2)
		else:
			syms = math.log(self.chansz)
		return syms


class Konv_R(MyModule):
	def __init__(self,out_state_num,
	             out_id_comp_num,
	             rsize,
	             in_state_num,
	             in_id_comp_num,
	             *args,
	             concentrate=1,
	             pad='same',
	             stride=1,
	             init_type='log',
	             init_coef=1,
	             is_biased=True,
	             **kwargs,
	             ):
		super(Konv_R,self).__init__(*args,**kwargs)
		self.out_state_num = out_state_num
		self.out_id_comp_num = out_id_comp_num
		self.rsize= rsize
		self.in_state_num= in_state_num
		self.in_id_comp_num= in_id_comp_num
		self.concentrate = concentrate
		self.pad = num_pad_from_symb_pad(pad,rsize)
		self.stride =stride
		self.init_type= init_type
		self.init_coef = init_coef
		self.is_biased = is_biased
		filter_size =(self.out_state_num*self.out_id_comp_num,  #0
		              self.in_state_num,  #1
		              self.in_id_comp_num,  # 2
		              self.rsize,  #3
		              self.rsize)           #4

		if self.init_type=='log':
			weight = torch.rand(filter_size).log()*self.init_coef
		elif self.init_type=='square':
			weight = torch.randn(filter_size) * self.init_coef
		else:
			Exception(self.init_type + "Not Implemetedn")
		self.weight = Parameter(weight)

		self.bias = Parameter(torch.zeros(1,self.out_state_num,1,1,out_id_comp_num))

		self.register_parameter('weight',self.weight)
		self.register_parameter('bias',self.bias)

		self.weight.requires_grad =True
		self.bias.requires_grad= self.is_biased




		return
	def entropy_filt_map(self,input_orig):
		''' Since in the padded area the information is not present, the entropy of the filter components is not added.
			. Also this function can be used in the future usage of DropOut of independant components in input tensor.
			Where entropy of filters is not added to the dropped components.
		'''
		input_orig = input_orig.exp().sum(dim=1)
		input_mask = input_orig.permute([0,3,1,2]).detach()
		weight = self.log_normalize(self.weight)
		weight_exp = weight.exp()
		weight_ent = -weight_exp*weight
		weight_ent = weight_ent.sum(dim=1)
		weight_ent[weight_ent !=weight_ent]=0

		ent_map = F.conv2d(input_mask,weight_ent,stride=self.stride,padding=self.pad)
		return ent_map

	def log_normalize(self,weight,dim=1):
		if self.init_type=='log':
			return weight.log_softmax(dim=1)
		elif self.init_type=='square':
			logprob = 2*self.weight.abs().log()
			logprob= logprob.log_softmax(dim=dim)
			return logprob
		else:
			Exception(self.init_type + " Not implemented")
	def normalize(self,weight,dim=1):
		if self.init_type=='log':
			return weight.log_softmax(dim=1).exp()
		elif self.init_type=='square':
			prob = weight**2
			prob= prob / prob.sum(dim=1 , keepdim=1)
			return prob
		else:
			Exception(self.init_type + " Not implemented")
	def forward(self, input_orig,input_mask=1):
		''' Calculates the Konv^r of input and output
		Input: needs to be in the log domain and normalized. The size of input is
		(_, #state, height,width, #independent components).
		Indpenent componets are per pixel.

		output: output is in the log domain of size (_ , #out_state, height, width, #out_id_comp)
		and is log normalized across the first dimension
		'''
		input = input_orig.permute([0,1,4,2,3])
		input = input.reshape(input.shape[0],-1,input.shape[3],input.shape[4])

		weight = self.log_normalize(self.weight,dim=1)
		weight= weight.reshape(weight.shape[0],-1,weight.shape[3],weight.shape[4])
		exp_weight = weight.exp()

		bias = self.log_normalize(self.bias,dim=1)

		output = F.conv2d(input,exp_weight,stride=self.stride,padding=self.pad)
		entropy_filt = self.entropy_filt_map(input_orig)
		output = output + entropy_filt
		output = output.reshape(output.shape[0],self.out_state_num,self.out_id_comp_num,output.shape[2],output.shape[3])
		output = output.permute([0,1,3,4,2])
		output = output + bias
		output = (output*self.concentrate).log_softmax(dim=1)

		return output
	
	
class Konv_S(Konv_R):
	def __init__(self,*args,**kwargs):
		super(Konv_S,self).__init__(*args,**kwargs)

		filter_size = (self.fnum * self.out_id_comp_num,  # 0
		               self.in_state_num,  # 1
		               self.in_id_comp_num,  # 2
		               self.rsize,  # 3
		               self.rsize)  # 4

	def entropy_map(self,input):
		inputexp = input.exp()
		entmap = -inputexp*input
		entmap = entmap.sum(dim=1)
		entmap[entmap!=entmap]= 0
		entmap = entmap.permute([0,4,2,3])
		weight = self.normalize(self.weight)
		weight = weight.sum(dim=1)
		entropy = F.conv2d(entmap, weight, stride=self.stride, padding=self.pad)
		return entropy
	
	def forward(self,input_orig, mask=None):
		input = input_orig.permute([0,1,4,2,3])
		input = input.reshape(input.shape[0],-1,input.shape[3],input.shape[4])
		inputexp = input.exp()
		weight = self.log_normalize(self.weight)
		bias = self.log_normalize(self.bias, dim=1)
		
		weight= weight.reshape(weight.shape[0],-1,weight.shape[3],weight.shape[4])
		cross_entropy = F.conv2d(inputexp, weight, stride=self.stride, padding=self.pad)
		entmap = self.entropy_map(input_orig)
		kld = cross_entropy + entmap
		kld = kld+bias
		
		y = kld.log_softmax(dim=1)
		return y
		

		

class KLAvgPool(MyModule):
	def __init__(self,spsize,stride,pad,isstoch=True, **kwargs):
		super(KLAvgPool,self).__init__(**kwargs)
		self.spsize= spsize
		self.stride = stride
		self.pad = num_pad_from_symb_pad(pad,spsize)
		self.isstoch = isstoch
	def print_output(self, y,epoch,batch):
		return
	def forward(self, x:Tensor,isinput=None,isuniform=False):

		chans = x.shape[1]
		icnum = x.shape[4]
		einput = x.exp()
		# einput = x.log_softmax(dim=1)
		einput = einput.permute(0,1,4,2,3)
		einput = einput.reshape((einput.shape[0],einput.shape[1]*einput.shape[2],einput.shape[3],einput.shape[4]))

		out = F.avg_pool2d(einput,
		                   self.spsize,
		                   stride=self.stride,
		                   padding=self.pad,
		                   count_include_pad=False)
		out = out.reshape((out.shape[0],chans,icnum,out.shape[2],out.shape[3]))
		out = out.permute(0,1,3,4,2)
		# out = out.log_softmax(dim=1)
		out = out.clamp(epsilon,None)
		out = out.log()
		return out
	def generate(self,y:Tensor):
		#y = LogSumExpStoch.sample(y,1)
		y = y.exp()
		x = F.upsample(y,scale_factor=self.stride,mode='bilinear')
		x = x.log()
		return x
	def print_filts(self,epoch,batch):
		pass

class FDropOut(MyModule):
	def __init__(self,rate,*args,exact=False,**kwargs):
		super(FDropOut,self).__init__(*args,**kwargs)
		self.rate= rate
		self.exact = exact
	def forward(self, inputs,force_drop=True):
		outputs =F.dropout(inputs,p=self.rate,training=force_drop)
		return outputs


''' Inputs'''
class ToFiniteProb(MyModule):
	def __init__(self, *args, **kwargs):
		super(ToFiniteProb,self).__init__(*args,**kwargs)

	def forward(self, inputs):
		inputs = inputs.unsqueeze(4).permute([0,4,2,3,1])*100
		output = alpha_lnorm(torch.cat((inputs/2,-inputs/2),dim=1),1,1)
		return output

class IDReg(MyModule):
	def __init__(self,alpha,*args,**kwargs):
		super(IDReg,self).__init__(*args,**kwargs)
		self.alpha = alpha
	def forward(self, inputs:Tensor):
		out = idreg(inputs, 1, math.log(2))
		out = out.relu()
		return out

""" Samplers"""






def num_pad_from_symb_pad(pad:str,ksize:int)->Tuple[int]:
	if pad=='same':
		rsize = ksize
		csize = ksize
		padr = (rsize-1)/2
		padc = (csize-1)/2
		return (int(padr),int(padc))
	elif pad=='valid':
		padr=0
		padc=0
		return (int(padr),int(padc))
	elif type(pad) is tuple:
		return pad
	else:
		raise(Exception('Padding is unknown--Pad:',pad))
