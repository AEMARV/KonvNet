from torch.nn import Module
from torch.nn import Parameter
from src.layers.klfunctions import *
from src.layers.Initializers import *
import torch.autograd.gradcheck
from torchvision.utils import save_image
import math
from definition import concentration as C
from definition import *
from torch.distributions.gamma import Gamma as Gamma


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
		probkernel = y.exp()
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
		self.weight = Parameter(torch.randn(fnum,input_ch,spsize,spsize)*init_coef)
		self.weight.requires_grad=True
		self.register_parameter('weight',self.weight)

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
		y = F.conv2d(input,self.weight,self.bias,stride=self.stride,padding=self.pad)
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
	             inp_icnum,
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
		self.fnum = fnum
		self.chansz = inp_chan_sz
		self.coefinit  = coefinit
		self.kernel_shape = (fnum*self.icnum,)+(inp_chan_sz*inp_icnum,)+(kersize,kersize)
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
		self.register_parameter('kernel', self.kernel)
		if self.isbiased:
			self.paraminit.coef= 0
			self.bias = Parameter(data=self.paraminit((1,self.fnum,1,1,self.icnum),isbias=True))
			self.register_parameter('bias', self.bias)

	'''Kernel/Bias Getters'''
	def get_log_kernel(self,index=0):
		k = self.kernel
		sp1 = k.shape[2]
		sp2 = k.shape[3]
		k = k.reshape((self.fnum*self.icnum,self.chansz,self.inp_icnum,k.shape[2],k.shape[3]))
		k = self.paraminit.get_log_kernel(k)
		return k
	def get_log_kernel_conv(self,):
		k = self.get_log_kernel()
		k = k.reshape((self.fnum * self.icnum, self.inp_icnum * self.chansz, self.kernel.shape[2], self.kernel.shape[3]))
		return k

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
			syms = self.kernel_shape[1]*math.log(2)
		else:
			syms = math.log(self.kernel_shape[1])
		return syms

class KLConv_LEGACY(KLConv_Base):
	def __init__(self,*args,
	             drop_rate=1,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(KLConv_LEGACY,self).__init__(*args,**kwargs)
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.drop_rate = drop_rate
		self.conc_par= Parameter(data= torch.zeros(1))
		self.register_parameter('conc',self.conc_par)
		self.build()


	''' Ent Functions'''
	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''Conv Functions'''

	def kl_xl_kp(self,x:torch.Tensor,mask=1):
		''' Relu KL Conv '''
		ent=0
		lkernel = self.get_log_kernel_conv()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y,ent = self.add_ker_ent(y, x, pkernel, lkernel,mask=mask)
		return y,ent

	def kl_xp_kl(self,xl):
		#TODO : not implemented yet
		xp = xl.exp()
		ent = -xp * xl #type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp,self.get_log_kernel_conv())
		ent = self.convwrap(ent,self.get_log_kernel_conv().exp()[0:1,0:1,0:,0:]*0 + 1)
		y = cross + ent
		ent = ent.sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False)
		return y,ent




	def forward(self, x:torch.Tensor,MAP=False):
		#dummy = x[0:,0:1,0:,0:,0:]*0 +1
		#dummy = F.dropout(dummy,0.5,self.training)
		#x = x * dummy
		x= self.reshape_input_for_conv(x)
		#self.project_params()
		if not self.isrelu:
			# Sigmoid Activated

			y,g = self.kl_xp_kl(x)
				#self.logprob = self.ent_kernel()
		else:
			# ReLU Activated

			y,ent = self.kl_xl_kp(x)

		y = self.reshape_input_for_nxt_layer(y)
		#y = y*math.log(self.fnum)
		#y = y *((self.conc_par.exp()).exp())
		if self.isbiased:
			pass
			y = y+  self.get_log_bias()

		#y = y + self.get_log_bias()
		y = y.log_softmax(dim=1)
		return y


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
	             train_conc=False,
	             cross=False,
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
		self.train_conc = train_conc
		self.cross= cross,
		filter_size =(self.out_state_num*self.out_id_comp_num,  #0
		              self.in_state_num,  #1
		              self.in_id_comp_num,  # 2
		              self.rsize,  #3
		              self.rsize)           #4


		''' Note: The initialization with log needs to be finite'''
		if self.init_type=='log':
			weight = -(torch.rand(filter_size)+epsilon).exponential_()*self.init_coef
			bias = torch.zeros(1, self.out_state_num, 1, 1, out_id_comp_num)
		elif self.init_type=='square':
			weight = -(torch.rand(filter_size)).exponential_()
			weight = weight / (weight.sum(dim=1, keepdim=True))
			weight = weight.clamp(epsilon, 1)** (self.init_coef/2)
			weight = weight / ((weight**2).sum(dim=1, keepdim=True)).sqrt()
			bias = torch.ones(1, self.out_state_num, 1, 1, out_id_comp_num)
		elif self.init_type=='logd':
			weight = -(torch.rand(filter_size)).exponential_()
			weight = weight/(weight.sum(dim=1,keepdim=True))
			weight = weight.clamp(epsilon,1)
			weight = (weight.log() * self.init_coef).log_softmax(dim=1)
			# weight= weight.log().log_softmax(dim=1) * self.init_coef
			bias = torch.zeros(1, self.out_state_num, 1, 1, out_id_comp_num)
		else:
			raise Exception(self.init_type + "Not Implemeted")

		self.weight = Parameter(data=weight,requires_grad=True)
		self.bias = Parameter(data=bias.detach(),requires_grad=self.is_biased)
		self.concentrate= Parameter(data=torch.ones(1)*self.concentrate,requires_grad=train_conc)

		self.register_parameter('weight',self.weight)
		self.register_parameter('bias',self.bias)
		self.register_parameter('concentrate', self.concentrate)

		self.weight.requires_grad =True
		self.bias.requires_grad= self.is_biased
		self.concentrate.requires_grad= self.train_conc



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
		ent_map = F.conv2d(input_mask*0+1,weight_ent,stride=self.stride,padding=self.pad)
		return ent_map

	def log_normalize(self,weight,dim=1):
		if self.init_type=='log' or  self.init_type=='logd':
			# weight = weight.log_softmax(dim=0)
			return weight.log_softmax(dim=dim)
		elif self.init_type=='square':
			logprob = 2*(weight.abs().clamp(epsilon,None).log())
			# weight = weight.log_softmax(dim=0)
			logprob= logprob.log_softmax(dim=dim)
			return logprob
		else:
			Exception(self.init_type + " Not implemented")
	def normalize(self,weight,dim=1):
		if self.init_type=='log' or  self.init_type=='logd':
			return weight.log_softmax(dim=dim).exp()
		elif self.init_type=='square':
			prob = weight**2
			prob= prob / prob.sum(dim=dim , keepdim=True)
			return prob
		else:
			Exception(self.init_type + " Not implemented")

	def entropy_map(self, input):
		inputexp = input.exp()
		entmap = -inputexp * input
		entmap = entmap.sum(dim=1)
		entmap[entmap != entmap] = 0
		entmap = entmap.permute([0, 3, 1, 2])
		weight = self.normalize(self.weight)
		weight = weight.sum(dim=1).detach()
		entropy = F.conv2d(entmap, weight, stride=self.stride, padding=self.pad)
		return entropy

	def kl_xl_kp(self, input_orig):
		input = input_orig.permute([0, 1, 4, 2, 3])
		input = input.reshape(input.shape[0], -1, input.shape[3], input.shape[4])
		weight = self.log_normalize(self.weight, dim=1)
		weight = weight.reshape(weight.shape[0], -1, weight.shape[3], weight.shape[4])
		exp_weight = weight.exp()


		kld = F.conv2d(input, exp_weight, stride=self.stride, padding=self.pad, bias=None)
		if not self.cross:
			entropy_filt = self.entropy_filt_map(input_orig)
			kld = kld + entropy_filt
		kld = kld * self.concentrate
		kld = kld.reshape(kld.shape[0], self.out_state_num, self.out_id_comp_num, kld.shape[2], kld.shape[3])
		kld = kld.permute([0, 1, 3, 4, 2])
		return kld
	def kl_xp_kl(self, input_orig):
		input = input_orig.permute([0, 1, 4, 2, 3])
		input = input.reshape(input.shape[0], -1, input.shape[3], input.shape[4])
		inputexp = input.exp()
		weight = self.log_normalize(self.weight)

		weight = weight.reshape(weight.shape[0], -1, weight.shape[3], weight.shape[4])
		cross_entropy = F.conv2d(inputexp, weight, stride=self.stride, padding=self.pad)
		entmap = self.entropy_map(input_orig)
		kld = cross_entropy + entmap
		kld = kld * self.concentrate
		kld = kld.reshape(kld.shape[0], self.out_state_num, self.out_id_comp_num, kld.shape[2],
		                  kld.shape[3])
		kld = kld.permute([0, 1, 3, 4, 2])
		return kld
	def forward(self, input_orig):
		''' Calculates the Konv^r of input and output
		Input: needs to be in the log domain and normalized. The size of input is
		(_, #state, height,width, #independent components).
		Indpenent componets are per pixel.

		output: output is in the log domain of size (_ , #out_state, height, width, #out_id_comp)
		and is log normalized across the first dimension
		'''
		kld1 = self.kl_xl_kp(input_orig)
		# kld2 = self.kl_xp_kl(input_orig)
		bias = self.log_normalize(self.bias)
		lprob = kld1 + bias
		# y = lprob - logsumexpstoch(lprob*self.concentrate,1)/self.concentrate
		y = (lprob).log_softmax(dim=1)
		# output = output - logsumexpstoch(output,1)
		return y
	





class Konv_S(Konv_R):
	def __init__(self,*args,**kwargs):
		super(Konv_S,self).__init__(*args,**kwargs)

		filter_size = (self.out_state_num * self.out_id_comp_num,  # 0
		               self.in_state_num,  # 1
		               self.in_id_comp_num,  # 2
		               self.rsize,  # 3
		               self.rsize)  # 4

	def entropy_map(self,input):
		inputexp = input.exp()
		entmap = -inputexp*input
		entmap = entmap.sum(dim=1)
		entmap[entmap!=entmap]= 0
		entmap = entmap.permute([0,3,1,2])
		weight = self.normalize(self.weight)
		weight = weight.sum(dim=1).detach()
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
		kld = kld *self.concentrate
		kld = kld.reshape(kld.shape[0], self.out_state_num, self.out_id_comp_num, kld.shape[2],
		                        kld.shape[3])
		kld = kld.permute([0, 1, 3, 4, 2])
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

		out = out.clamp(epsilon,None)
		out = out.log()
		# out = out.log_softmax(dim=1)
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

	def forward(self, inputs, isprobability=True):
		if isprobability:
			inputs = inputs.unsqueeze(4).permute([0, 4, 2, 3, 1])
			output = (torch.cat((inputs,1-inputs),dim=1)+epsilon).log()
		else:
			inputs = inputs.unsqueeze(4).permute([0,4,2,3,1])
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
