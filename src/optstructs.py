from definition import *
from torchvision.transforms import transforms
import torchvision as tv
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader

from src.datautils.synthetic import *
'''Model Imports'''
def print_section(symbol='*',printer=print):
	printer(symbol*100)
class opt(object):
	def __init__(self):
		pass

	def to_dict(self):
		raise Exception('Not implemented')
	def load_from_dict(self,dictionary:dict):
		raise Exception('Not implemented')
class allOpts(opt):
	def __init__(self,
	             name,
	             netopts=None,
	             optimizeropts=None,
	             epocheropts=None,
	             dataopts=None):
		super(allOpts,self).__init__()
		self.name = name
		self.netopts = netopts
		self.optimizeropts = optimizeropts
		self.epocheropts = epocheropts
		self.dataopts=dataopts
	def device(self):
		return self.device
	def validateopts(self):
		if not self.netopts.inputspatszvalidator(self.dataopts.inputspatsz):
			raise Exception('Input spatial size is not compatible with the model')
	def print(self, printer=print):
		printer('\n\n')
		print_section('*=*=',printer=printer)
		self.netopts.print(printer= printer)
		self.optimizeropts.print(printer=printer)


	def to_dict(self):
		out = dict(name=self.name,
		           netopts=self.netopts.to_dict(),
		           optimizeropts=self.optimizeropts.to_dict(),
		           datasetopts= self.dataopts.to_dict(),
		           epocheropts= self.epocheropts.to_dict()

		           )
	def load_from_dict(self,state_dict:dict):
		self.name = state_dict['name']
		self.netopts = NetOpts().load_from_dict(state_dict['netopts'])
		self.optimizeropts = OptimOpts().load_from_dict(state_dict['optimizeropts'])
		self.dataopts = DataOpts().load_from_dict(state_dict['datasetopts'])
		self.epocheropts = EpocherOpts().load_from_dict(state_dict['epocheropts'])


class EpocherOpts(opt):
	def __init__(self,
	             save_results,
	             epochnum=150,
	             batchsz=100,
	             batchsz_val=128,
	             shuffledata=True,
	             numworkers=1,
	             gpu=True):
		self.epochnum = epochnum
		self.batchsz = batchsz
		self.batchsz_val= batchsz_val
		self.shuffledata = shuffledata
		self.numworkers = numworkers
		self.gpu = gpu
		self.save_results=save_results
		if self.gpu:
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")


class NetOpts(opt):
	''' Weight and bias init has the form
		lambda x: x.zero_()'''
	def __init__(self,modelstring,
	             input_channelsize,
	             inputspatszvalidator,
	             data_transforms=[],
	             classicNet=False,
	             weightinit=lambda x: x,
				 biasinit=lambda x: x,
	             param_scale=1,
	             customdict=dict(exact=True)
				 ):
		self.inputspatszvalidator=inputspatszvalidator
		self.modelstring = modelstring
		self.classicNet=classicNet
		self.customdict = customdict
		self.parameter_scale= 1
		self.input_channelsize= input_channelsize
		if classicNet and (not chck_lambda(weightinit) or not chck_lambda(biasinit) or not chck_lambda(inputspatszvalidator)) :
			raise Exception('Weight/bias init and size validators must be lambda functions\n W/B inits must be called on weight.data or param.data')
		self.weightinit=weightinit
		self.biasinit = biasinit
		self.data_transforms = data_transforms
	def print(self,printer=print):
		print_section('-')
		printer("Model String:")
		printer(self.modelstring)
		print_section('-')
		printer("Custom Dictionary")
		for key in self.customdict.keys():
			printer(key, end=': ')
			printer(self.customdict[key],end=' | ')
			printer('\n')


class OptimOpts(opt):
	def __init__(self,lr=1,
	             lr_sched_lambda = None,
	             type='SGD',
	             momentum=0.9,
	             weight_decay=0,
	             dampening=0,
	             nestrov=False,
	             loss=None
				 ):
		self.lr = lr
		self.lr_sched = None
		self.lr_sched_lambda=lr_sched_lambda
		self.type = type
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.dampening = dampening
		self.nestrov = nestrov
		self.loss=loss
	def print(self,printer=print):
		printer("Optimization Options:")
		printer("Learning Rate: ", end='')
		printer(self.lr, end=' ')
		printer("|")
		printer("momentum: ",end='' )
		printer(self.momentum, end=' ')
		printer("|")
		printer("weight decay: ", end='')
		printer(self.weight_decay,end=' ')
		printer("|")

class DataOpts(opt):
	def __init__(self,name,inputdim=0,outputdim=0,samplenum=0
				 ):
		self.datasetname=name
		self.trainset = None
		self.testset = None
		if name == 'cifar10':
			inputspatsz=32
			channelsize=3
			inputrange=(0,1)
			classnum=10

		elif name == 'cifar100':
			inputspatsz=32
			channelsize=3
			inputrange=(0,1)
			classnum=100
		elif name == 'synthetic':
			self.joint = Joint(inputdim,outputdim)
			inputspatsz = 1
			channelsize = inputdim
			inputrange = (0,1)
			self.samplenum = samplenum
			classnum = outputdim
		else:
			raise Exception(name + ':Dataset options are not defined')
		self.inputspatsz = inputspatsz
		self.channelsize = channelsize
		self.inputrange = inputrange
		self.classnum = classnum
	def get_loaders(self,opts:allOpts):
		if self.datasetname == 'cifar10':
			return self.get_cifar10(opts)
		elif self.datasetname == 'cifar100':
			return self.get_cifar100(opts)
		elif self.datasetname == 'synthetic':
			return self.get_synth(opts)
		else:
			Exception("Dataset Not Found")

	def get_synth(self,opts:allOpts):

		self.trainset = self.joint.create_dataset(self.samplenum[0])
		self.testset = self.joint.create_dataset(self.samplenum[1])
		batchsz = opts.epocheropts.batchsz
		batchszval = opts.epocheropts.batchsz_val
		isshuffle = opts.epocheropts.shuffledata
		transform = transforms.Compose(
			[transforms.ToTensor()] + opts.netopts.data_transforms)
		# Construct loaders
		train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                           num_workers=1)
		test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batchszval, shuffle=isshuffle, sampler=None,
		                                          num_workers=1)
		return train_loader, test_loader
	def get_cifar10(self,opts: allOpts):
		# Obtain options from opts class
		batchsz = opts.epocheropts.batchsz
		isshuffle = opts.epocheropts.shuffledata
		transform_train = transforms.Compose(
			( opts.netopts.data_transforms + [transforms.ToTensor()]))
		transform_test = transforms.Compose(
			( [transforms.ToTensor()]))
		# Construct loaders
		trainset = tv.datasets.CIFAR10(PATH_DATA, train=True, download=True, transform=transform_train)
		testset = tv.datasets.CIFAR10(PATH_DATA, train=False, download=True, transform=transform_test)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                       num_workers=1)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                      num_workers=1)
		return train_loader, test_loader

	def get_cifar100(self, opts: allOpts):
		# Obtain options from opts class
		batchsz = opts.epocheropts.batchsz
		isshuffle = opts.epocheropts.shuffledata
		transform_train = transforms.Compose(
			(opts.netopts.data_transforms + [transforms.ToTensor()]))
		transform_test = transforms.Compose(
			([transforms.ToTensor()]))
		# Construct loaders
		trainset = tv.datasets.CIFAR100(PATH_DATA, train=True, download=True, transform=transform_train)
		testset = tv.datasets.CIFAR100(PATH_DATA, train=False, download=True, transform=transform_test)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                       num_workers=1)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
		                                      num_workers=1)
		return train_loader, test_loader
def chck_lambda(l):
	Lambda = lambda:0
	if isinstance(Lambda,type(l)):
		return True
	else:
		return False
class dummyclass(object):
	def __init__(self):
		return

