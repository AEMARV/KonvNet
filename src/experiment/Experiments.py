from src.experiment.Experiment import Experiment_
from src.models.klmodels import *
from src.datautils.datasetutils import *
import torch
import torchvision
''' Konv Experiments'''
class KNN_VGG_CIFAR10(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar10')
		model = self.KVGG_CIFAR10
		for coef in [4,5,6,7,8,9]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def KVGG_CIFAR10(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*128) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*128) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:0,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class KNN_VGG_CIFAR100(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar100')
		model = self.KVGG_CIFAR100
		for coef in [4,5,6,7,8,9]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def KVGG_CIFAR100(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*128) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*128) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:0,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class KNN_NIN_CIFAR10(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar10')
		model = self.KNIN_CIFAR10
		for coef in [3,4,5,6,7,8]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def KNIN_CIFAR10(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:6,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*160) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale * 96)) + d
		model_string += 'klavgpool|r:3,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192)) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d
		model_string += 'klavgpool|r:7,pad:valid,stride:2' + d


		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class KNN_NIN_CIFAR100(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar100')
		model = self.KNIN_CIFAR100
		for coef in [3,4,5,6,7,8]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def KNIN_CIFAR100(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:6,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*160) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale * 96)) + d
		model_string += 'klavgpool|r:3,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192)) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d
		model_string += 'klavgpool|r:7,pad:valid,stride:2' + d


		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class KNN_QuickCIFAR_CIFAR10(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar10')
		model = self.QC_CIFAR10
		for coef in [1,2,3,4]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def QC_CIFAR10(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:16'.format(ceil(scale*2) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:32'.format(ceil(scale*2) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:4,f:{},pad:same,stride:1,bias:1,icnum:32'.format(ceil(scale*2) ) + d
		model_string += konvrfixed+'r:7,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = discrete_exp_decay_lr(init_lr=1,step=20,exp_decay_perstep=2)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class KNN_QuickCIFAR_CIFAR10(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar10')
		model = self.QC_CIFAR10
		for coef in [1,2,3,4]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def QC_CIFAR10(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:16'.format(ceil(scale*2) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:32'.format(ceil(scale*2) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:4,f:{},pad:same,stride:1,bias:1,icnum:32'.format(ceil(scale*2) ) + d
		model_string += konvrfixed+'r:7,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = discrete_exp_decay_lr(init_lr=1,step=20,exp_decay_perstep=2)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


''' Baselines '''




class VGG_CIFAR10(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar10')
		model = self.KVGG_CIFAR10
		for coef in [4,5,6,7,8,9]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def KVGG_CIFAR10(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*128) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*128) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:0,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class VGG_CIFAR100(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar100')
		model = self.KVGG_CIFAR100
		for coef in [4,5,6,7,8,9]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def KVGG_CIFAR100(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*128) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*128) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*256) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:3,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*512) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:0,icnum:1'.format(ceil(scale*512) ) + d
		model_string += konvrfixed+'r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class NIN_CIFAR10(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar10')
		model = self.KNIN_CIFAR10
		for coef in [3,4,5,6,7,8]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def KNIN_CIFAR10(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:6,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*160) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale * 96)) + d
		model_string += 'klavgpool|r:3,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192)) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d
		model_string += 'klavgpool|r:7,pad:valid,stride:2' + d


		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class NIN_Konv_CIFAR100(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar100')
		model = self.KNIN_CIFAR100
		for coef in [3,4,5,6,7,8]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def KNIN_CIFAR100(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:6,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*160) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale * 96)) + d
		model_string += 'klavgpool|r:3,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192)) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*192) ) + d
		model_string += konvrfixed+'r:1,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d
		model_string += 'klavgpool|r:7,pad:valid,stride:2' + d


		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class QuickCIFAR_Konv_CIFAR10(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar10')
		model = self.QC_CIFAR10
		for coef in [1,5,6,7,8,9]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def QC_CIFAR10(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*32) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:4,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += konvrfixed+'r:7,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim


class QuickCIFAR_Konv_CIFAR100(Experiment_):
	''' The Konv_VGG network trained on CIFAR10 stripped of BN and Dropout
	The network is trained with variations of Initialization Coefficent.


	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
		                         epochnum=150,
		                         batchsz=128,
		                         shuffledata=True,
		                         numworkers=1,
		                         gpu=True)
		dataopt = DataOpts('cifar100')
		model = self.QC_CIFAR100
		for coef in [1,5,6,7,8,9]:
				''' Alpha Regularzation'''
				model_opt, optim_opt = model(dataopt, weight_decay=0, init_coef=coef, lr=1,conc=1,conc_train=False)
				experiment_name = model.__name__ + "|init_coef={}".format(coef)
				opt = allOpts(experiment_name, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
				              dataopts=dataopt)
				opt_list.append(opt)


		return opt_list
	def QC_CIFAR100(self,data_opts: DataOpts,
							init_coef=0.1,
	                        weight_decay=1e-4,
							lr=0.01,
	                        conc=1,
	                 conc_train=False,
							) -> Tuple[NetOpts, OptimOpts]:
		scale=1
		ceil = lambda x: str(math.ceil(x))
		''' With no regularization diverges'''
		model_string = ''
		d = '->'
		finish = 'fin'
		konvrfixed = 'konvr|param:logd,coef:{},conc:{},trainconc:{},isrelu:1,'.format(str(init_coef),str(conc),str(conc_train))
		konvsfixed = 'konvs|param:logd,coef:{},conc:{},trainconc:{},'.format(str(init_coef), str(conc),conc_train)
		nl = 'relu'

		model_string += 'tofin' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*32) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2' + d
		model_string += konvrfixed+'r:5,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += 'klavgpool|r:2,pad:valid,stride:2'  + d
		model_string += konvrfixed+'r:4,f:{},pad:same,stride:1,bias:1,icnum:1'.format(ceil(scale*64) ) + d
		model_string += konvrfixed+'r:7,f:' + str(data_opts.classnum) + ',pad:valid,stride:1,bias:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]# [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = constant_lr(init_lr=1)#vgg_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=None, reg_coef=None, reg_mode=None, alphaPrior=1,val_iters=1)
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',3),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=weight_decay,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduction='none')
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim