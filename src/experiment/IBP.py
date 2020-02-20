import torch
from src.experiment.Experiment import Experiment_
from src.trainvalid.epocher import Epocher
from src.netparsers.staticnets import StaticNet
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import typing
import pandas as pd
# sns.set(rc={'text.usetex':True,'text.latex.preamble':[r'\usepackage{amsfonts}', r'\usepackage{graphicx}']})
class IBP_Experiment(Experiment_):
	def __init__(self, exp:Experiment_, path):
		self.exp = exp
		self.path = path
		self.opt_list= []
	def extract_epoch_paths(self,path):
		filelist = os.listdir(path)
		model_list = []
		for filename in filelist:
			if filename.split('.')[-1] == 'model':
				model_list = model_list + [filename]
		return model_list
	def collect_opts(self):
		pass
	def load(self):
		self.opt_list = self.exp.collect_opts()

		for opt in self.opt_list:  # type: allOpts
			ep = Epocher(opt)
			epoch_dict= dict()
			filelist =self.extract_epoch_paths(self.path)
			data = next(iter(ep.testloader))
			for statefile in filelist:
				state_path = os.path.join(self.path,statefile)
				epocher_state = torch.load(state_path)
				epoch = epocher_state['epoch']
				ep.model.to(ep.opts.epocheropts.device)
				ep.model.load_state_dict(epocher_state['model_state'])
				print(epoch)

				mut_dict = self.calculate_mutual(ep,data)
				epoch_dict[str(epoch)]= mut_dict

			self.prepare_data_plot(epoch_dict)
			return epoch_dict

	def prepare_data_plot(self, epoch_dict:dict):
		total_frame = pd.DataFrame()
		for mut_type in list(list(epoch_dict.values())[0].values())[0].keys():
			for layerName in list(epoch_dict.values())[0].keys():# layerName:str
				vals = []
				epochs = []
				if not ('Konv_R' in layerName):
					continue
				for epochstr in epoch_dict.keys():
					vals = vals+ [ epoch_dict[epochstr][layerName][mut_type]]
					epochs = epochs + [int(epochstr)]
				clean_dict = dict(epoch=epochs, modulename=layerName, target=mut_type, nats=vals)
				temp_frame = pd.DataFrame(clean_dict)
				total_frame = total_frame.append(temp_frame)
		plot = sns.relplot(x="epoch", y="information", hue="target", height=3, aspect=1, col_wrap=5,
		                   linewidth=1.5, col="modulename", style='infotype',
		                   kind="line", data=total_frame)
		plot.savefig(os.path.join(self.path,"LossCompare.svg"))
		plt.show()







	def calculate_mutual(self,epocher,data):
		mut_calc = MutualInfoCalculator(epocher.model)
		with torch.set_grad_enabled(False):
			totalsamples= 0
			corrects = 0

			inputs, labels = data
			inputs, labels = inputs.to(epocher.opts.epocheropts.device), labels.to(epocher.opts.epocheropts.device)
			mut_dict_corrects = mut_calc.calc_stats_corrects(inputs,labels)

				# break

		epocher.model.to(torch.device('cpu'))
		return mut_dict_corrects

	def validate_epocher(self, epocher):
		acc =0
		with torch.set_grad_enabled(False):
			totalsamples= 0
			corrects = 0
			for batch_n,data in enumerate(epocher.testloader):
				inputs, labels = data
				inputs, labels = inputs.to(epocher.opts.epocheropts.device), labels.to(epocher.opts.epocheropts.device)
				output = epocher.model(inputs)[0]
				output = output.log_softmax(dim=1)
				# output = alpha_lnorm(output,1,1.2)
				# output = output - ((alpha*output).logsumexp(dim=1,keepdim=True))/alpha
				outputfull = output
				output = output.view(-1, epocher.opts.dataopts.classnum)
				acc_temp = epocher.get_acc(output,labels)
				loss = epocher.opts.optimizeropts.loss(output, labels).mean()
				corrects = (corrects + loss*inputs.shape[0])
				totalsamples += inputs.shape[0]
				print(corrects/(totalsamples))

				# break

		epocher.model.to(torch.device('cpu'))


class MutualInfoCalculator(object):
	def __init__(self, statnet:StaticNet):
		self.statnet=  statnet
		self.samples_seen = 0

	def mutual_info_label(self, curtensor, labels, numsamp=100):
		import numpy.random
		mutual_total = 0
		for i in range(numsamp):
			randsample_idx = numpy.random.randint(0, curtensor.shape[0])
			randlabel = labels[randsample_idx]
			randsample = curtensor[randsample_idx:randsample_idx + 1]
			samp, logprob = self.sample_manual(randsample)
			same_labels_inputs = curtensor[labels == randlabel]
			ent_pxgy = (samp * same_labels_inputs).sum(dim=(1, 2, 3, 4)).logsumexp(dim=0) - numpy.log(
				same_labels_inputs.shape[0])
			ent_px = (samp * curtensor).sum(dim=(1, 2, 3, 4)).logsumexp(dim=0) - numpy.log(curtensor.shape[0])
			mutual = ent_px - ent_pxgy
			mutual_total += -mutual

		mutual_total = mutual_total / numsamp
		return mutual_total

	def calc_stats(self,inputs, labels):
		ret_dict = {}
		x = inputs
		for i, layer in enumerate(self.statnet.layerlist):
			#layer : MyModule
			x = layer(x)
			mut_cur_label = self.mutual_info_label(x, labels)
			mut_cur_input = self.mutual_info_label(x, torch.arange(0,x.shape[0]))
			temp_dict = dict(input=mut_cur_input.item(),label=mut_cur_label.item())
			ret_dict[layer.compact_name] = temp_dict
		return ret_dict

	def calc_stats_corrects(self,inputs, labels):
		ret_dict = {}
		x = inputs
		y = self.statnet(x)[0]
		corrects = (y.argmax(dim=1,keepdim=True).squeeze()==labels).int()
		print(corrects.float().mean())
		for i, layer in enumerate(self.statnet.layerlist):
			#layer : MyModule
			x = layer(x)
			mut_cur_label = self.mutual_info_label(x, corrects)
			temp_dict = dict(correct_label=mut_cur_label.item())
			ret_dict[layer.compact_name] = temp_dict
		return ret_dict

	def calc_stats_pervariable(self,inputs, labels):
		ret_dict = {}
		x = inputs
		for i, layer in enumerate(self.statnet.layerlist):
			#layer : MyModule
			x = layer(x)
			x_var = x.logsumexp(dim=2,keepdim=True).logsumexp(dim=3,keepdim=True) - np.log(x.shape[2])- np.log(x.shape[3])
			mut_cur_label = self.mutual_info_label(x_var, labels)
			mut_cur_input = self.mutual_info_label(x_var, torch.arange(0,x.shape[0]))
			temp_dict = dict(mutual_input=mut_cur_input.item(),mutual_label=mut_cur_label.item())
			ret_dict[layer.compact_name] = temp_dict
		return ret_dict



	def sample_manual(self, lp: torch.Tensor, axis=1, manualrand=None, concentration=1):
		lnorm = lp.logsumexp(dim=axis,keepdim=True).detach()
		lp = lp-lnorm
		lp = lp.transpose(0,axis)
		p = lp.exp()
		cumprob = p.cumsum(dim=0)

		if manualrand is not None:
			rand = manualrand
			rand = rand.transpose(axis,0)
		else:
			rand = torch.rand_like(p[0:1])
		samps = cumprob >= rand
		samps[1:] = samps[1:] ^ samps[0:-1]
		samps = samps.type_as(p).detach()
		logprob = samps*lp
		logprob[logprob != logprob] = 0

		logprob = logprob.sum(dim=0,keepdim=True)
		samps = samps.transpose(0,axis)
		logprob = logprob.transpose(0,axis)
		return samps.detach(), logprob