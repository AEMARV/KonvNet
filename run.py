from src.experiment.Experiments import *
from src.experiment.IBP import *
import os

def boolprompt(question):
	answer=''
	while(answer.lower!='n' or answer.lower()!='y'):
		answer = 'n'# input(question+' [y]/[n]')
		if answer[0].lower()=='y':
			return True
		elif answer[0].lower()=='n':
			return False
		else:
			print('please answer with y/n characters')



if __name__ == '__main__':
	# exp = NIN_Dropout(1)
	# exp.run()

	# exp = Synthetic_PMaps(1)

	Exp = KNN_VGG_CIFAR10_NOBIAS_Conc(1).run()
	ibp = IBP_Experiment(Exp,'/home/student/Documents/Codes/Python/KonvNet/Results/KNN_VGG_CIFAR10/cifar10/KVGG_CIFAR10|init_coef=4/0')
	ibp.load()
