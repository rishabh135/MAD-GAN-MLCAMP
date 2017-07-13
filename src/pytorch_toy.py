from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import math
import numpy as np
from tqdm import *


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() 

parser = argparse.ArgumentParser()
parser.add_argument('--ngen', type = int, default = 3, help = 'number of generators')
parser.add_argument('--ndata', type = int, default = 100000, help = 'number of data per epoch')
parser.add_argument('--ncentres', type = int, default = 6, help = 'number of centres')
parser.add_argument('--std_dev', type = float, default = 0.1)
parser.add_argument('--lr', type = float, default = 0.0002, help = 'learning rate, default = 0.0002')
parser.add_argument('--beta1', type = float, default = 0.5, help = 'beta1 for adam. default = 0.5')
parser.add_argument('--nz', type = int, default = 3, help = 'size of the latent z vector')
parser.add_argument('--batchSize', type = int, default = 32, help = 'size of a batch')
parser.add_argument('--ndim', type = int, default = 2, help = 'Dimenstion to generate')
parser.add_argument('--R', type = int, default = 5, help = 'Radius of the circle')
parser.add_argument('--nvis', type = int, default = 3, help = 'Number of samples to be visualized')
parser.add_argument('--save_freq', type = int, default = 1, help = 'How frequently to save learned model')
parser.add_argument('--exp_name', default = '3gen/', help = 'Where to export the output')
parser.add_argument('--niter', type = int, default = 1200, help = 'number of epochs to train for')
parser.add_argument('--batchnorm', type = bool, default = True, help = 'Whether to do batchnorm')	#TODO if bool is correct

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

ngen = int(opt.ngen)
ndata = int(opt.ndata)
ncentres = int(opt.ncentres)
std_dev = float(opt.std_dev)
nz = int(opt.nz)
ndim = int(opt.ndim)
R = int(opt.R)
nvis = int(opt.nvis)
save_freq = int(opt.save_freq)
real_label = ngen
fake_labels = torch.LongTensor(ngen).copy_(torch.linspace(0, ngen-1, ngen))	#TODO: No need to have numpy here?

G = []
for i in range(ngen):	#TODO again: couldn't use clone over sequential
	G.append(nn.Sequential(
				nn.Linear(3, 128),
				nn.BatchNorm1d(128), #TODO: Add it
				nn.ReLU(),
				nn.Linear(128, 128),
				nn.BatchNorm1d(128),#TODO: Add it
				nn.ReLU(),
				nn.Linear(128, ndim)
			))

netD=nn.Sequential(
			nn.Linear(ndim, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Linear(128, ngen + 1)
		)
#TODO: Batch norm should be here as well
criterion = nn.CrossEntropyLoss()
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = []
for i in range(ngen):	#TODO:Can be problematic, any way of using model_utils.combine_all_parameters?
	optimizerG.append(optim.Adam(G[i].parameters(), lr = opt.lr, betas = (opt.beta1, 0.999)))

noise = Variable(torch.FloatTensor(opt.batchSize, nz))
input = Variable(torch.FloatTensor(opt.batchSize, ndim))
label = Variable(torch.LongTensor(opt.batchSize))
fake_cache = Variable(torch.FloatTensor(ngen, opt.batchSize, ndim))
real = torch.FloatTensor(opt.batchSize, ndim)
randints = torch.FloatTensor(opt.batchSize)
#parametersD, gradParametersD = netD:getParameters()
#parametersG, gradParametersG = model_utils.combine_all_parameters(G)

for epoch in tqdm(range(opt.niter)):
	for iter in range(ndata/opt.batchSize):
		############################
		# (1) Update D network
		###########################
		print("Started updating discriminator")
		errD_total = 0
		for i in range(ngen):
			netD.zero_grad() #TODO: should I get it out of for loop
			# train with real
			randints.random_(1, ncentres) #simply doing randints.random_(1, ncentres) gave error in ipython
			#print(randints)
			for j in range(opt.batchSize):
				k = randints[j]
				real[j][0] = torch.normal(means = torch.FloatTensor([0.0]), std = std_dev)[0] + R*math.cos((2*k*math.pi)/ncentres)
				real[j][1] = torch.normal(means = torch.FloatTensor([0.0]), std = std_dev)[0] + R*math.sin((2*k*math.pi)/ncentres)
			input.data.copy_(real)
			label.data.fill_(real_label)
			output = netD.forward(input)
			errD_real = criterion(output, label)
			errD_real.backward()

			# train with fake
			noise.data.normal_(0, 1)
			fake = G[i].forward(noise)
			fake_cache[i].data.copy_(fake.data)
			input.data.copy_(fake.data)
			label.data.fill_(fake_labels[i])
			output = netD.forward(input)
			errD_fake = criterion(output, label)
			errD_fake.backward()
			errD  =  errD_real + errD_fake
			errD_total = errD_total + errD
			optimizerD.step()	#TODO: should I get it out of for loop, but then will it use errD, how will effect of gradients change (as gradients change in the for loop)

		############################
		# (2) Update G network
		###########################
		print("Started updating generator")
		label.data.fill_(real_label)
		errG_total = 0
		for i in range(ngen):
			G[i].zero_grad()
			#output = netD.forward(G[i].output) #TODO: try it
			output = netD.forward(fake_cache[i])
			errG = criterion(output, label)
			errG.backward()
			optimizerG[i].step()
			errG_total = errG_total + errG
	
	if (epoch%save_freq == 0):
		print("saving epoch")
		try:
			os.makedirs(opt.exp_name + str(epoch))
		except OSError:
			pass
		randints.random_(1, ncentres)
		inp = np.zeros((opt.batchSize, 2))
		for j in range(opt.batchSize):
			k=randints[j]
			inp[j][0] = torch.normal(means = torch.FloatTensor([0.0]), std = std_dev)[0] + R*math.cos((2*k*math.pi)/ncentres)
			inp[j][1] = torch.normal(means = torch.FloatTensor([0.0]), std = std_dev)[0] + R*math.sin((2*k*math.pi)/ncentres)
		plt.scatter(inp[:, 0], inp[:, 1])
		plt.savefig(opt.exp_name + str(epoch) + '/input.png')
		plt.close()

		out = np.zeros((opt.batchSize * ngen * nvis, 2))
		for i in range(ngen):
			for j in range(nvis):
				noise.data.normal_(0, 1)
				fake = G[i].forward(noise)
				out[(i * nvis + j)* opt.batchSize : (i * nvis + j + 1)* opt.batchSize][:]=fake.data.numpy()
		plt.scatter(out[:, 0], out[:, 1])
		plt.savefig(opt.exp_name + str(epoch) + '/output.png')
		print("figure saved " + opt.exp_name + str(epoch) + '/output.png' )
		plt.close()