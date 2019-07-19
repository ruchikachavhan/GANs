from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets 
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
# use_cuda = 1
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


batchSize = 64 
imageSize = 64 

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])  

# Loading the dataset
dataset = torchvision.datasets .CIFAR10(root = './data', download = True, transform = transform) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



G = Generator().to(device)
G.apply(weights_init)


D = Discriminator().to(device)
D.apply(weights_init) 

# Training the DCGANs

criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 
optimizerG = optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 

Dis_loss=[]
gen_loss=[]
for epoch in range(25):
	print("***************Epoch is *******************", epoch +1) 
	for i, data in enumerate(dataloader, 0):
		D.zero_grad() 
		real, _ = data 
		input = Variable(real).to(device)
		target = Variable(torch.ones(input.size()[0])).to(device)
		output = D(input)
		output = output.to(device) 
		Derr_real = criterion(output, target) 
		z = Variable(torch.randn(input.size()[0], 100, 1, 1)).to(device)
		fake = G(z) 
		target = Variable(torch.zeros(input.size()[0])).to(device) 
		output = D(fake.detach()) 
		output = output.to(device)
		Derr_fake = criterion(output, target) 
		Dtotal_error = Derr_fake + Derr_real
		# Dis_loss.append(Dtotal_error.item())
		Dtotal_error.backward() 
		optimizerD.step() 


		G.zero_grad()
		target = Variable(torch.ones(input.size()[0])).to(device) 
		output = D(fake) 
		Gerr = criterion(output, target) 
		# gen_loss.append(Gerr)
		Gerr.backward() 
		optimizerG.step()  
		print("discriminator loss", Dtotal_error.item())
		print("generator loss", Gerr.item()) 
		if i % 100 == 0: 
			vutils.save_image(real, '%s/real.png' % "./results", normalize = True) 
			fake = G(z) 
			vutils.save_image(fake.data, '%s/fake_%03d.png' % ("./results", epoch), normalize = True) 

		# plt.ion()
		# plt.figure(200)
		# plt.plot(Dis_loss)
		# plt.figure(300)
		# plt.plot(gen_loss)
		# plt.show()
		# plt.pause(0.05)