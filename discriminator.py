from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
# use_cuda = 1
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


class Discriminator(nn.Module): 
    def __init__(self): 
        super(Discriminator, self).__init__() 
        self.model = nn.Sequential( 
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace = True), 
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), 
            nn.Sigmoid() 
        )

    def forward(self, input): 
        output = self.model(input) 
        return output.view(-1) 