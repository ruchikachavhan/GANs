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

class Generator(nn.Module): 

    def __init__(self): 
        super(Generator, self).__init__() 
        self.model = nn.Sequential( 
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), 
            nn.Tanh() 
        )

    def forward(self, input): 
        output = self.model(input) 
        return output