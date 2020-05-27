import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNet(nn.Module):
    def __init__(self):
        super(CNet,self).__init__()
        #input_c,output_c,kernel_size,stride,padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,5,1,2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5,1,2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear = nn.Sequential(
            nn.Linear(16 * 7 * 7 , 120),
            nn.ReLU(),
            nn.Linear(120 , 84),
            nn.ReLU(),
            nn.Linear(84 , 10)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        #print(x.shape)
        x = self.linear(x)
        x = torch.softmax(x,dim=1)
        return x

import torch
if __name__=='__main__':
    cnet = CNet()
    cnet = cnet.cuda()
    x = torch.randn(16,1,28,28).cuda()
    print(cnet(x).shape)
