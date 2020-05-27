import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from PIL import Image
import torch
import re
import os
import sys
sys.path.append("..")
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
import libadver.utils as u
import libadver.models.generators as generators
from utilities import *
from transforms import *
import torch.optim as optim
import torch.backends.cudnn as cudnn
from read_data import ChestXrayDataSet
import libadver.attack as attack
import math

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

print('===>load  data')
DATA_DIR = '/home/lrh/dataset/ChestXray-NIHCC/images_v1_small'
testTXTFile = '/home/lrh/git/CheXNet/ChestX-ray14/labels/test.txt'
trainTXTFile = '/home/lrh/git/CheXNet/ChestX-ray14/labels/train.txt'
normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                image_list_file=trainTXTFile,
                                transform = train_transform
                                )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16,
                         shuffle=True, num_workers=8, pin_memory=True)
test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                image_list_file=testTXTFile,
                                transform = test_transform
                                )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16,
                         shuffle=False, num_workers=8, pin_memory=True)
print('\ndone')

print('===>load model')
net = DenseNet121(14)
criterion = nn.BCELoss()
print('done\n')

print('\nmoving models to GPU ...')
clf = net.cuda()
clf = torch.nn.DataParallel(clf)
cudnn.benchmark = True
criterion = criterion.cuda()
print('done\n')

optimizer = optim.Adam(clf.parameters(), lr = 2e-3, betas = (0.9, 0.999))

epochNum = 30

pgd_params_train = {
            'ord': np.inf,
            'y': None,
            'eps': 2.0 / 255,
            'eps_iter': 1 / 255,
            'nb_iter': 1,
            'rand_init': True,
            'rand_minmax': 50.0 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True,
            'criterion' : nn.BCELoss()
        }

pgd_params_test = {
            'ord': np.inf,
            'y': None,
            'eps': 4.0 / 255,
            'eps_iter': 1 / 255,
            'nb_iter': 4,
            'rand_init': True,
            'rand_minmax': 50.0 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True,
            'criterion' : nn.BCELoss()
        }

acc_small = 0
acc_large = 100
for epoch in range(epochNum):
    torch.cuda.empty_cache()

    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    #run for one epoch

    for  batchIdx,(image,label) in enumerate(train_loader):
        image = image.cuda()
        label = label.cuda()
        p = np.random.uniform(0.0,1.0)
        if p > 0.5:
            clf.eval()
            PGDAttack_train = attack.ProjectGradientDescent(model = clf)
            e = np.random.uniform(0.01,0.04)
            n = math.floor(np.random.uniform(1,10))
            pgd_params_train['y'] = label
            pgd_params_train['clip_max'] = torch.max(image)
            pgd_params_train['clip_min'] = torch.min(image)
            pgd_params_train['eps'] = e
            pgd_params_train['nb_iter'] = n
            images = PGDAttack_train.generate(image, **pgd_params_train)
        else:
            images = image
#         clf.eval()
#         PGDAttack_train = attack.ProjectGradientDescent(model = clf)
# #        e = np.random.uniform(0.01,0.04)
# #        n = math.floor(np.random.uniform(1,10))
#         e = 4.0/255
#         n = 4
#         pgd_params_train['y'] = label
#         pgd_params_train['clip_max'] = torch.max(image)
#         pgd_params_train['clip_min'] = torch.min(image)
#         pgd_params_train['eps'] = e
#         pgd_params_train['nb_iter'] = n
#         images = PGDAttack_train.generate(image, **pgd_params_train)

        clf.train()
        clf.zero_grad()
        optimizer.zero_grad()
        output = clf(images)

        loss = 0
        for i in range(14):
            loss_iter = criterion(output[:,i],label[:,i])
            loss = loss + loss_iter
        loss = loss/14.0
        loss.backward()
        optimizer.step()

        predict = output.type(torch.cuda.FloatTensor)
        labels = label.type(torch.cuda.FloatTensor)

        gt = torch.cat((gt,labels.detach().cpu()),0)
        pred = torch.cat((pred,predict.detach().cpu()),0)

        ACCs_iter=[]
        for i in range(14):
            predictLabels= pred[:, i] > 0.5
            predictLabels = predictLabels.float()
            acc = float((gt[:, i] == predictLabels).sum() )/ gt.shape[0]
            ACCs_iter.append(acc)
        ACCs_avg = np.array(ACCs_iter).mean()

        u.progress_bar(batchIdx,len(train_loader) , 'loss:%.3f | Acc: %.3f%%'
                            % (loss, 100.*ACCs_avg))
    clf.eval()
    with torch.no_grad():
        gt = torch.FloatTensor()
        pred_advx = torch.FloatTensor()
        for i,(image,label) in enumerate(test_loader):
            image = image.cuda()
            label = label.cuda()
            pgd_params_test['y'] = label
            pgd_params_test['clip_max'] = torch.max(image)
            pgd_params_test['clip_min'] = torch.min(image)
            PGDAttack_test = attack.ProjectGradientDescent(model = clf)
            adv_test = PGDAttack_test.generate(image, **pgd_params_test)
            pred = clf(adv_test)

            gt = torch.cat((gt, label.detach().cpu()), 0)
            pred_advx = torch.cat((pred_advx,pred.detach().cpu()),0)

        ACCs = []
        for i in range(14):

            predictLabels_advx = pred_advx[:, i] > 0.5
            predictLabels_advx = predictLabels_advx.float()

            acc = float((gt[:, i] == predictLabels_advx).sum() )/ gt.shape[0]
            ACCs.append(acc)

        ACCs_avg = np.array(ACCs).mean()

        print('The average ACC is %.3f' %(ACCs_avg))
        if ACCs_avg>acc_small:
            torch.save(clf.state_dict(), os.path.join('/home/lrh/store/modelpath/adversarial_defense/ChestXay','adversarial_training_MPAdvT.pth'))
            acc_small = acc
