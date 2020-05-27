import torch
import torch.utils.data as data
import pandas
import os
from torchvision.transforms import transforms
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from libadver.utils import *
import torch.backends.cudnn as cudnn
#from read_data import ISIC2019
from networks import AttnVGG
import numpy as np
from loss import FocalLoss
import torch.optim.lr_scheduler as lr_scheduler
import libadver.utils as u
from transforms import *
from data import ISIC
import libadver.attack as attack
import math

print('====>load data')
mean = (0.7012, 0.5517, 0.4875)
std = (0.0942, 0.1331, 0.1521)
normalize = Normalize(mean,std)
transform_train = transforms.Compose([
    Resize((256,256)),
    RandomCrop((224,224)),
    RandomRotate(),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ToTensor(),
    normalize
])
transform_test = transforms.Compose([
    Resize((256,256)),
    RandomCrop((224,224)),
    ToTensor(),
    normalize
])

trainCSVFile = '/home/lrh/git/Evaluating_Robustness_Of_Deep_Medical_Models/Dermothsis/train.csv'
trainset = ISIC(csv_file=trainCSVFile, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True,
    num_workers=8, drop_last=True)

testCSVFile = '/home/lrh/git/Evaluating_Robustness_Of_Deep_Medical_Models/Dermothsis/test.csv'
testset = ISIC(csv_file=testCSVFile, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,batch_size = 16,shuffle = False)
print('\ndone')

print('======>loading the model')

net = AttnVGG(num_classes=2,attention=True,normalize_attn=False)
criterion = FocalLoss()
print('done\n')

print('\nmoving models to GPU')
clf = net.cuda()
clf = torch.nn.DataParallel(clf)
cudnn.benchmark = True
criterion = criterion.cuda()
print('done\n')

learningRate = 0.001
optimizer = optim.SGD(clf.parameters(),lr = learningRate, momentum=0.9, weight_decay=1e-4,nesterov=True)
lr_lambda = lambda epoch : np.power(0.1, epoch//10)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
            'sanity_checks': True
        }

pgd_params_test = {
            'ord': np.inf,
            'y': None,
            'eps': 2.0 / 255,
            'eps_iter': 1 / 255,
            'nb_iter': 2,
            'rand_init': True,
            'rand_minmax': 50.0 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True
        }
acc_test = 0
for epoch in range(epochNum):
    torch.cuda.empty_cache()
    #adjust learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print("\nepoch %d learning rate %f\n" % (epoch+1, current_lr))
    total = 0
    correct = 0
    a_t = 0
    for i, data in enumerate(trainloader,0):
        image,label = data['image'],data['label']
        image = image.cuda()
        label = label.cuda()
        p = np.random.uniform(0.0,1.0)
        if p > 0.5 :
            clf.eval()
            PGDAttack_train = attack.ProjectGradientDescent(model = clf)
            e = np.random.uniform(0.01,0.04)
            n = math.floor(np.random.uniform(1,10))
            # e = 4.0/255
            # n = 4
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
# # #            e = np.random.uniform(0.01,0.04)
# # #            n = math.floor(np.random.uniform(1,10))
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

        pred,_,_ = clf(images)
#        pred = torch.sigmoid(pred)
        loss = criterion(pred,label)
        loss.backward()
        optimizer.step()

        predict = torch.argmax(pred,1)
        total += label.size(0)
        correct += torch.eq(predict,label).sum().double().item()

        u.progress_bar(i, len(trainloader), 'loss:%.3f | Acc: %.3f%% (%d/%d)'
                    % (loss, 100.*float(correct)/total, correct, total))
    #adversarial test
    clf.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i , data in enumerate(testloader,0):
            image,label = data['image'],data['label']
            image = image.cuda()
            label = label.cuda()
            pgd_params_test['y'] = label
            pgd_params_test['clip_max'] = torch.max(image)
            pgd_params_test['clip_min'] = torch.min(image)
            PGDAttack_test = attack.ProjectGradientDescent(model = clf)
            adv_test = PGDAttack_test.generate(image, **pgd_params_test)
            pred,_,_ = clf(adv_test)
            predict = torch.argmax(pred,1)
            total += label.size(0)
            correct += torch.eq(predict,label).sum().double().item()
        acc = 100*(correct/total)
        print(acc)
        if acc > acc_test:
            checkpoint = {
            'state_dict': clf.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join('/home/lrh/store/modelpath/adversarial_defense/Dermothsis','adv_training_MPAdvT.pth'))
            acc_test = acc
