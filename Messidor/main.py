import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import progress_bar, weights_init
import torch
import torchvision
from networks import InceptionV3
from sklearn.metrics import roc_auc_score
import numpy as np
#from transforms import *

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)#随机数
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)#返回一个0到n-1的数组

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


## Parameters Setting
rootDir =  "/home/lrh/dataset/messidor/train"
testRootDir =  "/home/lrh/dataset/messidor/test_train"
batchSize = 32
testBatchSize = 16

nEpochs = 100
learningRate = 0.01
feature_extracted = True

lamda = 1e-5

isTrain = False

# Load Data
print("=====> Load Data")
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
train_transforms = transforms.Compose([
    transforms.Resize((299,299)),
#    transforms.RandomCrop((299,299)),
    transforms.RandomRotation(20),
#    transforms.Rotate(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])
test_transforms = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

if isTrain is True:
    trainset = torchvision.datasets.ImageFolder(rootDir,transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batchSize,shuffle=True,drop_last=False,num_workers=4)
testset = torchvision.datasets.ImageFolder(testRootDir,transform=test_transforms)
testloader = torch.utils.data.DataLoader(testset,batch_size=testBatchSize,shuffle=True,drop_last=False,num_workers=4)

print("==> Building models...")
# clf = net.VGG_s()
# clf.apply(weights_init)
#clf = net.AttnVGG(num_classes=2, attention=True, normalize_attn=True, dropout = 0.8)
clf = InceptionV3(num_classes=4)
#clf = models.vgg16_bn(pretrained=True)
# if feature_extracted:
#     for param in clf.parameters():
#         param.requires_grad = False
#
#
# classifier = nn.Sequential(
#     nn.Linear(25088,4096),
#     nn.Dropout(),
#     nn.ReLU(inplace=True),
#     nn.Linear(4096,4096),
#     nn.Dropout(),
#     nn.ReLU(inplace=True),
#     nn.Linear(4096,2)
# )
# clf.classifier = classifier

clf = torch.nn.DataParallel(clf)
cudnn.benchmark = True
clf = clf.cuda()
#print(clf)


params_to_update = []
for name,param in clf.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
#       print("\t",name)


#define loss and optimizer
print("==> Define loss and optimizer")
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(params_to_update,lr = 0.003,weight_decay = 0.0005, momentum = 0.95)
optimizer = optim.SGD(clf.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3, nesterov=True)
#optimizer = optim.Adam(params_to_update, lr = learningRate, betas =(0.5, 0.999))


def train(epoch):
    print("Epoch : %d" %epoch)
    clf.train()
    train_loss = 0
    correct = 0
    total = 0
    train_r_loss = 0

    ## decreased linearly from 0.01 -> 0.0001
    if epoch < 10:
        optimizer = optim.SGD(clf.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3, nesterov=True)
    elif epoch < 20:
        optimizer = optim.SGD(clf.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3, nesterov=True)
    else:
        optimizer = optim.SGD(clf.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-3, nesterov=True)

    for batch_idx, (images,labels) in enumerate(trainloader):
#        print(images.shape)

        clf.zero_grad()
        images,labels = images.cuda(),labels.cuda()
#        images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)
        outputs = clf(images)
#        outputs,aux = clf(images)

        if isinstance(outputs,tuple):
            outputs = outputs[0]
#        loss = criterion(outputs,labels)

        ## L_1 regularization
        regularization_loss = 0
        for param in clf.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        c_loss = criterion(outputs,labels)
#        c_loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss = c_loss + lamda * regularization_loss
#        print(regularization_loss)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_r_loss += lamda * regularization_loss.item()
        ## calculate train accuracy
        predicted = torch.argmax(outputs,dim=1)

        #print(labels)
        #print(predicted)

        total += labels.shape[0]
        correct += predicted.eq(labels).sum()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | R_Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), train_r_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))
#        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))
    avg_loss = float(train_loss)/(batch_idx+1)
    train_acc = 100.*float(correct)/total
    return avg_loss, train_acc

def test():
    print('\n Test')
    clf.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        test_acc = []
        for batch_idx, (images,labels) in enumerate(testloader):
            images,labels = images.cuda(),labels.cuda()
            outputs = clf(images)
            if isinstance(outputs,tuple):
                outputs = outputs[0]
            #print(outputs)
            predicted = torch.argmax(outputs,dim=1)
            #print(labels)
            #print(predicted)
            total += labels.shape[0]
            correct += predicted.eq(labels).sum()
            progress_bar(batch_idx, len(testloader), ' Acc: %.3f%% (%d/%d)'
                % (100.*float(correct)/total, correct, total))

    acc = 100.*float(correct)/total
    return acc


if __name__=='__main__':

    best_acc = 0
    if isTrain is True:
        g_loss = []
        train_ACC = []
        test_ACC = []
        for epoch in range(nEpochs):
            avg_loss,train_acc = train(epoch)
            g_loss.append(avg_loss)
            train_ACC.append(train_acc)
            acc = test()
            test_ACC.append(acc)
            # save the best model with the highest accuracy in test dataset
            if acc > best_acc:
                best_acc = acc
                torch.save(clf.state_dict(),'messidor_inceptionv3.pkl')
                print("best_acc: %.3f" % (best_acc))

        loss = np.array(g_loss)
        np.save("npyresults/loss.npy",loss)
        train_acc = np.array(train_ACC)
        np.save("npyresults/train_acc.npy",train_acc)
        test_acc = np.array(test_ACC)
        np.save("npyresults/test_acc.npy",test_acc)
    else:
        model_file = "messidor_inceptionv3.pkl"
        clf.load_state_dict(torch.load(model_file))
        print('\n Test')
        clf.eval()
        correct = 0
        total = 0
        gt = torch.FloatTensor()
        pred = torch.FloatTensor()

        with torch.no_grad():
            for batch_idx, (images,labels) in enumerate(testloader):
                images = images.cuda()
                outputs = clf(images)
                if isinstance(outputs,tuple):
                    outputs = outputs[0]
                pred = torch.cat((pred,torch.sigmoid(outputs).cpu()),0)

                labels = labels.unsqueeze(1)
                one_hot_labels = torch.zeros(len(labels),4).scatter_(1,labels,1)
                gt = torch.cat((gt,one_hot_labels),0)

                total = len(gt)
                correct = torch.argmax(gt,1).eq(torch.argmax(pred,1)).sum()
                progress_bar(batch_idx, len(testloader), ' Acc: %.3f%% (%d/%d)'
                    % (100.*float(correct)/total, correct, total))

        AUC_ROC = roc_auc_score(gt,pred,average='micro')
        print("AUC :"+str(AUC_ROC))

        acc = 100.*float(correct)/total
        print("ACC:%.4f" %acc)
#        test()
