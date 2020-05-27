from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

#from wideresnet import *
from networks import InceptionV3
from sklearn.metrics import roc_auc_score
import numpy as np
import libadver.utils as u
#from resnet import *
from mart import mart_loss
import numpy as np
import torch.backends.cudnn as cudnn
import math
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch MELANOMA MART Defense')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=7e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=8.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='wideresnet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
print('=====>load data')
trainRootDir = "/home/lrh/dataset/messidor/train"
testRootDir = "/home/lrh/dataset/messidor/test"

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
train_transforms = transforms.Compose([
        #    transforms.CenterCrop((1500,1500)),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
])
test_transforms = transforms.Compose([
        #    transforms.CenterCrop((1500,1500)),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
])
trainset = torchvision.datasets.ImageFolder(trainRootDir,transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=2,shuffle=True,drop_last=False,num_workers=4)
testset = torchvision.datasets.ImageFolder(testRootDir,transform=test_transforms)
testloader = torch.utils.data.DataLoader(testset,batch_size=16,shuffle=False,drop_last=False,num_workers=4)
print('done\n')

print("==> Building models...")
clf = InceptionV3(num_classes=4)
clf = torch.nn.DataParallel(clf)
cudnn.benchmark = True
clf = clf.cuda()
params_to_update = []
for name,param in clf.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
#       print("\t",name)
print('\ndone')

#define loss and optimizer
print("==> Define loss and optimizer")
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(clf.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3, nesterov=True)


def train(args, model, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target)in enumerate(trainloader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        #epsilon = np.random.uniform(0.01,0.04),
        epsilon = 4.0/255
        perturb_steps = 4
        # print(epsilon[0])
        # print(type(epsilon[0]))
        perturb_steps = np.random.randint(1,10)

       # calculate robust loss
        loss = mart_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon= epsilon,
                           perturb_steps= perturb_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))

# def adjust_learning_rate(optimizer, epoch):
#     """decrease the learning rate"""
#     lr = args.lr
#     if epoch >= 100:
#         lr = args.lr * 0.001
#     elif epoch >= 90:
#         lr = args.lr * 0.01
#     elif epoch >= 75:
#         lr = args.lr * 0.1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon= 4.0/255,
                  num_steps=4,
                  step_size=args.epsilon/10.0):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def eval_adv_test_whitebox(model, testloader):

    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for (data, target) in testloader:
        data = data.cuda()
        target = target.cuda()
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_acc: ', 1 - natural_err_total / len(testloader.dataset))
    print('robust_acc: ', 1- robust_err_total / len(testloader.dataset))
    return 1 - natural_err_total / len(testloader.dataset), 1- robust_err_total / len(testloader.dataset)


def main():

    model = clf

#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(clf.parameters(),lr = learningRate, momentum=0.9, weight_decay=1e-4,nesterov=True)
    # lr_lambda = lambda epoch : np.power(0.1, epoch//10)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
#    natural_acc = []
#    robust_acc = []
    acc_small = 0
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
#        adjust_learning_rate(optimizer, epoch)

        start_time = time.time()

        # adversarial training
        train(args, model, trainloader, optimizer, epoch)


        print('================================================================')

        natural_err_total, robust_err_total = eval_adv_test_whitebox(model, testloader)

        print('using time:', time.time()-start_time)

#        natural_acc.append(natural_err_total)
#        robust_acc.append(robust_err_total)
#        print('natural acc : %.f ' % natural_err_total)
#        print('robust acc: %.f ' %robust_err_total)
        print('================================================================')
        acc = robust_err_total
        if acc > acc_small:
            torch.save(clf.state_dict(), os.path.join('/home/lrh/store/modelpath/adversarial_defense/Messidor','adversarial_mart2.pth'))
            acc_small = acc
        # file_name = os.path.join(log_dir, 'train_stats.npy')
        # np.save(file_name, np.stack((np.array(natural_acc), np.array(robust_acc))))
        #
        # # save checkpoint
        # if epoch % args.save_freq == 0:
        #     torch.save(model.state_dict(),
        #                os.path.join(model_dir, 'model-res-epoch{}.pt'.format(epoch)))
        #     torch.save(optimizer.state_dict(),
        #                os.path.join(model_dir, 'opt-res-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
