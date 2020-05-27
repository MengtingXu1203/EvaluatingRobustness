import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
#from utils import progress_bar, weights_init
import torch
import torchvision
from networks import InceptionV3
from sklearn.metrics import roc_auc_score
import numpy as np
import libadver.utils as u
import math
#from transforms import *

class ProjectGradientDescent():
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to 0. or the
    Madry et al. (2017) method when rand_minmax is larger than 0.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    """
    def __init__(self, model):
        if not isinstance(model, nn.Module):
            raise TypeError("The model argument should be an instance of"
                          "torch.nn.Module")
        self.model = model
        self.default_rand_init = True

    def generate(self, inputs, **kwargs):
        """
        To generate adversarial samples corresponding to batch images x.

        Generate function paramters
        :param inputs : input image, torch floatTensor with shape [None, in_channel, height, width]
        """
        # assure parameters parse
        assert self.parse_params(**kwargs)
        ## judge targeted or non-targeted
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        else:
            y = self.y
            targeted = False


        x = inputs.detach()
        ## with random perturbations
        if self.rand_init:
            #print(self.eps)
            x = x + torch.zeros_like(x).uniform_(-self.eps, self.eps)

        ## BIM
        criterion = self.criterion
        for i in range(self.nb_iter):
            self.model.zero_grad()
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]

                loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            #print(grad)
#            print(self.eps_iter)
            gradient = self.eps_iter * torch.sign(grad)
#            print(gradient[1,1,1,1])
            if targeted is True:
                x = x - gradient
            else:
                x = x + gradient
#                print(x[1,1,1,1])

            ## norm constrains on perturbations
            x = torch.min(torch.max(x, inputs - self.eps), inputs + self.eps)
#            print(torch.min(x))
#            print(torch.max(x))
            x = torch.clamp(x, self.clip_min, self.clip_max)
#            print(torch.min(x))
#            print(torch.max(x))

        return x


    def parse_params(self,
                   eps=0.3,
                   eps_iter=0.05,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   rand_init=None,
                   rand_minmax=0.3,
                   sanity_checks=True,
                   criterion = nn.CrossEntropyLoss(),
                   **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (optional float) step size for each attack iteration
        :param nb_iter: (optional int) Number of attack iterations.
        :param y: (optional) A tensor with the true labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param sanity_checks: bool Insert tf asserts checking values
            (Some tests need to run with no sanity checks because the
             tests intentionally configure the attack strangely)
        """

        # Save attack-specific parameters
        self.eps = eps
        if rand_init is None:
          rand_init = self.default_rand_init
        self.rand_init = rand_init
        if self.rand_init:
          self.rand_minmax = eps
        else:
          self.rand_minmax = 0.
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.criterion = criterion

        if isinstance(eps, float) and isinstance(eps_iter, float):
          # If these are both known at compile time, we can check before anything
          # is run. If they are tf, we can't check them yet.
          assert eps_iter <= eps, (eps_iter, eps)

        if self.y is not None and self.y_target is not None:
          raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
          raise ValueError("Norm order must be either np.inf, 1, or 2.")
        self.sanity_checks = sanity_checks

        if len(kwargs.keys()) > 0:
          warnings.warn("kwargs is unused and will be removed on or after "
                        "2019-04-26.")

        return True


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
            'eps': 4.0 / 255,
            'eps_iter': 1 / 255,
            'nb_iter': 4,
            'rand_init': True,
            'rand_minmax': 50.0 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True
        }
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
trainloader = torch.utils.data.DataLoader(trainset,batch_size=16,shuffle=True,drop_last=False,num_workers=4)
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
optimizer = optim.SGD(clf.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3, nesterov=True)

epochNum = 30
#lamda = 1e-5
lamda = 0.01
acc_small = 0
for epoch in range(epochNum):
    torch.cuda.empty_cache()
    correct = 0
    total = 0
    for  batchIdx,(image,label) in enumerate(trainloader):
        image = image.cuda()
        label = label.cuda()
        p = np.random.uniform(0.0,1.0)
        if p > 0.5:
            clf.eval()
            PGDAttack_train = ProjectGradientDescent(model = clf)
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
        # clf.eval()
        # PGDAttack_train = ProjectGradientDescent(model=clf)
        # e = 4.0/255
        # n = 4
        # pgd_params_train['y'] = label
        # pgd_params_train['clip_max'] = torch.max(image)
        # pgd_params_train['clip_min'] = torch.min(image)
        # pgd_params_train['eps'] = e
        # pgd_params_train['nb_iter'] = n
        # images = PGDAttack_train.generate(image, **pgd_params_train)


        pred = clf(images)
        if isinstance(pred,tuple):
            pred = pred[0]
#        pred = torch.sigmoid(pred)
        c_loss = criterion(pred,label)

        regularization_loss = 0
        for param in clf.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        loss = c_loss + lamda * regularization_loss

        loss.backward()
        optimizer.step()

        predict = torch.argmax(pred,1)
        total += label.size(0)
        correct += torch.eq(predict,label).sum().double().item()

        u.progress_bar(batchIdx, len(trainloader), 'loss:%.3f | Acc: %.3f%% (%d/%d)'
                        % (loss, 100.*float(correct)/total, correct, total))

    clf.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i , (image,label) in enumerate(testloader):
            image = image.cuda()
            label = label.cuda()
            pgd_params_test['y'] = label
            pgd_params_test['clip_max'] = torch.max(image)
            pgd_params_test['clip_min'] = torch.min(image)
            PGDAttack_test = ProjectGradientDescent(model = clf)
            adv_test = PGDAttack_test.generate(image, **pgd_params_test)
            pred = clf(adv_test)
            if isinstance(pred,tuple):
                pred = pred[0]
            predict = torch.argmax(pred,1)
            total += label.size(0)
            correct += torch.eq(predict,label).sum().double().item()
        acc = 100*(correct/total)
        print(acc)
        if acc > acc_small:
            torch.save(clf.state_dict(), os.path.join('/home/lrh/store/modelpath/adversarial_defense/Messidor','adversarial_training_MPAdvT.pth'))
            acc_small = acc
