import libadver
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
from networks import InceptionV3
from torchvision.transforms import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn as cudnn
from data import load_data

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


mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
test_transforms = transforms.Compose([
        #    transforms.CenterCrop((1500,1500)),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
])

pgd_params = {
            'ord': np.inf,
            'y': None,
            'eps': 100.0 / 255,
            'eps_iter': 1.0 / 255,
            'nb_iter': 4,
            'rand_init': True,
            'rand_minmax': 50.0 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True,
            'y_target':None,
        }

print("=====>loading pretrained model...")
clf = InceptionV3(num_classes=4)
clf = torch.nn.DataParallel(clf)
cudnn.benchmark = True
clf = clf.cuda()

model_file = "messidor_inceptionv3.pkl"
clf.load_state_dict(torch.load(model_file))
clf.eval()
print("done")

PGDAttack = ProjectGradientDescent(model = clf)

gt = torch.FloatTensor()
#pred = torch.FloatTensor().numpy()
#pred = np.zeros([0,4])
pred = torch.FloatTensor()
pred_advx = torch.FloatTensor()

isAttackDataset = False
if isAttackDataset is True:
    testBatchSize = 8
    #modelFile = "/home/lrh/git/libadver/examples/Messidor/multi-classification/messidor_inceptionv3.pkl"
    #modelFile = "messidor_inceptionv3.pkl"
    testRootDir = "/home/lrh/dataset/messidor/test_train"

    print("=====>loading data")

    testset = torchvision.datasets.ImageFolder(testRootDir,transform=test_transforms)
    testloader = torch.utils.data.DataLoader(testset,batch_size=testBatchSize,shuffle=True,drop_last=False,num_workers=4)
    print("done")

    with torch.no_grad():
        for i, (images_test,labels_test) in enumerate(testloader):
            print(i)
        #    images_test, labels_test = data['image'], data['label']
            images_test = images_test.cuda()
            labels_test = labels_test.cuda()

            pgd_params['y'] = labels_test
            pgd_params['clip_max'] = torch.max(images_test)
            pgd_params['clip_min'] = torch.min(images_test)
            pgd_params['eps'] = 2.0 /255
            pgd_params['nb_iter'] = 1
            adv_x = PGDAttack.generate(images_test, **pgd_params)

            torchvision.utils.save_image(adv_x[0], 'pgd_image_show'+'/adv{}.jpg'.format(i), nrow = 1 ,normalize = True)

            outputs_x = clf(images_test)
            if isinstance(outputs_x,tuple):
                outputs_x = outputs_x[0]

            outputs_x = torch.sigmoid(outputs_x).cpu()
            pred = torch.cat((pred,outputs_x),0)
    #        pred = np.concatenate((pred,outputs_x),0)
            # print(pred.type())
            #
            outputs_advx = clf(adv_x)
            if isinstance(outputs_advx,tuple):
                outputs_advx = outputs_advx[0]

            outputs_advx = torch.sigmoid(outputs_advx).cpu()
            pred_advx = torch.cat((pred_advx,outputs_advx),0)


            labels_test = labels_test.unsqueeze(1)
            one_hot_labels = torch.zeros(len(labels_test),4).scatter_(1,labels_test.cpu(),1)
            gt = torch.cat((gt,one_hot_labels),0)
    #        print(gt.type())


    AUC_ROC = roc_auc_score(gt,pred_advx,average='micro')
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print ("\nArea under the ROC curve: " +str(AUC_ROC))

    correct_acc = 0
    correct_fr = 0
    print(torch.argmax(gt,1))
    print(torch.argmax(gt,1).shape)
    print(torch.argmax(pred_advx,1))
    correct_acc = torch.argmax(gt,1).eq(torch.argmax(pred_advx,1)).sum()
    correct_fr =  torch.argmax(pred,1).eq(torch.argmax(pred_advx,1)).sum()
    total = len(gt)

    print("adv_ACC: %.8f" %(float(correct_acc)/total))
    print("FR: %.8f" %(1-float(correct_fr)/total))

else:

    t_transforms = transforms.Compose([
                transforms.CenterCrop((1000,1000)),
                transforms.Resize((299,299)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
    ])
    file_dir = "/home/lrh/dataset/messidor/test_train/0"
    file_names = os.listdir(file_dir)
    file_names.sort()
    file_paths = [os.path.join(file_dir,file_name) for file_name in file_names]
    images = load_data(file_paths,t_transforms)
    images = images[0:2]
    print(images.shape)

    pgd_params['y'] = torch.LongTensor(len(images)).zero_().cuda()
#    pgd_params['y_target'] = torch.LongTensor(len(images)).zero_().cuda()+1
    pgd_params['clip_min'] = torch.min(images)
    pgd_params['clip_max'] = torch.max(images)
    pgd_params['eps'] = 4.0 /255
    pgd_params['nb_iter'] = 4
    print(pgd_params['clip_min'])
    print(pgd_params['clip_max'])
#    print(pgd_params['y'])

    adv_x = PGDAttack.generate(images, **pgd_params)

    outputs_x = clf(images)
#    print(outputs_x.shape)
    if isinstance(outputs_x,tuple):
        outputs_x = outputs_x[0]
    pred_x = torch.softmax(outputs_x,1)
    print(pred_x)


    outputs_advx = clf(adv_x)
    if isinstance(outputs_advx,tuple):
        outputs_advx = outputs_advx[0]
#    outputs_advx = torch.softmax(outputs_advx,1)
    pred_advx = torch.softmax(outputs_advx,1)
#    print(outputs_advx)
    print(pred_advx)

    adv_images = adv_x.cpu()
    images = images.cpu()
    delta_ims = (adv_images - images)
    print(torch.max(delta_ims))

    for i in range(len(images)):
        adv_image = adv_images[i]
        delta_im = delta_ims[i]
        image = images[i]

        adv_image_np = libadver.visutils.recreate_image(adv_image,mean,std)
        libadver.visutils.save_image(adv_image_np,"adversarial_result/PGD/0/adv_img_%d.png" %i)

        image_np = libadver.visutils.recreate_image(image,mean,std)
        libadver.visutils.save_image(image_np,"adversarial_result/PGD/0/ori_img_%d.png" %i)

        delta_im = delta_ims[i].data.numpy()
        libadver.visutils.save_gradient_images(delta_im,"adversarial_result/PGD/0/delta_im_%d.png" %i)
