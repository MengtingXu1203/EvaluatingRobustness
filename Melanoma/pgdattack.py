import libadver
import os
import csv
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import torchvision.transforms as torch_transforms
from networks import AttnVGG, VGG
from loss import FocalLoss
from data import preprocess_data_2016, preprocess_data_2017, ISIC, load_data
from utilities import *
from transforms import *
import libadver.attack as attack
from torchvision.transforms import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

pgd_params = {
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

def main():
    modelFile = "/home/lrh/store/modelpath/adversarial_defense/Dermothsis/adv_training.pth"
    testCSVFile = "/home/lrh/git/libadver/examples/IPIM-AttnModel/test.csv"
    print("======>load pretrained models")
    net = AttnVGG(num_classes=2, attention=True, normalize_attn=False, vis = False)
    # net = VGG(num_classes=2, gap=False)
    checkpoint = torch.load(modelFile)
    net.load_state_dict(checkpoint['state_dict'])
    pretrained_clf = nn.DataParallel(net).cuda()
    pretrained_clf.eval()

    print("=======>load ISIC2016 dataset")
    normalize = Normalize((0.7012, 0.5517, 0.4875), (0.0942, 0.1331, 0.1521))
    transform_test = torch_transforms.Compose([
             RatioCenterCrop(0.8),
             Resize((256,256)),
             CenterCrop((224,224)),
             ToTensor(),
             normalize
        ])
    testset = ISIC(csv_file=testCSVFile, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)

    PGDAttack = attack.ProjectGradientDescent(model = pretrained_clf)

    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    pred_advx = torch.FloatTensor()

    for i, data in enumerate(testloader,0):
        print(i)
        images_test, labels_test = data['image'], data['label']
        images_test = images_test.cuda()
        labels_test = labels_test.cuda()

        pgd_params['y'] = labels_test
        pgd_params['clip_max'] = torch.max(images_test)
        pgd_params['clip_min'] = torch.min(images_test)
        adv_x = PGDAttack.generate(images_test, **pgd_params)
#        torchvision.utils.save_image(adv_x, 'pgd_image_show'+'/adv{}.jpg'.format(i), nrow = 50 ,normalize = True)

        outputs_x,_,_= pretrained_clf(images_test)
        x_pred = torch.argmax(outputs_x,dim = 1).float()
        outputs_advx,_,_ = pretrained_clf(adv_x)
        adv_pred = torch.argmax(outputs_advx,dim = 1).float()
        labels_test = labels_test.float()
        gt = torch.cat((gt, labels_test.detach().cpu()), 0)
        pred = torch.cat((pred, x_pred.detach().cpu()), 0)
        pred_advx = torch.cat((pred_advx, adv_pred.detach().cpu()),0)

    fpr, tpr, thresholds = roc_curve(gt,pred_advx)

    AUC_ROC = roc_auc_score(gt,pred_advx)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print ("\nArea under the ROC curve: " +str(AUC_ROC))
    # ROC_curve =plt.figure()
    # plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    # plt.title('ROC curve')
    # plt.xlabel("FPR (False Positive Rate)")
    # plt.ylabel("TPR (True Positive Rate)")
    # plt.legend(loc="lower right")
    # plt.savefig("pgd_image_show/ROC.png")

    correct_acc = 0
    correct_fr = 0
    correct_acc = correct_acc+gt.eq(pred_advx).sum()
    correct_fr = correct_fr+pred.eq(pred_advx).sum()
    total = len(gt)

    print("adv_ACC: %.8f" %(float(correct_acc)/total))
    print("FR: %.8f" %(1-float(correct_fr)/total))

if __name__ == "__main__":
    main()
