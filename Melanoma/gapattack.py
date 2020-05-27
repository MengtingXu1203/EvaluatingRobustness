import torch.nn as nn
import sys
sys.path.append("../")
import libadver.utils as u
import libadver.models.generators as generators
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import warnings
import os
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from networks import AttnVGG, VGG
from data import preprocess_data_2016, preprocess_data_2017, ISIC, load_data
from utilities import *
from transforms import *
import torchvision.transforms as torch_transforms
import torch.optim as optim
import torchvision

class GenerativeAdversarialPerturbations():
    """
    This class implements CVPR2018 paper Generative Adversarial Perturbations
    Only on Image-dependent Perturbations
    Paper Link: (Poursaeed et al. 2018): https://arxiv.org/abs/1712.02328
    Official implements (github): https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations
    Date : 2019.5.13
    """
    def __init__(self, model, attackModel, **kwargs):
        if not isinstance(model, nn.Module):
            raise TypeError("The model argument should be the instance of"
                            "torch.nn.Module")
        self.pretrained_clf = model
        self.attackModel = attackModel
        self.parse_params(**kwargs)

    def train(self, trainLoader, saveModelPath):
        """
        attackModel output [-1...1], the last layer activated with tanh function
        """
        self.attackModelPath = saveModelPath
        self.pretrained_clf.eval()
        self.attackModel.train()

        minAcc = 100
        for epoch in range(self.epochNum):
            print("\n Epoch : %d" %epoch)
            total = 0
            correct = 0
            correct_fr = 0

            gt = torch.FloatTensor()
            pred = torch.FloatTensor()
            pred_advx = torch.FloatTensor()

            for batchIdx, data in enumerate(trainLoader):
                if batchIdx > self.MaxIter:
                    break
                ## for IPIM-2019 paper.
                if isinstance(data, dict):
                    images, labels = data['image'], data['label']
                else:
                    images, labels = data
                images, labels = images.cuda(), labels.cuda()

                #non-targeted
                if self.targeted is False:

                    pretrained_label_float = self.pretrained_clf(images)
                    if isinstance(pretrained_label_float, list):
                        pretrained_label_float = pretrained_label_float[0]
                    _, self.y_target = torch.min(pretrained_label_float, 1)
                #target
                else:
                    target_label = torch.LongTensor(images.size(0))
                    target_label.fill_(self.y_target)
                    self.y_target = target_label


                deltaIm = self.attackModel(images)
                deltaIm = self._normalize_and_scale(deltaIm, self.mean, self.std)
                self.attackModel.zero_grad()
                recons = torch.add(images, deltaIm)
                # do clamping per channel
                for cii in range(self.ncInput):
                    recons.data[:,cii,:,:] = recons.data[:,cii,:,:].clamp(images.data[:,cii,:,:].min(), images.data[:,cii,:,:].max())
                outputs_advx = self.pretrained_clf(recons)
                outputs_x = self.pretrained_clf(images)
                if isinstance(outputs_advx, list):
                    outputs_advx = outputs_advx[0]
                if isinstance(outputs_x, list):
                    outputs_x = outputs_x[0]
                loss = self.criterion(outputs_advx, self.y_target)
                loss.backward()
                self.optimizerG.step()

                ##output result
                # _, adv_pred = torch.max(outputs_advx, 1)
                # adv_pred = adv_pred.type(torch.cuda.FloatTensor)
                #
                # _, x_pred = torch.max(outputs_x, 1)
                # x_pred = x_pred.type(torch.cuda.FloatTensor)
                adv_pred = torch.argmax(outputs_advx, 1)
                adv_pred = adv_pred.type(torch.cuda.FloatTensor)

                x_pred = torch.argmax(outputs_x, 1)
                x_pred = x_pred.type(torch.cuda.FloatTensor)

                total += labels.size(0)

                true_labels = labels.type(torch.cuda.FloatTensor)
                gt = torch.cat((gt,true_labels.detach().cpu()),0)
                pred = torch.cat((pred,x_pred.detach().cpu()),0)
                pred_advx = torch.cat((pred_advx,adv_pred.detach().cpu()),0)

                correct += adv_pred.eq(true_labels).sum().item()

                u.progress_bar(batchIdx, len(trainLoader), 'loss:%.3f | Acc: %.3f%% (%d/%d)'
                            % (loss, 100.*float(correct)/total, correct, total))
#                print("epoch: %d, loss: %.3f | Acc: %.3f%%(%d/%d)" %(epoch,loss, 100.*float(correct)/total,correct,total))



            curAcc = 100.*float(correct)/total
            if minAcc > curAcc:
                minAcc = curAcc
                torch.save(self.attackModel.state_dict(), self.attackModelPath)
                print("\n epoch: %d, minAcc : %.4f" %(epoch,minAcc))

                fpr, tpr, thresholds = roc_curve(gt,pred_advx)
                AUC_ROC = roc_auc_score(gt,pred_advx)
                # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
                print ("\nArea under the ROC curve: " +str(AUC_ROC))
                ROC_curve =plt.figure()
                plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
                plt.title('ROC curve')
                plt.xlabel("FPR (False Positive Rate)")
                plt.ylabel("TPR (True Positive Rate)")
                plt.legend(loc="lower right")
                plt.savefig("gap_image_show/ROC%d.png" %epoch)

                correct_acc = 0
                correct_fr = 0
                correct_acc = correct_acc+gt.eq(pred_advx).sum()
                correct_fr = correct_fr+pred.eq(pred_advx).sum()
                alltotal = len(gt)

                print("adv_ACC: %.8f" %(float(correct_acc)/total))
                print("FR: %.8f" %(1-float(correct_fr)/total))

    def generate(self, inputs):
        """
        Generate adversarial images

        Generate function parameters:
        :param inputs : input images, with shape [:, ncInput, height, width]
        """
        ## assure that attack model has been trained before
        if self.attackModelPath is None:
            raise ValueError("Training function is should be invoked"
                        "before generating")
        x = inputs
        self.attackModel.load_state_dict(torch.load(self.attackModelPath))
        deltaIm = self.attackModel(x)

        deltaIm = self._normalize_and_scale(deltaIm, self.mean, self.std)

        recons = torch.add(x, deltaIm)
        for cii in range(self.ncInput):
            recons.data[:,cii,:,:] = recons.data[:,cii,:,:].clamp(inputs.data[:,cii,:,:].min(), inputs.data[:,cii,:,:].max())

        #post_l_inf = (recons.data - inputs[0:recons.size(0)].data).abs().max() * 255.0
        #print("Specified l_inf: ", self.mag_in, ", maximum l_inf of generated perturbations: ", post_l_inf)
        return recons



    def parse_params(self,
                    attackModelPath = None,
                    ord = "inf",
                    mag_in = 15.0,
                    epochNum = 200,
                    criterion = None,
                    optimizerG = None,
                    ncInput = 3,
                    ncOutput = 3,
                    mean = [0.5, 0.5, 0.5],
                    std = [0.5, 0.5, 0.5],
                    y_target = None,
                    MaxIter = 100,
                    **kwargs):
        self.attackModelPath = attackModelPath
        self.epochNum = epochNum
        self.ord = ord
        self.mag_in = mag_in
        self.ncInput = ncInput
        self.ncOutput = ncOutput
        self.MaxIter = MaxIter
        self.mean = mean
        self.std = std


        self.criterion = criterion
        self.optimizerG = optimizerG

        if y_target is None:
            self.targeted = False
        else:
            self.targeted = True

        if self.attackModelPath is not None and not os.path.exists(self.attackModelPath):
            raise FileNotFoundError("%s file is not exists" %self.attackModelPath)

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in ["inf", "1", "2"]:
            raise ValueError("Norm order must be either \" inf \", \" 1 \", or \" 2 \".")

        if not isinstance(self.ncInput, int) or not isinstance(self.ncOutput, int):
            raise ValueError("Dim of image should be integer")
        if self.ncInput != self.ncOutput:
            warnings.warn("In general setting, input dim should equal to output")
        if len(kwargs.keys()) > 0:
            #print(kwargs)
            warnings.warn("kwargs is unused and will be removed on or after "
                          "2019-05-13.")



    def _normalize_and_scale(self, delta_im, mean, std):
        """
        Normalize and scale the generated perturbations with norm mag_in
        fixed norm type "inf"
        """
        delta_im.data += 1 # now 0..2
        delta_im.data *= 0.5 # now 0..1

        # normalize image color channels
        for c in range(self.ncInput):
            delta_im.data[:,c,:,:] = (delta_im.data[:,c,:,:] - mean[c]) / std[c]

        # threshold each channel of each image in deltaIm according to inf norm
        # do on a per image basis as the inf norm of each image could be different
        bs = delta_im.size(0)
        for i in range(bs):
            # do per channel l_inf normalization
            for ci in range(self.ncInput):
                l_inf_channel = delta_im[i,ci,:,:].detach().cpu().abs().max()
                mag_in_scaled_c = self.mag_in/(255.0*std[ci])
                delta_im[i,ci,:,:].data *= torch.tensor(np.minimum(1.0, mag_in_scaled_c / l_inf_channel)).float().cuda()

        return delta_im

if __name__ == "__main__":
    print("======>load pretrained models")
    net = AttnVGG(num_classes=2, attention=True, normalize_attn=True, vis = False)
    # net = VGG(num_classes=2, gap=False)
    modelFile = "/home/lrh/git/libadver/examples/IPIM-AttnModel/models/checkpoint.pth"
    testCSVFile = "/home/lrh/git/libadver/examples/IPIM-AttnModel/test.csv"
    trainCSVFile = "/home/lrh/git/libadver/examples/IPIM-AttnModel/train.csv"

    checkpoint = torch.load(modelFile)
    net.load_state_dict(checkpoint['state_dict'])
    pretrained_clf = nn.DataParallel(net).cuda()
    pretrained_clf.eval()

    print("=======>load ISIC2016 dataset")
    mean = (0.7012, 0.5517, 0.4875)
    std = (0.0942, 0.1331, 0.1521)
    normalize = Normalize(mean, std)
    transform = torch_transforms.Compose([
             RatioCenterCrop(0.8),
             Resize((256,256)),
             CenterCrop((224,224)),
             ToTensor(),
             normalize
        ])
    testset = ISIC(csv_file=testCSVFile, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=4)

    trainset = ISIC(csv_file=trainCSVFile, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)
    print(len(trainloader))
    isTrain = False

    params = {
            "attackModelPath" : None,
            "mag_in" : 11.0,
            "ord" : "inf",
            "epochNum" : 13,
            "criterion" : nn.CrossEntropyLoss(),
            "ncInput" : 3,
            "ncOutput" : 3,
            "mean" : mean,
            "std" : std,
            "MaxIter" : 100
        }
    print(params)
    saveModelPath = "adversarial_result/GAP/GAP_im_m11n13_1.pth"
    attackModel = generators.define(input_nc = params["ncInput"], output_nc = params["ncOutput"],
                                    ngf = 64, gen_type = "unet", norm="batch", act="relu", gpu_ids = [0])


    if isTrain is True:
        print("===>Train")
        optimizerG = optim.Adam(attackModel.parameters(), lr = 2e-4, betas = (0.9, 0.999))
        params["optimizerG"] = optimizerG
        GAPAttack = GenerativeAdversarialPerturbations(pretrained_clf, attackModel, **params)
        GAPAttack.train(trainloader, saveModelPath)
    else:
        print("===>Test")
        ## test
        params["attackModelPath"] = saveModelPath
        GAPAttack = GenerativeAdversarialPerturbations(pretrained_clf, attackModel, **params)

        gt = torch.FloatTensor()
        pred = torch.FloatTensor()
        pred_advx = torch.FloatTensor()

        for i, data in enumerate(testloader):
            images, labels = data['image'], data['label']
            images, labels = images.cuda(), labels.cuda()
            adv_images = GAPAttack.generate(images)

            torchvision.utils.save_image(adv_images, 'gap_image_show'+'/adv{}.jpg'.format(i), nrow = 50 ,normalize = True)


            #adv_images = images
            outputs_advx,_,_ = pretrained_clf(adv_images)
            adv_pred = torch.argmax(outputs_advx,1).float()

            outputs_x,_,_ = pretrained_clf(images)
            x_pred = torch.argmax(outputs_x,1).float()

            labels = labels.float()

            gt = torch.cat((gt, labels.detach().cpu()), 0)
            pred = torch.cat((pred, x_pred.detach().cpu()), 0)
            pred_advx = torch.cat((pred_advx, adv_pred.detach().cpu()),0)


        fpr, tpr, thresholds = roc_curve(gt,pred_advx)

        AUC_ROC = roc_auc_score(gt,pred_advx)
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
        print ("\nArea under the ROC curve: " +str(AUC_ROC))
        ROC_curve =plt.figure()
        plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
        plt.title('ROC curve')
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("TPR (True Positive Rate)")
        plt.legend(loc="lower right")
        plt.savefig("pgd_image_show/ROC.png")

        correct_acc = 0
        correct_fr = 0
        correct_acc = correct_acc+gt.eq(pred_advx).sum()
        correct_fr = correct_fr+pred.eq(pred_advx).sum()
        total = len(gt)

        print("adv_ACC: %.8f" %(float(correct_acc)/total))
        print("FR: %.8f" %(1-float(correct_fr)/total))
