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
from matplotlib import pyplot as plt
from networks import InceptionV3
#import torchvision.transforms as torch_transforms
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
from net import UGenerator_Net

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
                    if isinstance(pretrained_label_float, tuple):
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
#                outputs_x = self.pretrained_clf(images)
                if isinstance(outputs_advx, tuple):
                    outputs_advx = outputs_advx[0]
#                if isinstance(outputs_x, tuple):
#                    outputs_x = outputs_x[0]
                loss = self.criterion(outputs_advx, self.y_target)
                loss.backward()
                self.optimizerG.step()

                ##output result
                adv_pred = torch.argmax(outputs_advx, 1)
#                adv_pred = adv_pred.type(torch.cuda.FloatTensor)
#                x_pred = torch.argmax(outputs_x, 1)
#                x_pred = x_pred.type(torch.cuda.FloatTensor)
                total += labels.size(0)
                correct += adv_pred.eq(labels).sum().item()

                u.progress_bar(batchIdx, len(trainLoader), 'loss:%.3f | Acc: %.3f%% (%d/%d)'
                            % (loss, 100.*float(correct)/total, correct, total))
#                print("epoch: %d, loss: %.3f | Acc: %.3f%%(%d/%d)" %(epoch,loss, 100.*float(correct)/total,correct,total))



            curAcc = 100.*float(correct)/total
            if minAcc > curAcc:
                minAcc = curAcc
                torch.save(self.attackModel.state_dict(), self.attackModelPath)
                print("\n epoch: %d, minAcc : %.4f" %(epoch,minAcc))

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
#        delta_im = nn.ConstantPad2d((0,-1,-1,0),0)(delta_im)
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
    clf = InceptionV3(num_classes=4)
    clf = torch.nn.DataParallel(clf)
    cudnn.benchmark = True
    clf = clf.cuda()
    model_file = "messidor_inceptionv3.pkl"
    clf.load_state_dict(torch.load(model_file))
    clf.eval()
    print("done")


    print("=======>load  dataset")
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
            #    transforms.CenterCrop((1500,1500)),
                transforms.Resize((299,299)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
    ])
    isTrain = True
    rootDir = "/home/lrh/dataset/messidor/train"
    testRootDir =  "/home/lrh/dataset/messidor/test_train"
    batchSize = 9
    testBatchSize = 8

    trainset = torchvision.datasets.ImageFolder(rootDir,transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batchSize,shuffle=True,drop_last=False,num_workers=4)
    testset = torchvision.datasets.ImageFolder(testRootDir,transform=test_transforms)
    testloader = torch.utils.data.DataLoader(testset,batch_size=testBatchSize,shuffle=True,drop_last=False,num_workers=4)
    print("done")

    params = {
            "attackModelPath" : None,
            "mag_in" : 3.0,
            "ord" : "inf",
            "epochNum" : 10,
            "criterion" : nn.CrossEntropyLoss(),
            "ncInput" : 3,
            "ncOutput" : 3,
            "mean" : mean,
            "std" : std,
            "MaxIter" : 100
        }
    print(params)
    saveModelPath = "adversarial_result/GAP/GAP_im_m3n10_1.pth"
    attackModel = UGenerator_Net(input_nc = params["ncInput"], output_nc = params["ncOutput"],ngf = 64)
    attackModel = attackModel.cuda()
#    attackModel = generators.define(input_nc = params["ncInput"], output_nc = params["ncOutput"],
#                                    ngf = 64, gen_type = 'unet', norm="batch", act="relu", gpu_ids = [0])


    if isTrain is True:
        print("===>Train")
        optimizerG = optim.Adam(attackModel.parameters(), lr = 2e-4, betas = (0.9, 0.999))
        params["optimizerG"] = optimizerG
        GAPAttack = GenerativeAdversarialPerturbations(clf, attackModel, **params)
        GAPAttack.train(trainloader, saveModelPath)
    else:
        print("===>Test")
        ## test
        params["attackModelPath"] = saveModelPath
        GAPAttack = GenerativeAdversarialPerturbations(clf, attackModel, **params)

        gt = torch.FloatTensor()
        pred = torch.FloatTensor()
        pred_advx = torch.FloatTensor()

        with torch.no_grad():

            for i, (images,labels) in enumerate(testloader):
                print(i)
#            for i, data in enumerate(testloader):
#                images, labels = data['image'], data['label']
                images, labels = images.cuda(), labels.cuda()
                adv_images = GAPAttack.generate(images)

                outputs_advx = clf(adv_images)
                if isinstance(outputs_advx,tuple):
                    outputs_advx = outputs_advx[0]
                outputs_advx = torch.sigmoid(outputs_advx).cpu()
                pred_advx = torch.cat((pred_advx,outputs_advx),0)


                outputs_x = clf(images)
                if isinstance(outputs_x,tuple):
                    outputs_x = outputs_x[0]
                outputs_x = torch.sigmoid(outputs_x).cpu()
                pred = torch.cat((pred,outputs_x),0)

                labels = labels.unsqueeze(1)
                one_hot_labels = torch.zeros(len(labels),4).scatter_(1,labels.cpu(),1)
                gt = torch.cat((gt,one_hot_labels),0)


        AUC_ROC = roc_auc_score(gt,pred_advx,average='micro')
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
        print ("\nArea under the ROC curve: " +str(AUC_ROC))

        correct_acc = 0
        correct_fr = 0
        correct_acc = torch.argmax(gt,1).eq(torch.argmax(pred_advx,1)).sum()
        correct_fr = torch.argmax(pred,1).eq(torch.argmax(pred_advx,1)).sum()
        total = len(gt)

        print("adv_ACC: %.8f" %(float(correct_acc)/total))
        print("FR: %.8f" %(1-float(correct_fr)/total))
