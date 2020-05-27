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

            gt = torch.FloatTensor()
            pred = torch.FloatTensor()
            pred_advx = torch.FloatTensor()

            for batchIdx, data in enumerate(trainLoader):
                # if batchIdx > self.MaxIter:
                #     break
                ## for IPIM-2019 paper.
                if isinstance(data, dict):
                    images, labels = data['image'], data['label']
                else:
                    images, labels = data
                images, labels = images.cuda(), labels.cuda()

                #non-targeted
                if self.targeted is False:

                    pretrained_label_float = self.pretrained_clf(images)
                    self.y_target = pretrained_label_float < 0.5
                    self.y_target = self.y_target.float()
                #target
                else:
                    target_label = torch.FloatTensor(images.size(0))
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

                loss = 0
                for i in range(14):

                    loss_iter = self.criterion(outputs_advx[:,i], self.y_target[:,i])
                    loss = loss + loss_iter
 #               loss = self.criterion(outputs_advx, self.y_target)
                loss = loss/14.0
#                print(loss)
                loss.backward()
                self.optimizerG.step()

                ##output result
                adv_pred = outputs_advx.type(torch.cuda.FloatTensor)
                x_pred = outputs_x.type(torch.cuda.FloatTensor)
                true_labels = labels.type(torch.cuda.FloatTensor)
                gt = torch.cat((gt,true_labels.detach().cpu()),0)
                pred = torch.cat((pred,x_pred.detach().cpu()),0)
                pred_advx = torch.cat((pred_advx,adv_pred.detach().cpu()),0)


                ACCs_iter=[]
                for i in range(14):
                    predictLabels_advx = pred_advx[:, i] > 0.5
                    predictLabels_advx = predictLabels_advx.float()
                    acc = float((gt[:, i] == predictLabels_advx).sum() )/ gt.shape[0]
                    ACCs_iter.append(acc)
                ACCs_avg = np.array(ACCs_iter).mean()

                u.progress_bar(batchIdx, len(trainLoader), 'loss:%.3f | Acc: %.3f%%'
                            % (loss, 100.*ACCs_avg))
#                print("epoch: %d, loss: %.3f | Acc: %.3f" %(epoch,loss, 100.*ACCs_avg))

            curAcc = 100.*ACCs_avg
            if minAcc > curAcc:
                minAcc = curAcc
                torch.save(self.attackModel.state_dict(), self.attackModelPath)
                print("\n epoch: %d, minAcc : %.4f" %(epoch,minAcc))
                AUROCs = []
                ACCs = []
                ACCs_Fr = []
                for i in range(14):
#                    print(np.unique(gt[:, i]))
                    AUROCs.append(roc_auc_score(gt[:, i], pred_advx[:, i]))

                    predictLabels_x = pred[:, i] > 0.5
                    predictLabels_x = predictLabels_x.float()

                    predictLabels_advx = pred_advx[:, i] > 0.5
                    predictLabels_advx = predictLabels_advx.float()

                    acc = float((gt[:, i] == predictLabels_advx).sum() )/ gt.shape[0]
                    ACCs.append(acc)
                    acc_fr = float((predictLabels_x == predictLabels_advx).sum() )/ gt.shape[0]
                    ACCs_Fr.append(acc_fr)

                AUROC_avg = np.array(AUROCs).mean()
                ACCs_avg = np.array(ACCs).mean()
                ACCs_Fr_avg = np.array(ACCs_Fr).mean()

                print('The average ACC is %.3f' %(ACCs_avg))
                print('The average AUROC is %.3f' %(AUROC_avg))
                print('The average FR is %.3f' %(1-ACCs_Fr_avg))


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
                    MaxIter = 300,
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

    isTrain = True
    isResult = True

    print("======>load pretrained models")

    DATA_DIR = '/home/lrh/dataset/ChestXray-NIHCC/images_v1_small'
    modelFile = '/home/lrh/git/libadver/examples/ChestXNet/model.pth.tar'
    testTXTFile = '/home/lrh/git/CheXNet/ChestX-ray14/labels/test.txt'
    trainTXTFile = '/home/lrh/git/CheXNet/ChestX-ray14/labels/train.txt'

    N_CLASSES = 14
    cudnn.benchmark = True
    # initialize and load the model
    net = DenseNet121(N_CLASSES).cuda()
    pretrained_clf = torch.nn.DataParallel(net).cuda()
    pretrained_clf.eval()
    if os.path.isfile(modelFile):
        print("=> loading checkpoint")
        checkpoint = torch.load(modelFile)

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        pretrained_clf.load_state_dict(state_dict)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    print("=======>load dataset")

    BATCH_SIZE = 6
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    print("=>load train dataset")
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=trainTXTFile,
                                    transform = test_transform
                                    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)
    print("=>load test dataset")
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=testTXTFile,
                                    transform = test_transform
                                    )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    params = {
            "attackModelPath" : None,
            "mag_in" : 15.0,
            "ord" : "inf",
            "epochNum" : 8,
            "criterion" : nn.BCELoss(),
            "ncInput" : 3,
            "ncOutput" : 3,
            "mean" : mean,
            "std" : std,
            "MaxIter" : 100
        }
    print(params)
    saveModelPath = "adversarial_result/GAP/GAP_im_m15n8_2.pth"
    attackModel = generators.define(input_nc = params["ncInput"], output_nc = params["ncOutput"],
                                    ngf = 64, gen_type = "unet", norm="batch", act="relu", gpu_ids = [0])

    if isTrain == True:
        print("===>Train")
        optimizerG = optim.Adam(attackModel.parameters(), lr = 2e-4, betas = (0.9, 0.999))
        params["optimizerG"] = optimizerG
        GAPAttack = GenerativeAdversarialPerturbations(pretrained_clf, attackModel, **params)
        GAPAttack.train(train_loader, saveModelPath)
    if isResult == True:
        params["attackModelPath"] = saveModelPath
        GAPAttack = GenerativeAdversarialPerturbations(pretrained_clf, attackModel, **params)

        gt = torch.FloatTensor()
        pred = torch.FloatTensor()
        pred_advx = torch.FloatTensor()

        print("===>TestResult")
        for batchIdx, (images, labels) in enumerate(test_loader):
            print("[%d|%d]" %(batchIdx,len(test_loader)))
            images, labels = images.cuda(), labels.cuda()
            adv_x = GAPAttack.generate(images)
            torchvision.utils.save_image(adv_x, 'gap_image_show'+'/adv{}.jpg'.format(batchIdx), nrow = 50 ,normalize = True)
            x_pred = pretrained_clf(images)
            adv_pred = pretrained_clf(adv_x)
            gt = torch.cat((gt, labels.detach().cpu()), 0)
            pred = torch.cat((pred, x_pred.detach().cpu()), 0)
            pred_advx = torch.cat((pred_advx, adv_pred.detach().cpu()),0)

        AUROCs = []
        ACCs = []
        ACCs_Fr = []
        for i in range(14):
            AUROCs.append(roc_auc_score(gt[:, i], pred_advx[:, i]))

            predictLabels_x = pred[:, i] > 0.5
            predictLabels_x = predictLabels_x.float()

            predictLabels_advx = pred_advx[:, i] > 0.5
            predictLabels_advx = predictLabels_advx.float()

            acc = float((gt[:, i] == predictLabels_advx).sum() )/ gt.shape[0]
            ACCs.append(acc)
            acc_fr = float((predictLabels_x == predictLabels_advx).sum() )/ gt.shape[0]
            ACCs_Fr.append(acc_fr)

        AUROC_avg = np.array(AUROCs).mean()
        ACCs_avg = np.array(ACCs).mean()
        ACCs_Fr_avg = np.array(ACCs_Fr).mean()

        print('The average ACC is %.3f' %(ACCs_avg))
        print('The average AUROC is %.3f' %(AUROC_avg))
        print('The average FR is %.3f' %(1-ACCs_Fr_avg))
