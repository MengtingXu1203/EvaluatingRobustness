import torch.nn as nn
import sys
sys.path.append("../")
import libadver.utils as utils

import libadver.models.generators as generators
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import numpy as np
import warnings
import os

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

        for epoch in range(self.epochNum):
            print("\n Epoch : %d" %epoch)
            total = 0
            correct = 0
            minAcc = 100


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
                outputs = self.pretrained_clf(recons)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                loss = self.criterion(outputs, self.y_target)
                loss.backward()
                self.optimizerG.step()

                ##output result
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.type(torch.cuda.FloatTensor)
                total += labels.size(0)
                true_labels = labels.type(torch.cuda.FloatTensor)
                correct += predicted.eq(true_labels).sum().item()

                utils.progress_bar(batchIdx, len(trainLoader), 'loss:%.3f | Acc: %.3f%% (%d/%d)'
                            % (loss, 100.*float(correct)/total, correct, total))
            curAcc = 100.*float(correct)/total
            if minAcc > curAcc:
                minAcc = curAcc
                torch.save(self.attackModel.state_dict(), self.attackModelPath)
                print("\n minAcc : %.4f" %minAcc)

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
    isTrain = False

    ## Load Dataset
    print("===> Load validation dataset")
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    rootDir = "/store/dataset/imagenet/ILSVRC2012_img_val"
    labelDir = "/store/dataset/imagenet/caffe_ilsvrc12/val.txt"
    valdata = utils.valDataset(root_dir = rootDir, labelDir = labelDir, transforms = val_transform)
    valdataloader = torch.utils.data.DataLoader(valdata, batch_size=16, shuffle=True, drop_last=False)
    print(len(valdataloader))

    ## Load pretrained model
    print("====> Load pretrained models")
    vgg16 = models.vgg16_bn(pretrained = True)
    vgg16 = vgg16.cuda()
    vgg16 = vgg16.eval()


    import torch.nn as nn
    import torch.optim as optim

    params = {
        "attackModelPath" : None,
        "mag_in" : 7.0,
        "ord" : "inf",
        "epochNum" : 3,
        "criterion" : nn.CrossEntropyLoss(),
        "ncInput" : 3,
        "ncOutput" : 3,
        "mean" : mean,
        "std" : std,
        "MaxIter" : 100
    }
    print(params)
    saveModelPath = "./GAP_im_test.pth"
    attackModel = generators.define(input_nc = params["ncInput"], output_nc = params["ncOutput"],
                                    ngf = 64, gen_type = "unet", norm="batch", act="relu", gpu_ids = [0])
    #"optimizerG" : optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)),


    if isTrain is True:
        print("===>Train")
        optimizerG = optim.Adam(attackModel.parameters(), lr = 2e-4, betas = (0.5, 0.999))
        params["optimizerG"] = optimizerG
        GAPAttack = GenerativeAdversarialPerturbations(vgg16, attackModel, **params)
        GAPAttack.train(valdataloader, saveModelPath)
    else:
        print("===>Test")
        ## test
        params["attackModelPath"] = saveModelPath
        GAPAttack = GenerativeAdversarialPerturbations(vgg16, attackModel, **params)
        correct = 0
        total = 0
        for i, (images, targets) in enumerate(valdataloader):
            images, targets = images.cuda(), targets.cuda()
            adv_images = GAPAttack.generate(images)
            predicted = vgg16(adv_images)
            predicted_labels = torch.argmax(predicted,1)
            #print(predicted_labels)
            correct += torch.sum(predicted_labels.eq(targets))
            #print(targets)
            total += images.shape[0]
            print("ACC:%.3f | %d,%d" %(100.0*float(correct) / total, correct, total))
            #break
