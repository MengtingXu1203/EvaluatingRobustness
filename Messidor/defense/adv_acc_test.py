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



print('====>load numpy data')
NumpyDataPath = "/home/lrh/dataset/messidor/adv_test.npz"
Data = np.load(NumpyDataPath)
image = Data['image']
label = Data['label']
image = torch.from_numpy(image)
label = torch.from_numpy(label)
print('\ndone')

print("=====>loading pretrained model...")
clf = InceptionV3(num_classes=4)
clf = torch.nn.DataParallel(clf)
cudnn.benchmark = True
clf = clf.cuda()

model_file = "../model_path/messidor_inceptionv3.pkl"
clf.load_state_dict(torch.load(model_file))
clf.eval()
print("done")

total = 0
correct =0
for i in range(image.size(0)):
    print(i)
    images = image[i].unsqueeze(0).float()
    images = images.cuda()
    labels = label[i].long().cuda()
    pred = clf(images)
    if isinstance(pred,tuple):
        pred = pred[0]

    predict = torch.argmax(pred,1)
    total += images.size(0)
    correct += torch.eq(predict,labels).sum().double().item()
    print('\n test result: accuracy %.2f%%,(%d/%d)' %(100*(correct/total),correct,total))
