import csv
import os
import os.path
from PIL import Image
import glob
import numpy as np
import torch
import torch.utils.data as udata
from transforms import *
from torchvision.transforms import transforms

def preprocess_data_2016(root_dir):
    print('pre-processing data ...\n')
    # training data
    benign    = glob.glob(os.path.join(root_dir, 'Train', 'benign', '*.jpg')); benign.sort()
    malignant = glob.glob(os.path.join(root_dir, 'Train', 'malignant', '*.jpg')); malignant.sort()
    benign_seg    = glob.glob(os.path.join(root_dir, 'Train_Lesion', 'benign', '*.png')); benign_seg.sort()
    malignant_seg = glob.glob(os.path.join(root_dir, 'Train_Lesion', 'malignant', '*.png')); malignant_seg.sort()
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(benign)):
            filename = benign[k]
            filename_seg = benign_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(malignant)):
            filename = malignant[k]
            filename_seg = malignant_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
    # training data oversample
    benign    = glob.glob(os.path.join(root_dir, 'Train', 'benign', '*.jpg')); benign.sort()
    malignant = glob.glob(os.path.join(root_dir, 'Train', 'malignant', '*.jpg')); malignant.sort()
    benign_seg    = glob.glob(os.path.join(root_dir, 'Train_Lesion', 'benign', '*.png')); benign_seg.sort()
    malignant_seg = glob.glob(os.path.join(root_dir, 'Train_Lesion', 'malignant', '*.png')); malignant_seg.sort()
    with open('train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(benign)):
            filename = benign[k]
            filename_seg = benign_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for i in range(4):
            for k in range(len(malignant)):
                filename = malignant[k]
                filename_seg = malignant_seg[k]
                writer.writerow([filename] + [filename_seg] + ['1'])
    # val data
    benign    = glob.glob(os.path.join(root_dir, 'Val', 'benign', '*.jpg')); benign.sort()
    malignant = glob.glob(os.path.join(root_dir, 'Val', 'malignant', '*.jpg')); malignant.sort()
    #### segmentation of val data is not used! ######
    benign_seg    = glob.glob(os.path.join(root_dir, 'Val', 'benign', '*.jpg')); benign_seg.sort()
    malignant_seg = glob.glob(os.path.join(root_dir, 'Val', 'malignant', '*.jpg')); malignant_seg.sort()
    with open('val.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(benign)):
            filename = benign[k]
            filename_seg = benign_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(malignant)):
            filename = malignant[k]
            filename_seg = malignant_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
    # test data
    benign    = glob.glob(os.path.join(root_dir, 'Test', 'benign', '*.jpg')); benign.sort()
    malignant = glob.glob(os.path.join(root_dir, 'Test', 'malignant', '*.jpg')); malignant.sort()
    benign_seg    = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'benign', '*.png')); benign_seg.sort()
    malignant_seg = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'malignant', '*.png')); malignant_seg.sort()
    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(benign)):
            filename = benign[k]
            filename_seg = benign_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(malignant)):
            filename = malignant[k]
            filename_seg = malignant_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])

def preprocess_data_2017(root_dir, seg_dir='Train_Lesion'):
    print('pre-processing data ...\n')
    # training data
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    melanoma_seg = glob.glob(os.path.join(root_dir, seg_dir, 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, seg_dir, 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, seg_dir, 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # training data oversample
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    melanoma_seg = glob.glob(os.path.join(root_dir, seg_dir, 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, seg_dir, 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, seg_dir, 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open('train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(4):
            for k in range(len(melanoma)):
                filename = melanoma[k]
                filename_seg = melanoma_seg[k]
                writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # val data
    melanoma = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    #### segmentation of val data is not used! ######
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk_seg.sort()
    with open('val.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # test data
    melanoma = glob.glob(os.path.join(root_dir, 'Test', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Test', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Test', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])

class ISIC(udata.Dataset):
    def __init__(self, csv_file, transform=None):
        file = open(csv_file, newline='')
        reader = csv.reader(file, delimiter=',')
        self.pairs = [row for row in reader]
        self.transform = transform
    def __len__(self):
        return len(self.pairs)
    def  __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair[0])
        image_seg = Image.open(pair[1])
        label = int(pair[2])
        # construct one sample
        sample = {'image': image, 'image_seg': image_seg, 'label': label}
        # transform
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data(isBenign = False, transform = None):
    """
        Return Images (torch.FloatTensor), Labels (torch.LongTensor)
    """
    #isBenign = False
    benignRoot = "./adversarial_result/ori_img/benign"
    malignantRoot = "./adversarial_result/ori_img/malignant"

    benignImgs = [
        "ISIC_0000213.jpg","ISIC_0000222.jpg",
        "ISIC_0000254.jpg","ISIC_0000470.jpg"
    ]
    malignImgs = [
        "ISIC_0000013.jpg", "ISIC_0000022.jpg",
        "ISIC_0000466.jpg", "ISIC_0000549.jpg"
    ]

    if isBenign is True:
        Imgs = benignImgs
        Root = benignRoot
        label = 0
    else:
        Imgs = malignImgs
        Root = malignantRoot
        label = 1

    images = torch.zeros([4,3,224,224])
    labels = torch.zeros([4])

    for batchIdx, Img in enumerate(Imgs):
        benignPath = os.path.join(Root, Img)
        img = Image.open(benignPath)
        sample = {'image': img, 'image_seg': img, 'label': label}
        if transform is not None:
            t_sample = transform(sample)
        img = t_sample["image"]
        #img.unsqueeze_(0)
        images[batchIdx] = img
        labels[batchIdx] = t_sample['label']
        #print(images.shape)
    # print(img1.shape)
    images = images.cuda()
    labels = labels.type(torch.LongTensor).cuda()

    return images, labels
