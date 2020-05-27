

from PIL import Image
import torch
def load_data(file_list,transform=None):
    images = torch.zeros([len(file_list),3,299,299])
    for batchIdx, filename in enumerate(file_list):
        img = Image.open(filename)
        if transform is not None:
            img = transform(img)
        images[batchIdx] = img
    return images
