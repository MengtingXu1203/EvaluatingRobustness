import warnings
import torch.nn as nn
import torch
class FastGradientSignMethod():
    def __init__(self, model):

        if not isinstance(model, nn.Module):
            raise TypeError("The model argument should be the instance of"
                        "torch.nn.Module")

        self.model = model

    def generate(self, inputs, **kwargs):
        assert self.parse_params(**kwargs)
        ## targeted or non-targeted
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        else:
            y = self.y
            targeted = False

        x = inputs.detach()
        x.requires_grad_()

        criterion = nn.CrossEntropyLoss()
        with torch.enable_grad():
            outputs = self.model(x)
            if isinstance(outputs, list):
                outputs = outputs[0]
            l = criterion(outputs, y)
        gradient = torch.autograd.grad(l, [x])[0]

        if targeted:
            x = x - self.eps * torch.sign(gradient)
        else:
            x = x + self.eps * torch.sign(gradient)

        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x


    def parse_params(self,
                    eps = 0.3,
                    y = None,
                    ord = "inf",
                    clip_min = 0.,
                    clip_max = 1.,
                    y_target = None,
                    **kwargs):
        self.eps = eps
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        if self.ord not in ["inf", 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        if len(kwargs.keys()) > 0:
          warnings.warn("kwargs is unused and will be removed on or after "
                        "2019-04-26.")
        return True


if __name__=="__main__":

    import sys
    sys.path.append("../models")
    import cnet as CNet
    import torchvision
    import torchvision.transforms as transforms

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    pretrained_clf = CNet.Net()
    pretrained_clf = pretrained_clf.cuda()
    pretrained_clf.load_state_dict(torch.load("../models/lenet_mnist_model.pth"))
    pretrained_clf.eval()

    trainDataset = torchvision.datasets.MNIST(root="/home/lrh/dataset/mnist",train=True,download=False,transform = transform_train)
    trainLoader = torch.utils.data.DataLoader(trainDataset,batch_size=1000,shuffle=True,num_workers=2)
    FGSMAttack = FastGradientSignMethod(model=pretrained_clf)
    correct = 0
    total = 0
    for image,label in trainLoader:
        image, label = image.cuda(), label.cuda()
        params = {
            "eps" : 0.3,
            "y" : label,
            "clip_min" : 0,
            "clip_max" : 1,
            "ord" : "inf"
        }
        adv_x = FGSMAttack.generate(image, **params)
        adv_output = pretrained_clf(adv_x)
        adv_label = torch.argmax(adv_output, dim = 1)

        total = total + image.size(0)
        correct = correct + label.eq(adv_label).sum()
        print("ACC : %.4f (%d,%d)" %(float(correct)/total, correct, total))
