import warnings
import torch.nn as nn
import numpy as np
import torch

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
                if isinstance(logits, list):
                    logits = logits[0]

                loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            #print(grad)
            gradient = self.eps_iter * torch.sign(grad)
            if targeted is True:
                x = x - gradient
            else:
                x = x + gradient

            ## norm constrains on perturbations
            x = torch.min(torch.max(x, inputs - self.eps), inputs + self.eps)
            x = torch.clamp(x, self.clip_min, self.clip_max)

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


if __name__=="__main__":
    from torchvision import models
    import torchvision
    import numpy as np
    import torch
    from torchvision import transforms
    transform_train = transforms.Compose([
        transforms.ToTensor()
        #transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
    ])
    #resnet18 = models.resnet18(pretrained = True)

    pgd_params = {
            'ord': np.inf,
            'y': None,
            'eps': 76.5 / 255,
            'eps_iter': 2.55 / 255,
            'nb_iter': 40,
            'rand_init': True,
            'rand_minmax': 76.5 / 255,
            'clip_min': 0.,
            'clip_max': 1.,
            'sanity_checks': True
        }

    # DEFINE PRE-TRAINED model
    import sys
    sys.path.append("../models")
    import cnet as CNet

    pretrained_clf = CNet.Net()
    pretrained_clf = pretrained_clf.cuda()
    pretrained_clf.load_state_dict(torch.load("../models/lenet_mnist_model.pth"))
    pretrained_clf.eval()

    PGDAttack = ProjectGradientDescent(model = pretrained_clf)

    train_dataset = torchvision.datasets.MNIST(root="/home/lrh/dataset/mnist",train=True,download=False,transform = transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1000,shuffle=True,num_workers=2)
    correct = 0
    total = 0
    for image,label in train_loader:
        image, label = image.cuda(), label.cuda()
        ## non targeted
        pgd_params['y'] = label
        adv_x = PGDAttack.generate(image, **pgd_params)

        outputs = pretrained_clf(adv_x)
        pred_adv = torch.argmax(outputs, dim = 1)
        torchvision.utils.save_image(adv_x, "adv.jpg", nrow = 50 )

        total = total + image.size(0)
        correct = correct + label.eq(pred_adv).sum()
        print("ACC: %.4f (%d, %d)" %(float(correct) / total, correct, total))
