import torch.nn as nn
class DeepFool():
    def __init__(self, model, **kwargs):
        self.model = model
        self.parse_params(**kwargs)


    def generate(self, inputs):
        x = inputs.detach()
        x.requires_grad_()
        fs = self.model(x)

        num_classes = logits.shape[1]
        ## ind with shape [1, n_classes]
        _, ind = torch.sort(fs, dim=1)
        ### sort the logits by decending
        fs_list = [fs[0,ind[k]] for k in range(num_classes)]
        label = fs_list[0]
        k_fake = label

        while k_fake == label and iter < self.max_iters:

            pert = torch.Tensor(float["Inf"])

            fs[0,ind[0]].backward(retain_graph=True)
            grad_orig = x.grad.data

            for k in range(1, num_classes):
                self.model.zero_grad()

                f = fs[0, ind[k]]
                f.backward(retain_graph = True)
                grad_curr = x.grad.data

                #calculate w_k and f_k
                w_k = grad_curr - grad_orig
                f_k = (fs[0, ind[k]] - fs[0, ind[0]])

                pert_k = torch.abs(f_k) / torch.norm(w_k)
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r_i =  (pert + 1e-4) * w / torch.norm(w)
            r_tot = r_tot + r_i

            pert_image = image + (1 + self.overshoot) * torch.from_numpy(r_tot).cuda()
            fs = net.forward(x)
            k_i = torch.argmax(fs)
            loop_i += 1
        r_tot = (1 + self.overshoot) * r_tot
        return pert_image, r_tot

    def parse_params(self,
                    max_iters,
                    overshoot):
        self.max_iters = max_iters
        self.overshoot = overshoot



if __name__=="__main__":

    import torchvision.models as models
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    pretrained_clf = models.resnet18(pretrained = True)
    pretrained_clf = pretrained_clf.cuda()
    pretrained_clf.eval()

    im_orig = Image.open('../models/cat_dog.png')
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]

    params = {
        "max_iters" : 100,
        "overshoot" : 0.2
    }

    deepFoolAttack = DeepFool(pretrained_clf, **params)
    # Remove the mean
    im = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)])(im_orig)

    print(im.shape)

    im.unsqueeze_(0)
    im = im.cuda()
    #im.requires_grad_()
    output = pretrained_clf(im)
    adv_x = deepFoolAttack.generate(im)
    #output[0,0].backward()
    #print(im.grad.shape)
    #print(torch.argmax(output, dim=1))

    #trainDataset = torchvision.dataset.CIFAR10()
