import os
import sys
import time
import warnings
import torch.utils.data as data
from PIL import Image
import pandas as pd
## 输出 对抗样本 指标
## adversarial accuracy



## Load ImageNet Validation dataset
class valDataset(data.Dataset):
    def __init__(self, root_dir, labelDir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = os.listdir(root_dir)
        self.image_paths.sort()
        self.labels = pd.read_csv(labelDir,delimiter=" ",header=None ,names=["image_path","label"])

    def __getitem__(self, i):
        assert self.image_paths[i] == self.labels["image_path"][i]
        image_path = os.path.join(self.root_dir, self.image_paths[i])
        img = Image.open(image_path)

        img = self.transforms(img)
        if img.shape[0] == 1:
            img = img.expand(3,224,224)

        label = self.labels["label"][i]
        return img, label
    def __len__(self):
        return len(self.image_paths)




## Define Progress Bar
isDefinePB = True
try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    isDefinePB = True
except ValueError:
    warnings.warn("Code Running Environment is not in Terminal, we can not invoke progress_bar!!")
    isDefinePB = False

if isDefinePB is True:
    TOTAL_BAR_LENGTH = 30.
    last_time = time.time()
    begin_time = last_time
    def progress_bar(current, total, msg=None):
        global last_time, begin_time
        if current == 0:
            begin_time = time.time()  # Reset for new bar.

        cur_len = int(TOTAL_BAR_LENGTH*current/total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - last_time
        last_time = cur_time
        tot_time = cur_time - begin_time

        L = []
        L.append('  Step: %s' % format_time(step_time))
        L.append(' | Tot: %s' % format_time(tot_time))
        if msg:
            L.append(' | ' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current+1, total))

        if current < total-1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def format_time(seconds):
        days = int(seconds / 3600/24)
        seconds = seconds - days*3600*24
        hours = int(seconds / 3600)
        seconds = seconds - hours*3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes*60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds*1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

if __name__=="__main__":
    for i in range(520):
        progress_bar(i, 520, "step : %d" %(i))
        time.sleep(0.1)
