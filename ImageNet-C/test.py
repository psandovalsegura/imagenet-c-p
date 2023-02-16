# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
# from resnext_50_32x4d import resnext_50_32x4d
# from resnext_101_32x4d import resnext_101_32x4d
# from resnext_101_64x4d import resnext_101_64x4d
# from densenet_cosine_264_k48 import densenet_cosine_264_k48
# from condensenet_converted import CondenseNet

parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Architecture
parser.add_argument('--model-name', '-m', type=str,
                    choices=['alexnet', 'squeezenet1.0', 'squeezenet1.1', 'condensenet4', 'condensenet8',
                             'vgg11', 'vgg', 'vggbn',
                             'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet264',
                             'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                             'resnext50', 'resnext101', 'resnext101_64'])
parser.add_argument('--ckpt-path', type=str, default='', help='Checkpoint path to load and test.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--test-bs', type=int, default=1024, help='Test batch size.')
args = parser.parse_args()
print(args)

model_dir = os.environ['MODEL_DIR']
imagenet_dir = os.environ['IMAGENET_DIR']
imagenet_c_dir = os.environ['IMAGENET_C_DIR']

# /////////////// Model Setup ///////////////

class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

def apply_blurpool(mod: torch.nn.Module):
    for (name, child) in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
            setattr(mod, name, BlurPoolConv2d(child))
        else: apply_blurpool(child)

if args.model_name == 'alexnet':
    net = models.AlexNet()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
                                           model_dir=model_dir))

elif args.model_name == 'squeezenet1.0':
    net = models.SqueezeNet(version=1.0)
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
                                           model_dir=model_dir))

elif args.model_name == 'squeezenet1.1':
    net = models.SqueezeNet(version=1.1)
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
                                           model_dir=model_dir))

elif args.model_name == 'condensenet4':
    args.evaluate = True
    args.stages = [4,6,8,10,8]
    args.growth = [8,16,32,64,128]
    args.data = 'imagenet'
    args.num_classes = 1000
    args.bottleneck = 4
    args.group_1x1 = 4
    args.group_3x3 = 4
    args.reduction = 0.5
    args.condense_factor = 4
    net = CondenseNet(args)
    state_dict = torch.load('./converted_condensenet_4.pth')['state_dict']
    for i in range(len(state_dict)):
        name, v = state_dict.popitem(False)
        state_dict[name[7:]] = v     # remove 'module.' in key beginning
    net.load_state_dict(state_dict)

elif args.model_name == 'condensenet8':
    args.evaluate = True
    args.stages = [4,6,8,10,8]
    args.growth = [8,16,32,64,128]
    args.data = 'imagenet'
    args.num_classes = 1000
    args.bottleneck = 4
    args.group_1x1 = 8
    args.group_3x3 = 8
    args.reduction = 0.5
    args.condense_factor = 8
    net = CondenseNet(args)
    state_dict = torch.load('./converted_condensenet_8.pth')['state_dict']
    for i in range(len(state_dict)):
        name, v = state_dict.popitem(False)
        state_dict[name[7:]] = v     # remove 'module.' in key beginning
    net.load_state_dict(state_dict)

elif 'vgg' in args.model_name:
    if 'bn' not in args.model_name:
        net = models.vgg19()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
                                               model_dir=model_dir))
    elif '11' in args.model_name:
        net = models.vgg11()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
                                               model_dir=model_dir))
    else:
        net = models.vgg19_bn()
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
                                               model_dir=model_dir))

elif args.model_name == 'densenet121':
    net = models.densenet121()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet121-a639ec97.pth',
                                           model_dir=model_dir))

elif args.model_name == 'densenet169':
    net = models.densenet169()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
                                           model_dir=model_dir))

elif args.model_name == 'densenet201':
    net = models.densenet201()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet201-c1103571.pth',
                                           model_dir=model_dir))

elif args.model_name == 'densenet161':
    net = models.densenet161()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet161-8d451a50.pth',
                                           model_dir=model_dir))

elif args.model_name == 'densenet264':
    net = densenet_cosine_264_k48
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/densenet_cosine_264_k48.pth',
                                           model_dir=model_dir))

elif args.model_name == 'resnet18':
    net = models.resnet18()
    if args.ckpt_path != '':
        # ffcv imagenet checkpoints have blurpool layers
        apply_blurpool(net)
        print('Loading checkpoint from {}'.format(args.ckpt_path))
        net.load_state_dict(torch.load(args.ckpt_path))
    else:
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
                                                model_dir=model_dir))
    

elif args.model_name == 'resnet34':
    net = models.resnet34()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                                           model_dir=model_dir))

elif args.model_name == 'resnet50':
    net = models.resnet50()
    if args.ckpt_path != '':
        # ffcv imagenet checkpoints have blurpool layers
        apply_blurpool(net)
        print('Loading checkpoint from {}'.format(args.ckpt_path))
        net.load_state_dict(torch.load(args.ckpt_path))
    else:
        net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                            model_dir=model_dir))

elif args.model_name == 'resnet101':
    net = models.resnet101()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                                           model_dir=model_dir))

elif args.model_name == 'resnet152':
    net = models.resnet152()
    net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                                           model_dir=model_dir))

elif args.model_name == 'resnext50':
    net = resnext_50_32x4d
    net.load_state_dict(torch.load('/share/data/lang/users/dan/.torch/models/resnext_50_32x4d.pth'))

elif args.model_name == 'resnext101':
    net = resnext_101_32x4d
    net.load_state_dict(torch.load('/share/data/lang/users/dan/.torch/models/resnext_101_32x4d.pth'))

elif args.model_name == 'resnext101_64':
    net = resnext_101_64x4d
    net.load_state_dict(torch.load('/share/data/lang/users/dan/.torch/models/resnext_101_64x4d.pth'))

args.prefetch = 4

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

torch.manual_seed(1)
np.random.seed(1)
if args.ngpu > 0:
    torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders

print('Model Loaded')

# /////////////// Data Loader ///////////////

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
    root=imagenet_dir,
    transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])),
    batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)


# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area

# /////////////// Clean Evaluation ///////////////
correct = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(clean_loader):
        data = data.cuda()
        output = net(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.cuda()).sum()

clean_error = 1 - correct / len(clean_loader.dataset)
print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


def show_performance(distortion_name):
    errs = []

    for severity in range(1, 6):
        distorted_dataset = dset.ImageFolder(
            root=imagenet_c_dir + distortion_name + '/' + str(severity),
            transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                data = data.cuda()
                output = net(data)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.cuda()).sum().item()

        errs.append(1 - 1.*correct / len(distorted_dataset))

    print('\n=Average', errs)
    return np.mean(errs)


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

print('Starting ImageNet-C eval...')

alexnet_errors = [88.6, 89.4, 92.3, 82.0, 82.6, 78.6, 79.8, 86.7, 82.7, 81.9, 56.5, 85.3, 64.6, 71.8, 60.7, 84.5, 78.7, 71.8, 65.8]

distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]
normalized_error_rates = []
error_rates = []
for i, distortion_name in enumerate(distortions):
    rate = show_performance(distortion_name)
    error_rates.append(rate)
    print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))

    normalized_rate = (rate * 100) / alexnet_errors[i]
    normalized_error_rates.append(normalized_rate)
    print('Distortion: {:15s}  | CE (normalized) (%): {:.2f}'.format(distortion_name, 100 * normalized_rate))

print('*'*20)
print('\nSummary:')
print('Clean error of model name: {} (%): {:.2f}'.format(args.model_name, 100 * clean_error))
print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))
print('mCE (normalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(normalized_error_rates)))
