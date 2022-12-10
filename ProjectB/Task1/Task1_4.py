import sys
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import profiler
import collections
import numpy as np
import time
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import platform

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Function from https://github.com/kuangliu/pytorch-cifar
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

# Function from https://github.com/kuangliu/pytorch-cifar
if platform.system() != "Windows":
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
else:
    term_width = int(os.popen('mode con | findstr Columns','r').read().strip('\n').strip(' ').split(':')[1].strip(' '))

TOTAL_BAR_LENGTH = 65.
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


# Function from https://github.com/kuangliu/pytorch-cifar
# Training
def train(epoch, model, device, trainloader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Function from https://github.com/kuangliu/pytorch-cifar
def test(epoch, model, device, testloader, criterion):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR10/MNIST vanilla and synthetic training for ECE1512 Project B')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 200)')    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--model', type=str, default="ConvNet", help='Model name to run training on (default: resnet18)')
    parser.add_argument('--device', type=str, default="0", help='The GPU to use (default: "0"')
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Name of dataset to use')
    parser.add_argument('--synthetic-path', type=str, default=None, help='Path to the synthetic dataset')


    args = parser.parse_args()
    #args creator to use utils.py code
    args.dsa = False
    args.dsa_strategy = 'None'
    args.dsa_param = ParamDiffAug()
    args.augment = True
    args.ipc = 10

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:"+args.device if use_cuda else "cpu")
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training on: "+str(device))

    channel = 1
    image_size = (28, 28)
    num_classes = 10

    model = get_network(args.model, channel, num_classes, image_size)
    if model is None:
        print("Model doesn't exist.")
        return
    print("Selected NN Architecture: "+str(args.model))
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
    trainset_task1 = datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transform)
    trainset_task2 = datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transform)
    idx_task1 = (trainset_task1.targets==0) | (trainset_task1.targets==1) | (trainset_task1.targets==2) | (trainset_task1.targets==3) | (trainset_task1.targets==4)
    idx_task2 = (trainset_task1.targets==5) | (trainset_task1.targets==6) | (trainset_task1.targets==7) | (trainset_task1.targets==8) | (trainset_task1.targets==9)
    # we divide the 10 classes of MNIST into two different datasets so that we can simulate continual learning
    # by first showing the network the first half of the dataset and then the second half
    # this way we can try to simulate catastrophic forgetting
    trainset_task1.targets = trainset_task1.targets[idx_task1]
    trainset_task1.data = trainset_task1.data[idx_task1]
    trainset_task2.targets = trainset_task2.targets[idx_task2]
    trainset_task2.data = trainset_task2.data[idx_task2]
    testset = datasets.MNIST(root='./mnistdata', train=False, download=True, transform=transform)
    train_loader_task1 = torch.utils.data.DataLoader(trainset_task1, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_loader_task2 = torch.utils.data.DataLoader(trainset_task2, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()
    
    # normal continual training to simulate catastrophic forgetting
    for ep in range(1, 10):
        train(ep, model, device, train_loader_task1, optimizer, criterion)
       
    for ep in range(1, 10):
        train(ep, model, device, train_loader_task2, optimizer, criterion)



    print("Starting training on continual learning with remembering")
    #args creator to use utils.py code
    args.dsa = False
    args.dsa_strategy = 'None'
    args.dsa_param = ParamDiffAug()
    args.augment = True
    args.ipc = 10

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    print("Training on: "+str(device))

    channel = 1
    image_size = (28, 28)
    num_classes = 10

    # continual learning with post-training remembering with the synthetic dataset
    model_syn = get_network(args.model, channel, num_classes, image_size)
    if model_syn is None:
        print("Model doesn't exist.")
        return
    print("Selected NN Architecture: "+str(args.model))
    model_syn.to(device)

    # load the synthetic dataset, MNIST specifically for this experiment
    data = torch.load(args.synthetic_path, map_location='cpu')
    synthetic_data = data['data'][0]
    synthetic_data_image_info = synthetic_data[0].to(args.device)
    synthetic_data_label_info = synthetic_data[1].to(args.device)
    synthetic_dataset = TensorDataset(synthetic_data_image_info, synthetic_data_label_info)
    synthetic_train_loader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=1, shuffle=True, num_workers=0)
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
    print('DC augmentation parameters: \n', args.dc_aug_param)
    args.augment = True

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
    trainset_task1 = datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transform)
    trainset_task2 = datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transform)
    idx_task1 = (trainset_task1.targets==0) | (trainset_task1.targets==1) | (trainset_task1.targets==2) | (trainset_task1.targets==3) | (trainset_task1.targets==4)
    idx_task2 = (trainset_task1.targets==5) | (trainset_task1.targets==6) | (trainset_task1.targets==7) | (trainset_task1.targets==8) | (trainset_task1.targets==9)
    # we divide the 10 classes of MNIST into two different datasets so that we can simulate continual learning
    # by first showing the network the first half of the dataset and then the second half
    # this way we can try to simulate catastrophic forgetting
    trainset_task1.targets = trainset_task1.targets[idx_task1]
    trainset_task1.data = trainset_task1.data[idx_task1]
    trainset_task2.targets = trainset_task2.targets[idx_task2]
    trainset_task2.data = trainset_task2.data[idx_task2]
    testset = datasets.MNIST(root='./mnistdata', train=False, download=True, transform=transform)
    train_loader_task1 = torch.utils.data.DataLoader(trainset_task1, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_loader_task2 = torch.utils.data.DataLoader(trainset_task2, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(model_syn.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()

    # normal continual training to simulate catastrophic forgetting
    for ep in range(1, 10):
        train(ep, model_syn, device, train_loader_task1, optimizer, criterion)
       
    for ep in range(1, 10):
        train(ep, model_syn, device, train_loader_task2, optimizer, criterion)

    for ep in range(1, 10):
        loss_train, acc_train = epoch('train', synthetic_train_loader, model_syn, optimizer, criterion, args, aug = args.augment)

    print("Evaluating normal continual learning")
    test(ep, model, device, test_loader, criterion)
    print("Evaluating continual learning with condensed dataset remembering")
    test(ep, model_syn, device, test_loader, criterion) 
           

if __name__ == '__main__':
    main()