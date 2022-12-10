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

from pthflops import count_ops


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
    total_ops = 0
    with torch.autograd.profiler.profile(with_flops=True) as prof:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                ops, data = count_ops(model, inputs)
                total_ops += ops
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                            
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print("Total FLOPs: " + str(total_ops))


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
    parser.add_argument('--model', type=str, default="ResNet18", help='Model name to run training on (default: resnet18)')
    parser.add_argument('--device', type=str, default="0", help='The GPU to use (default: "0"')
    parser.add_argument('--dataset', type=str, default="MNIST", help='Name of dataset to use')
    parser.add_argument('--use-synthetic', action='store_true', default=False, help='use synthetic dataset for training')
    parser.add_argument('--synthetic-path', type=str, default=None, help='Path to the synthetic dataset')
    parser.add_argument('--trajectory-matching', action='store_true', default=False, help='use synthetic dataset from trajectory matching')


    args = parser.parse_args()
    #args creator to use utils.py code
    args.dsa = False
    args.dsa_strategy = 'None'
    args.dsa_param = ParamDiffAug()
    args.augment = True
    args.ipc = 10

    images_trajectory_matching_images = "..\Task_2\logged_files\CIFAR10\glamorous-hill-13\images_best.pt"
    labels_trajectory_matching_images = "..\Task_2\logged_files\CIFAR10\glamorous-hill-13\labels_best.pt"

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:"+args.device if use_cuda else "cpu")
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training on: "+str(device))

    if args.dataset == "MNIST":
        channel = 1
        image_size = (28, 28)
        num_classes = 10
    elif args.dataset == "CIFAR10":
        channel = 3
        image_size = (32, 32)
        num_classes = 10
    else:
        print("Unrecognized dataset, exiting")
        exit(1)

    model = get_network(args.model, channel, num_classes, image_size)
    if model is None:
        print("Model doesn't exist.")
        return
    print("Selected NN Architecture: "+str(args.model))
    model.to(device)


    if args.use_synthetic:
        if not args.trajectory_matching:
            data = torch.load(args.synthetic_path, map_location='cpu')
            synthetic_data = data['data'][0]
            synthetic_data_image_info = synthetic_data[0].to(args.device)
            synthetic_data_label_info = synthetic_data[1].to(args.device)
        else:
            images = torch.load(images_trajectory_matching_images, map_location='cpu')
            labels = torch.load(labels_trajectory_matching_images, map_location='cpu')
            synthetic_data_image_info = images.to(args.device)
            synthetic_data_label_info = labels.to(args.device)

        synthetic_dataset = TensorDataset(synthetic_data_image_info, synthetic_data_label_info)
        train_loader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=1, shuffle=True, num_workers=0)
        args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        print('DC augmentation parameters: \n', args.dc_aug_param)
        args.augment = True
        if args.dataset == "CIFAR10":
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = datasets.CIFAR10(
                root='./cifar10data', train=False, download=True, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        elif args.dataset == "MNIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
            testset = datasets.MNIST(root='./mnistdata', train=False, download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        else:
            print("Unrecognized dataset, exiting now")
            exit(1)
        print("Using synthetic dataset to train the network")
    else:
        if args.dataset == "CIFAR10":
            transform_train = transforms.Compose([
                #transforms.RandomCrop(32, padding=4), # we could apply data augmentation to the dataset, but to keep it consistent we don't
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = datasets.CIFAR10(
                root='./cifar10data', train=True, download=True, transform=transform_train)
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

            testset = datasets.CIFAR10(
                root='./cifar10data', train=False, download=True, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
            print("Using original CIFAR10 dataset to train the network")
        elif args.dataset == "MNIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
            trainset = datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transform) # no augmentation
            testset = datasets.MNIST(root='./mnistdata', train=False, download=True, transform=transform)

            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        else:
            print("Unrecognized dataset, exiting now")
            exit(1)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_train = time.time()
    lr_schedule = [args.epochs//2+1]
    lr = args.lr
    total_train = 0.0
    for ep in range(1, args.epochs + 1):
        if args.use_synthetic:
            start_time_epoch = time.time()
            loss_train, acc_train = epoch('train', train_loader, model, optimizer, criterion, args, aug = args.augment)
            loss_test, acc_test = epoch('test', test_loader, model, optimizer, criterion, args, aug = False)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            time_train = time.time() - start_train
            total_epoch_time = time.time() - start_time_epoch
            total_train += total_epoch_time
            print('%s Epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), ep, int(time_train), loss_train, acc_train, acc_test))
        else:
            train(ep, model, device, train_loader, optimizer, criterion)
            test(ep, model, device, test_loader, criterion)    
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) 
            #scheduler.step()   
    end_train = time.time()
    print("Total elapsed training time: " + str(end_train - start_train))
    print("Total training time: " + str(total_train))
        
        

if __name__ == '__main__':
    main()