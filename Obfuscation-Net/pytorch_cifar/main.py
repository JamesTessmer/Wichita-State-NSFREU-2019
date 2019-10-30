'''Train CIFAR10 with PyTorch.'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import sys
import time

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
RESUME = True
RES_CLASS_MODEL_DIR = os.path.join(os.pardir, 'models/class-models/Base_Class_2018-06-29_21-28-00')
RES_CLASS_MODEL_NAME = 'class-model_resnet18_ep-50_ac-70.820.t7'

start_epoch = 51  # start from epoch 0 or last checkpoint epoch
epoch_count = 50

# Data
print('==> Preparing data..')
USE_SUBSET = False
TRAIN_START_IDX = 0
TRAIN_CNT = 20000
TRAIN_BATCH_SZ = 100
TEST_START_IDX = 0
TEST_CNT = 5000
TEST_BATCH_SZ = 100
DATETIME = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
model_filepath = os.path.join(os.pardir, "models", "class-models", DATETIME)
if not os.path.isdir(model_filepath):
    os.mkdir(model_filepath)
info_filepath = os.path.join(model_filepath, "info.txt")
info_note = "Resuming ResNet18 classification training with full CIFAR-10 dataset"


def datasubset(loader, start, count, batch_size):
    """Return a subset of the dataloader from the start batch index to the count specified."""
    # Note: start is the start index of batch, not image
    smaller_dataset = []
    end_idx = count / batch_size
    for batch_idx, (orig_images, labels) in enumerate(loader):
        if start <= batch_idx < end_idx:
            smaller_dataset.append((orig_images, labels))
        if batch_idx > end_idx:
            break
    return smaller_dataset


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SZ, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SZ, shuffle=False, num_workers=2)

if USE_SUBSET:
    trainloader = datasubset(trainloader, TRAIN_START_IDX, TRAIN_CNT, TRAIN_BATCH_SZ)
    testloader = datasubset(testloader, TEST_START_IDX, TEST_CNT, TEST_BATCH_SZ)

# Store info to file
with open(info_filepath, "w") as f:
    print(info_note, file=f)
    print("\nDatetime:", DATETIME, file=f)
    print("Initial Model:", os.path.join(RES_CLASS_MODEL_DIR, RES_CLASS_MODEL_NAME), file=f)
    print("Number of training images:", len(trainloader) * TRAIN_BATCH_SZ, file=f)
    print("Number of testing images:", len(testloader) * TEST_BATCH_SZ, file=f)
    print("\n", file=f)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('\n==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if RESUME:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(RES_CLASS_MODEL_DIR), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(RES_CLASS_MODEL_DIR, RES_CLASS_MODEL_NAME))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if epoch % 5 == 0 or epoch == epoch_count:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        filename = "class-model_resnet18_ep-%d_ac-%.3f.t7" % (epoch, acc)
        torch.save(state, os.path.join(model_filepath, filename))
        with open(info_filepath, "a") as f:
            print("Epoch: %d, Accuracy: %.3f" % (epoch, acc), file=f)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+epoch_count):
    print('\nEpoch: %d' % epoch)
    print("Training..")
    train(epoch)
    print("Evaluating..")
    test(epoch)
