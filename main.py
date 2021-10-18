'''Train CIFAR10 with PyTorch.'''

import argparse
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.resnet56 import ResNet56

# Args
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument(
    '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)'
)
parser.add_argument(
    '-e', '--epoch', default=200, type=int, metavar='N', help='number of total epochs to run'
)
parser.add_argument(
    '--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)'
)
parser.add_argument(
    '-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)'
)
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)'
)
parser.add_argument('-n', '--name', default='', type=str, help='name of the training')
parser.add_argument('-r', '--resume', default='', type=str, metavar='D', help='resume from checkpoint')
args = parser.parse_args()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),  # Important
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Log
date_str = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime())
if args.name != '':
    args.name = '-' + args.name
log_dir = Path('./runs') / (date_str + args.name)
wdir = log_dir / 'weights'
writer = SummaryWriter(log_dir)
open(log_dir / 'args.txt', 'w').write(str(args.__dict__))

# Model
print('==> Building model..')
net = ResNet56()
net = net.cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

best_acc = 0  # best test accuracy

if args.resume != '':
    # Load checkpoint
    print('==> Resuming from checkpoint..')
    ckpt_path = Path(args.resume)
    assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(ckpt_path / 'weights/last.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    args.start_epoch = checkpoint['epoch']

# Original weight decay = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer, milestones=[100, 150], last_epoch=start_epoch - 1
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch: int):
    print('\nEpoch: %d' % epoch)
    start_time = time.time()
    net.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / len(trainloader)
    avg_acc = 100. * correct / total
    print('Train Loss: %.3f | Acc: %.3f | Time: %.3fms' % (avg_loss, avg_acc, time.time() - start_time))
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/acc', avg_acc, epoch)


def test(epoch: int):
    start_time = time.time()
    net.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(testloader)
    avg_acc = 100. * correct / total
    print('Test Loss: %.3f | Acc: %.3f | Time: %.3fms' % (avg_loss, avg_acc, time.time() - start_time))
    writer.add_scalar('test/loss', avg_loss, epoch)
    writer.add_scalar('test/acc', avg_acc, epoch)

    # Save checkpoint
    global best_acc
    if avg_acc > best_acc:
        best_acc = avg_acc
        save_checkpoint(wdir / 'best.pth', epoch)


def save_checkpoint(path, epoch: int):
    global best_acc
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': best_acc,
        'epoch': epoch,
    }
    torch.save(state, path)


if __name__ == '__main__':
    os.makedirs(wdir, exist_ok=True)
    try:
        for epoch in range(args.start_epoch, args.start_epoch + args.epoch):
            train(epoch)
            test(epoch)
            scheduler.step()
    finally:
        save_checkpoint(wdir / 'last.pth', epoch)
