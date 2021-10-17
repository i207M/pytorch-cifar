'''Train CIFAR10 with PyTorch.'''

import argparse
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.resnet56 import ResNet56        
# from utils_old import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data2', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
    root='./data2', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = ResNet56()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/last.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# Original: weight decay = 1e-4
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
# Original scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], last_epoch=start_epoch - 1
)

# Log
date_str = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())
log_dir = Path('./runs') / date_str
wdir = log_dir / 'weights'
writer = SummaryWriter(log_dir)
PRINT_FREQ = 100


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
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

        # progress_bar(
        #     batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
        #     (train_loss / (batch_idx + 1), 100. * correct / total, correct, total)
        # )
        if batch_idx % PRINT_FREQ == 0:
            print(
                'idx: %d, Loss: %.3f | Acc: %.3f' %
                (batch_idx, train_loss / (batch_idx + 1), 100. * correct / total)
            )
    writer.add_scalar('train/loss', train_loss / len(trainloader), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


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

            # progress_bar(
            #     batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            #     (test_loss / (batch_idx + 1), 100. * correct / total, correct, total)
            # )
            if batch_idx % PRINT_FREQ == 0:
                print(
                    'idx: %d, Loss: %.3f | Acc: %.3f' %
                    (batch_idx, test_loss / (batch_idx + 1), 100. * correct / total)
                )
    writer.add_scalar('test/loss', test_loss / len(testloader), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(wdir / 'best.pth', epoch)


def save_checkpoint(path, epoch: int):
    global best_acc
    print(f'Saving.. Epoch: {epoch}, Acc: {best_acc}')
    state = {
        'net': net.state_dict(),
        'acc': best_acc,
        'epoch': epoch,
    }
    torch.save(state, path)


if __name__ == '__main__':
    os.makedirs(wdir, exist_ok=True)
    for epoch in range(start_epoch, start_epoch + 200):
        try:
            train(epoch)
            test(epoch)
            scheduler.step()
        except KeyboardInterrupt:
            break
    save_checkpoint(wdir / 'last.pth', epoch)
