'''Train CIFAR10 with PyTorch.'''

# Cifar-10 dataset을 closed-set으로 학습을 시키고 SVHN test dataset을 openset으로 Test하는 코드입니다.
# SVHN 데이터셋은 검색해보시면 어떠한 데이터셋인지 나올 겁니다.



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn



import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


import os
import argparse

from models import *
from utils import progress_bar

import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_cls', default=4, type=int, help="num classes")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data

# data의 전처리를 정의하는 부분입니다.
print('==> Preparing data..')
transform_train = transforms.Compose([

    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



# dataset을 불러오는 부분입니다.
#---------------------------------------------------------------------
# train data를 불러오는 부분입니다.
trainset = torchvision.datasets.ImageFolder(
    root='./data/custom/train_data', transform=transform_train)

# train dataloader를 불러오는 부분입니다.
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

# test data를 불러오는 부분입니다.
testset = torchvision.datasets.ImageFolder(
    root='./data/custom/test_data', transform=transform_test)

# test dataloader를 불러오는 부분입니다.
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)



# --------------------------------------------------------------------------------

# 학습 Model을 정의하는 부분입니다. Resnet18을 사용하겠습니다.

print('==> Building model..')

num_classes =args.num_cls
lamda= 1

# Resnet을 분류 모델로 사용하겠습니다.
net = models.resnet50(pretrained=True)

# 마지막 fc layer를 클래수 개수에 맞게 수정하는 부분입니다.
net.fc =nn.Linear(2048,num_classes)

# gpu device에 모델을 올리는 부분입니다.
net = net.to(device)



# 저장된 모델을 load하는 부분입니다.
# ----------------------------------------------------------------------------------
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
# ----------------------------------------------------------------------------------



# loss function 및 optimizaer, learning rate scheduler를 정의하는 부분입니다.
# -------------------------------------------------------------------------------------

# 분류 문제이기 때문에 CrossEntropyLoss를 사용하겠습니다.
criterion = nn.CrossEntropyLoss()

# optimizer는 SGD를 사용하겠습니다.
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

# learning rate scheduler를 사용하겠습니다. 이는 epoch이 변할 때마다 learning rate를 조절해주는 역할을 합니다.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# --------------------------------------------------------------------------------------




def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(epoch):
    print('\nEpoch: %d' % epoch)
    print("Current lr : {}".format(get_lr(optimizer)))
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # data를 batch 단위로 불러오는 부분입니다.
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # data를 device에 올리는 부분입니다.
        inputs ,targets = inputs.to(device) ,targets.to(device)

        # gradient를 0으로 초기화하는 부분입니다.
        optimizer.zero_grad()

        # 모델에 data를 넣어 output을 얻는 부분입니다.
        outputs = net(inputs)

        # loss를 계산하는 부분입니다.
        loss = criterion(outputs, targets)

        # loss를 이용해 backpropagation을 하는 부분입니다.
        loss.backward()

        # optimizer를 이용해 parameter를 업데이트하는 부분입니다.
        optimizer.step()

        # loss 및 accuracy를 계산하는 부분입니다.
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))




# test 하는 함수입니다.
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    pred_all = []
    target_all = []

    # torch.no_grad()를 사용하면 gradient를 계산하지 않겠다는 의미입니다.
    # test 과정에서는 gradient를 계산할 필요가 없기 때문에 사용합니다.
    with torch.no_grad():

        # data를 batch 단위로 불러오는 부분입니다.
        for batch_idx, (inputs, targets) in enumerate(testloader):

            # data를 device에 올리는 부분입니다.
            inputs, targets = inputs.to(device), targets.to(device)

            # 모델에 data를 넣어 output을 얻는 부분입니다.
            outputs = net(inputs)

            # loss를 계산하는 부분입니다.
            # test 과정에서 loss를 계산하는 이유는 모델이 얼마나 정확한지를 확인하기 위함입니다. (업데이트는 하지 않습니다.)
            loss = criterion(outputs, targets)

            # loss 및 accuracy를 계산하는 부분입니다.
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pred_all.extend(predicted.data.cpu().numpy())
            target_all.extend(targets.data.cpu().numpy())

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        print("Closed-Set Confusion Matrix")
        print(metrics.confusion_matrix(target_all, pred_all, labels=range(num_classes)))


    # Save checkpoint.

    # accuracy가 높아질 때마다 모델을 저장하는 부분입니다.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__=='__main__':
    #실제 코드 실행하는 부분입니다.
    for epoch in range(start_epoch, start_epoch+300):
        train(epoch) #train 함수 호출
        test(epoch)  #test 함수 호출
