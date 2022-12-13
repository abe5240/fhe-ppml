import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import wandb
from datetime import datetime

from utils import *
from models import *

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

os.environ["WANDB_MODE"] = "offline"

def create_splits():

    (trainloader, validloader) = get_train_valid_loader(data_dir='./data', 
                                                        batch_size=128,
                                                        augment=True,
                                                        random_seed=42,
                                                        valid_size=0.2,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=False)

    return trainloader, validloader


# Train epoch
def train_epoch(net, device, epoch, trainloader, criterion, optimizer):
    # Boilerplate training code.
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total

    return train_loss, acc


# Test epoch
def test(net, device, loader, best_acc, name, activation, order, save=True):
    # Boilerplate testing code.
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save best accuracy model
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        if not os.path.isdir(f'checkpoint/{name}'):
            os.mkdir(f'checkpoint/{name}')

        if not os.path.isdir(f'checkpoint/{name}/{order}'):
            os.mkdir(f'checkpoint/{name}/{order}')
        if save:
            torch.save(state, f'./checkpoint/{name}/{order}/ckpt_{activation}.pth')
        best_acc = acc

    return acc, best_acc, test_loss


# Train model
def train_model(net, epochs, name, activation, order, time):
        
        run = wandb.init(reinit=True, project=f'final-proj-{time}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        
        trainloader, validloader = create_splits()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5, weight_decay=0.0, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.001)

        # Train the model and test it on the validation set
        best_acc = 0  # best validation accuracy
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(net, device, epoch, trainloader, criterion, optimizer)
            valid_acc, best_acc, valid_loss = test(net, device, validloader, best_acc, name, activation, order)
            scheduler.step()

            wandb.log({"Training Accuracy": train_acc,
                       "Training Loss": train_loss,
                       "Validation Accuracy": valid_acc,
                       "Best Valid. Acc": best_acc,
                       "Validation Loss": valid_loss})

        run.finish()


if __name__ == "__main__":

    polynomial_orders = np.arange(2,11)
    model_names = ["MLPNet","LeNet"]
    activations = ["LeakyRelu", "Sigmoid", "Softplus"]

    epochs = 50
    time = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    for model_name in model_names:
        for activation in activations:
            for order in polynomial_orders:
                if model_name == "MLPNet":
                    model = MLPNet()
                else:
                    model = LeNet()

                a = order
                if activation == "LeakyRelu":
                    model.change_all_activations(leakyReluX(order=order, a=a))
                elif activation == "Sigmoid":
                    model.change_all_activations(sigmoidX(order=order, a=a))
                else:
                    model.change_all_activations(softplusX(order=order, a=a))
                
                print(f"\nTraining: model={model_name}, order={order}, a={a}, activation={activation}") 
                train_model(model, epochs, model_name, activation, order, time)
