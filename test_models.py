import torch
import torch.nn as nn

from utils import *
from models import *


def test(net, loader, name, activation, order):
    """"Helper function to get the loss and accuracy of the best performing model on the
    unseen test set."""

     # Load the best performing model on the validation set
    checkpoint = torch.load(f'./checkpoint/{name}/{order}/ckpt_{activation}.pth')
    net.load_state_dict(checkpoint['net'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

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


    test_acc = 100 * correct/total
    return test_acc, test_loss


if __name__ == "__main__":

    # Get test data and create testloader
    testloader = get_test_loader(data_dir='./data', 
                                 batch_size=128,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=False)


    polynomial_orders = np.arange(2,11)
    model_names = ["MLPNet","ConvNet"]
    activations = ["LeakyRelu", "Sigmoid", "Softplus"]

    for model_name in model_names:
        for activation in activations:
            for order in polynomial_orders:
                if model_name == "MLPNet":
                    model = MLPNet()
                else:
                    model = ConvNet()

                a = order
                if activation == "LeakyRelu":
                    model.change_all_activations(leakyReluX(order=order, a=a))
                elif activation == "Sigmoid":
                    model.change_all_activations(sigmoidX(order=order, a=a))
                else:
                    model.change_all_activations(softplusX(order=order, a=a))

                test_acc, test_loss = test(model, testloader, model_name, activation, order)
                
                with open(f'checkpoint/{model_name}/{order}/stats_{activation}.txt', "w") as f:
                    f.write(f"Accuracy: {test_acc}\n")
                    f.write(f"Loss: {test_loss}\n")


    
