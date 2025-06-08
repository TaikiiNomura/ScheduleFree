import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(log_interval, model, device, train_loader, optimizer, epoch, is_schedulefree):
    model.train()
    if is_schedulefree:
        optimizer.train()
    
    running_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def test(model, device, test_loader, optimizer, is_schedulefree):
    model.eval()
    
    if is_schedulefree:
        optimizer.eval()
    
    test_loss = 0
    correct = 0
    # accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # test_losses.append(test_loss)
    # test_accuracies.append(accuracy)

    print(f'\nTest set: Average loss: {test_loss:.4f}')
    print(f'Accuracy: {correct}/{len(test_loader.dataset)}')
    print(f'({accuracy:.2f}%)\n')

    return test_loss, accuracy

# 初期状態での損失と精度を計算する関数
def evaluate_initial_state(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def run_schedulefree(
        model,
        optimizer,
        num_epochs,
        log_interval,
        device,
        train_loader,
        test_loader,
        optimizer_type,
        optimizer_name):

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    first_train_loss, first_train_accuracy = evaluate_initial_state(
        model,
        test_loader,
        device
    )
    
    test_losses.append(first_train_loss)
    test_accuracies.append(first_train_accuracy)

    for epoch in range(1, num_epochs + 1):
        
        if optimizer_type == 'schedulefree':
            is_schedulefree=True
        else:
            is_schedulefree=False
        
        epoch_train_loss, epoch_train_accuracy = train(
            log_interval,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            is_schedulefree=is_schedulefree
        )
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        epoch_test_loss, epoch_test_accuracy = test(
            model,
            device,
            test_loader,

            optimizer,
            is_schedulefree=is_schedulefree
        )
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

    return train_losses, train_accuracies, test_losses, test_accuracies