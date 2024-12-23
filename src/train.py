import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from config import Config
from model import CIFAR10ResNet50

def get_datasets():
    """Prepare CIFAR-10 datasets with appropriate transforms."""
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
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATASET_ROOT,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATASET_ROOT,
        train=False,
        download=True,
        transform=transform_test
    )
    
    return train_dataset, test_dataset

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
            pbar.set_description(
                f'Loss: {running_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%'
            )
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

def main():
    # Set device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Get datasets and create dataloaders
    train_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model, criterion, and optimizer
    model = CIFAR10ResNet50(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch: {epoch+1}/{Config.NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")
        
        # Save checkpoint if best accuracy
        if test_acc > best_acc:
            print('Saving checkpoint...')
            state = {
                'model': model.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            torch.save(state, f'{Config.CHECKPOINT_DIR}/best_model.pth')
            best_acc = test_acc

if __name__ == '__main__':
    main()