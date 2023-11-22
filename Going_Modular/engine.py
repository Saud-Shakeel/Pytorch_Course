import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_step(model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
            dataloader: DataLoader, accuracy_fn, device: torch.device)-> Tuple[float, float]:
    
    model.train()
    trainLoss, trainAcc = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        train_preds = model(x)
        loss = loss_fn(train_preds, y)
        trainLoss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainAcc += accuracy_fn(y, train_preds.argmax(dim=1))
    
    trainLoss /= len(dataloader)
    trainAcc /= len(dataloader)

    return trainLoss, trainAcc

def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, accuracy_fn, 
            device: torch.device)-> Tuple[float, float]:
    model.eval()
    testLoss, testAcc = 0, 0
    with torch.inference_mode():
        for x_test, y_test in dataloader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            test_preds = model(x_test)
            loss = loss_fn(test_preds, y_test)
            testLoss += loss.item()
            testAcc += accuracy_fn(y_test, test_preds.argmax(dim=1))
        
        testLoss /= len(dataloader)
        testAcc /= len(dataloader)
    
    return testLoss, testAcc


def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, loss_fn: nn.Module, 
        optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device, epochs:int,
        writer: torch.utils.tensorboard.writer.SummaryWriter = None)-> Dict[str, float]:
    
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    
    for epoch in tqdm(range(epochs)):
        
        train_loss, train_acc = train_step(model= model, loss_fn= loss_fn, optimizer= optimizer, dataloader = train_dataloader, 
                accuracy_fn = accuracy_fn, device= device)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        
        test_loss, test_acc = test_step(model = model, dataloader= test_dataloader, loss_fn= loss_fn, accuracy_fn = accuracy_fn, 
                device= device)
        
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)


        print(f"Epoch: {epoch+1} | Train_loss: {train_loss:.4f} | Train_acc: {train_acc:.3f}% | Test_loss: {test_loss:.4f} | Test_acc: {test_acc:.3f}%")    
        
        if writer:
            writer.add_scalars(main_tag='Loss', tag_scalar_dict={'train_loss': train_loss, 'test_loss': test_loss},
                                global_step= epoch)
            writer.add_scalars(main_tag='Accuracy', tag_scalar_dict={'train_acc': train_acc, 'test_acc': test_acc},
                            global_step= epoch)
            writer.add_graph(model = model, input_to_model= torch.randn(32,3,224,224).to(device))
            writer.close()
        else:
            pass
        
    return results
