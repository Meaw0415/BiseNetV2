from Model.net import BiSeNetV2
from utils.Camvid_dataset import CamVidDataset

from d2l import torch as d2l
from tqdm import tqdm
import pandas as pd
import monai
from torchcontrib.optim import SWA
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np


model = BiSeNetV2(num_classes=2)
 
train_path = "E:/DATABASE/DAVIS/ImageSets/480p/train.txt"
val_path   = "E:/DATABASE/DAVIS/ImageSets/480p/trainval.txt"
root_path  = 'E:/DATABASE/DAVIS'


train_dataset = CamVidDataset("E:/DATABASE/DAVIS/ImageSets/480p/train.txt",'E:/DATABASE/DAVIS')
val_dataset = CamVidDataset("E:/DATABASE/DAVIS/ImageSets/480p/trainval.txt",'E:/DATABASE/DAVIS')

 
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True,drop_last=True)
# training loop 100 epochs
epochs_num = 100
# 选用SGD优化器来训练

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
schedule = monai.optimizers.LinearLR(optimizer, end_lr=0.05, num_iter=int(epochs_num*0.75))
# 使用SWA优化 来提升SGD的效果

steps_per_epoch = int(len(train_loader.dataset) / train_loader.batch_size)
swa_start = int(epochs_num*0.75)
optimizer = SWA(optimizer, swa_start=swa_start*steps_per_epoch, swa_freq=steps_per_epoch, swa_lr=0.05)
 
# 损失函数选用多分类交叉熵损失函数
lossf = nn.CrossEntropyLoss(ignore_index=255)
 
 
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
 
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            output = net(X)
            pred = output[0]
            metric.add(d2l.accuracy(pred, y), d2l.size(y))
    return metric[0] / metric[1]
 
 
# 训练函数
def train_ch13(net, train_iter, test_iter, loss, optimizer, num_epochs, schedule, swa_start=swa_start, devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # 用来保存一些训练参数
 
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []
    lr_list = []
    
 
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (X, labels) in enumerate(train_iter):
            timer.start()
 
            if isinstance(X, list):
                X = [x.to(devices[0]) for x in X]
            else:
                X = X.to(devices[0])
            gt = labels.long().to(devices[0])
 
            net.train()
            optimizer.zero_grad()
            result = net(X)
            pred = result[0]
            seg_loss = loss(result[0], gt)
 
            aux_loss_1 = loss(result[1], gt)
            aux_loss_2 = loss(result[2], gt)
            aux_loss_3 = loss(result[3], gt)
            aux_loss_4 = loss(result[4], gt)
 
 
            loss_sum = seg_loss + 0.2*aux_loss_1 + 0.2*aux_loss_2 + 0.2*aux_loss_3 + 0.2*aux_loss_4
            l = loss_sum
            loss_sum.sum().backward()
            optimizer.step()
 
            acc = d2l.accuracy(pred, gt)
            metric.add(l, acc, labels.shape[0], labels.numel())
 
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[2], metric[1] / metric[3], None))
                
        if optimizer.state_dict()['param_groups'][0]['lr']>0.05:
            schedule.step()
 
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        
        if (epoch + 1) >= swa_start:
            if epoch == 0 or epoch % 5 == 5 - 1 or epoch == num_epochs - 1:
                # Batchnorm update
                optimizer._reset_lr_to_swa()
                optimizer.swap_swa_sgd()
                optimizer.bn_update(train_iter, net, device='cuda')
                test_acc = evaluate_accuracy_gpu(net, test_iter)
                optimizer.swap_swa_sgd()
        
        animator.add(epoch + 1, (None, None, test_acc))
 
        print(f"epoch {epoch+1}/{epochs_num} --- loss {metric[0] / metric[2]:.3f} --- train acc {metric[1] / metric[3]:.3f} --- test acc {test_acc:.3f} --- lr {optimizer.state_dict()['param_groups'][0]['lr']} --- cost time {timer.sum()}")
        
        #---------保存训练数据---------------
        df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch+1)
        time_list.append(timer.sum())
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        
        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df["lr"] = lr_list
        df['time'] = time_list
        
        df.to_excel("savefile/BiseNetv2_camvid.xlsx")
        #----------------保存模型------------------- 
        if np.mod(epoch+1, 5) == 0:
            torch.save(net.state_dict(), f'checkpoints/BiseNetv2_{epoch+1}.pth')
 
    # 保存下最后的model
    torch.save(net.state_dict(), f'checkpoints/BiseNetv2_last.pth')
 
train_ch13(model, train_loader, val_loader, lossf, optimizer, epochs_num, schedule=schedule)